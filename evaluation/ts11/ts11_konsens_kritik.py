#!/usr/bin/env python3
"""
TS-11: Konsens und Kritik - fünf Varianten auf demselben Testset
  A  Primary allein
  B  Self-Correction (Primary prüft/korrigiert die eigene Antwort, KEIN externes Feedback)
  C  Primary + Critic (Critic aus anderer Familie bewertet; bei Score < Schwelle überarbeitet der Primary mit der Kritik)
  D  Homogenes Voting (3 x dasselbe Modell, Temperatur > 0, verschiedene Seeds)
  E  Heterogenes Voting (Qwen, Mistral, Gemma; Temperatur 0)

Fairness/Kosten: B, C und E nutzen die Antwort von A als Ausgangspunkt bzw. als Qwen-Stimme; deren Token und
Latenz werden den Varianten KUMULATIV angerechnet (Kosten = alle LLM-Aufrufe der Variante, sequenziell addiert,
bei Voting inkl. Embedding-Zeit). Judge-Aufrufe zählen nicht.
Mehrheitsentscheidung bei Freitext: Antworten gelten als "gleich", wenn ihre Cosine-Similarity (nomic-embed-text)
>= SIM_SCHWELLE ist; gewählt wird die Antwort der größten Übereinstimmungsgruppe (Mehrheit = mehr als die Hälfte der
gültigen Stimmen, sonst "kein Konsens" -> Antwort mit höchster Gesamt-Ähnlichkeit zu den anderen).
Metriken: Korrektheit (Judge 1-5), Halluzinationsrate (%) = Judge "nicht korrekt" ODER Faithfulness <= 2,
Recall/Precision des Critics (Referenz: Judge-Urteil über Antwort A), Cosine-Similarity der Endantworten
(Heatmap A-E; zusätzlich zur Referenzantwort), Token, Latenz.

WICHTIG (PowerShell, vor 'ollama serve'): alle Modelle gleichzeitig im RAM halten, sonst enthält die Latenz Ladezeiten:
    $env:OLLAMA_MAX_LOADED_MODELS = "5"; $env:OLLAMA_KEEP_ALIVE = "2h"
Laufzeit auf CPU: ca. 8-10 LLM-Aufrufe je Frage und Wiederholung -> zuerst  --limit 3 --repetitions 1  testen.

Beispiel:  python ts11_konsens_kritik.py --repetitions 1
Trockenlauf: python ts11_konsens_kritik.py --dry-run --out .\\dryrun
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

import eval_utils as eu  # noqa: E402
import eval_utils_extension as ex  # noqa: E402
import eval_utils_phase_d as pd_  # noqa: E402
from llm_client import query_model  # noqa: E402

# =========================== KONFIGURATION ================================== #
SYSTEM_PROMPT = ("Du bist ein hilfreicher, präziser Assistent. Antworte auf Deutsch, sachlich und knapp. "
                 "Wenn ein Kontext angegeben ist, stütze dich ausschließlich darauf und erfinde nichts dazu. "
                 "Wenn die Antwort nicht im Kontext steht, sage das ausdrücklich.")
MAX_TOKENS = 250
CRITIC_MAX_TOKENS = 400
HETERO_MODELS = ["mistral_small_4", "gemma3"]      # + Primary (qwen3) = heterogenes Trio
HOMO_TEMPERATURE, HOMO_SEEDS = 0.7, [11, 22, 33]
SIM_SCHWELLE = 0.85                                 # Cosine-Schwelle für "gleiche Antwort" - ANNAHME, ggf. kalibrieren
CRITIC_AKZEPTANZ = 4                                # Critic-Score >= 4 = Antwort akzeptiert (wie TS-04)
LABELS = {"A": "A\nPrimary allein", "B": "B\nSelf-Correction", "C": "C\nPrimary + Critic",
          "D": "D\nVoting homogen", "E": "E\nVoting heterogen"}
# Nutzwert/Radar: Ankerwerte (worst -> 0, best -> 1) für CPU-Betrieb, Gewichte Summe 1.0 - ANNAHMEN
NUTZWERT = {"korrektheit_mean":       {"weight": 0.40, "worst": 1.0,      "best": 5.0},
            "latenz_ms_mean":         {"weight": 0.20, "worst": 300000.0, "best": 20000.0},
            "tokens_mean":            {"weight": 0.15, "worst": 3000.0,   "best": 200.0},
            "halluzinationsrate_pct": {"weight": 0.25, "worst": 50.0,     "best": 0.0}}
RADAR = {"korrektheit_mean": "Korrektheit", "latenz_ms_mean": "Geschwindigkeit\n(Latenz)",
         "tokens_mean": "Token-Sparsamkeit", "halluzinationsrate_pct": "Halluzinations-\nfreiheit"}

PROMETHEUS = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[Is the response factually correct, free of hallucinations and (if a context is given) fully supported by that context, while answering the question?]
Score 1: The response is factually wrong, contains clear hallucinations, or contradicts the context.
Score 2: The response is mostly incorrect and contains significant factual errors or unsupported claims.
Score 3: The response is partially correct but has notable errors, omissions or unsupported details.
Score 4: The response is correct with only minor omissions or imprecise wording and contains no hallucinations.
Score 5: The response is fully correct, complete and fully supported, with no hallucinations.

###Feedback:"""
# ============================================================================ #
RESULT_RE = re.compile(r"\[RESULT\]\s*\(?\s*([1-5])\b")


def new_cost():
    return {"tokens": 0, "latency_ms": 0.0, "calls": 0, "errors": 0}


def add_cost(c, res):
    c["tokens"] += int(res.get("token_count") or 0)
    lat = res.get("total_latency_ms")
    c["latency_ms"] += float(lat) if lat == lat and lat is not None else 0.0
    c["calls"] += 1
    c["errors"] += 1 if res.get("error") else 0


def call(key, prompt, cost, temperature=0.0, max_tokens=None, system=SYSTEM_PROMPT):
    max_tokens = max_tokens or MAX_TOKENS
    try:
        res = query_model(key, prompt, stream=False, temperature=temperature, system_prompt=system,
                          max_tokens=max_tokens, retries=1)
    except Exception as exc:  # noqa: BLE001
        res = {"response_text": "", "error": f"{type(exc).__name__}: {exc}", "total_latency_ms": np.nan, "token_count": 0}
    add_cost(cost, res)
    if res.get("error"):
        eu.log.warning("LLM-Fehler | %s | %s", key, res["error"])
    return res


def vote(answers, cost):
    """Mehrheitsentscheidung über semantische Ähnlichkeit. Rückgabe: (Antwort, Konsens?, Stimmen, Ø-Ähnlichkeit)."""
    valid = [i for i, a in enumerate(answers) if a.strip()]
    if len(valid) < 2:
        return (answers[valid[0]] if valid else ""), False, len(valid), np.nan
    try:
        vecs, ms = pd_.embed_texts([f"search_document: {answers[i]}" for i in valid])
    except Exception as exc:  # noqa: BLE001
        eu.log.warning("Embedding-Fehler beim Voting: %s", exc)
        return answers[valid[0]], False, 1, np.nan
    cost["latency_ms"] += ms
    S = pd_.cosine_matrix(vecs, vecs)
    counts = (S >= SIM_SCHWELLE).sum(axis=1)
    best = max(range(len(valid)), key=lambda k: (counts[k], S[k].sum()))
    intra = float(S[np.triu_indices(len(valid), 1)].mean())
    return answers[valid[best]], bool(counts[best] > len(valid) / 2), int(counts[best]), intra


def merge(*costs):
    out = new_cost()
    for c in costs:
        for k in out:
            out[k] += c[k]
    return out


def process(q, rep, sel, cfg):
    """Alle Varianten für eine Frage/Wiederholung. Rückgabe: Record (JSON-serialisierbar)."""
    prompt = eu.build_prompt(q)
    V = {}
    cA = new_cost()
    a = call(cfg["primary"], prompt, cA)["response_text"]
    V["A"] = {"answer": a, "cost": cA, "extra": {}}
    if "B" in sel:
        cB = new_cost()
        pb = (f"{prompt}\n\nDeine bisherige Antwort:\n{a}\n\nPrüfe diese Antwort kritisch auf sachliche Fehler und erfundene "
              "Angaben (ohne externe Hilfe). Gib NUR die endgültige, gegebenenfalls korrigierte Antwort aus.")
        b = call(cfg["primary"], pb, cB)["response_text"] if a.strip() else ""
        V["B"] = {"answer": b or a, "cost": merge(cA, cB), "extra": {"geaendert": bool(b and b.strip() != a.strip())}}
    if "C" in sel:
        cC = new_cost()
        instr = prompt if not q["context"] else f"Context:\n{q['context']}\n\nQuestion: {q['question']}"
        crit = call(cfg["critic"], PROMETHEUS.format(instruction=instr, response=a,
                                                    reference=q["reference_answer"] or "(no reference answer available)"),
                    cC, max_tokens=CRITIC_MAX_TOKENS, system=None)["response_text"] if a.strip() else ""
        m = RESULT_RE.search(crit)
        score = int(m.group(1)) if m else None
        reject = score is not None and score < CRITIC_AKZEPTANZ
        final = a
        if reject:
            feedback = crit.split("[RESULT]")[0].replace("Feedback:", "").strip()
            pr = (f"{prompt}\n\nBisherige Antwort:\n{a}\n\nKritik eines unabhängigen Prüfers:\n{feedback}\n\n"
                  "Überarbeite die Antwort unter Berücksichtigung der Kritik, ohne neue Fakten zu erfinden. "
                  "Gib NUR die endgültige Antwort aus.")
            final = call(cfg["primary"], pr, cC)["response_text"] or a
        if score is None and a.strip():
            eu.log.warning("Critic ohne [RESULT]-Score | %s | Rep %d", q["id"], rep)
        V["C"] = {"answer": final, "cost": merge(cA, cC), "extra": {"critic_score": score, "critic_ablehnung": reject,
                                                                    "critic_gueltig": score is not None, "ueberarbeitet": reject}}
    if "D" in sel:
        cD = new_cost()
        ans = [call(f"ts11_hom_{i}", prompt, cD, temperature=HOMO_TEMPERATURE)["response_text"] for i in range(len(HOMO_SEEDS))]
        win, cons, votes, intra = vote(ans, cD)
        V["D"] = {"answer": win, "cost": cD, "extra": {"konsens": cons, "stimmen": votes, "intra_aehnlichkeit": intra}}
    if "E" in sel:
        cE = new_cost()
        ans = [a] + [call(k, prompt, cE)["response_text"] for k in cfg["hetero"]]
        win, cons, votes, intra = vote(ans, cE)
        V["E"] = {"answer": win, "cost": merge(cA, cE), "extra": {"konsens": cons, "stimmen": votes, "intra_aehnlichkeit": intra}}
    # Cosine-Similarity der Endantworten (Heatmap) und zur Referenz - nicht in den Kosten enthalten
    keys = [k for k in "ABCDE" if k in sel]
    texts = [V[k]["answer"] or " " for k in keys] + [q["reference_answer"] or " "]
    try:
        vecs, _ = pd_.embed_texts([f"search_document: {t}" for t in texts])
        S = pd_.cosine_matrix(vecs, vecs)
        sim, cref = S[: len(keys), : len(keys)].tolist(), S[: len(keys), -1].tolist()
    except Exception as exc:  # noqa: BLE001
        eu.log.warning("Embedding-Fehler (Similarity): %s", exc)
        sim, cref = None, [np.nan] * len(keys)
    for k, c in zip(keys, cref):
        V[k]["cos_ref"] = float(c)
    return {"question_id": q["id"], "repetition": rep, "keys": keys, "sim_matrix": sim, "variants": V}


def main():
    ex.fix_console()
    parser = ex.common_parser(__doc__)
    parser.add_argument("--variants", nargs="+", default=list("ABCDE"), choices=list("ABCDE"), help="Nur diese Varianten")
    parser.add_argument("--primary", default="qwen3")
    parser.add_argument("--critic", default="prometheus2", help="Critic (andere Familie als Primary)")
    parser.add_argument("--resume", action="store_true", help="Bereits berechnete Fragen/Wiederholungen überspringen")
    args = parser.parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    sel = set(args.variants) | {"A"}
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    cfg = {"primary": args.primary, "critic": args.critic, "hetero": HETERO_MODELS}
    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens or MAX_TOKENS

    if args.dry_run:
        ex.setup_dry_run(args.judge_model)
        pd_.set_dry_run(True)
        for k, fam, resp in ((args.primary, "qwen", "Neun Bundesländer."), ("mistral_small_4", "mistral", "Neun Bundesländer."),
                             ("gemma3", "google", "Zehn Bundesländer."), (args.critic, "mistral", "Feedback: unklar. [RESULT] 3")):
            ex.register_model(k, {"provider": "mock", "family": fam, "mock_response": resp, "mock_delay_s": 0.02})
        for i, resp in enumerate(["Neun Bundesländer.", "Neun Bundesländer.", "Zehn Bundesländer."]):
            ex.register_model(f"ts11_hom_{i}", {"provider": "mock", "family": "qwen", "mock_response": resp, "mock_delay_s": 0.02})
    else:
        for i, seed in enumerate(HOMO_SEEDS):
            pd_.register_variant(f"ts11_hom_{i}", args.primary, seed=seed)
    if eu.model_family(args.critic) == eu.model_family(args.primary):
        eu.log.warning("Critic %s gehört zur selben Familie wie Primary %s!", args.critic, args.primary)
    models = {args.primary, args.critic, *HETERO_MODELS}
    print(f"TS-11 | {len(questions)} Fragen | Varianten: {sorted(sel)} | Wiederholungen: {args.repetitions}")
    pd_.embed_texts(["search_document: Aufwärmen"])
    for k in sorted(models | {"ts11_hom_0"}):
        w = query_model(k, "Antworte mit einem Wort: Hallo.", stream=False, max_tokens=8, retries=0)
        if w["error"]:
            raise SystemExit(f"Warm-up für '{k}' fehlgeschlagen: {w['error']}")

    # ---- Phase 1: Generierung (ohne Judge, damit Modellwechsel die Latenz nicht verfälschen) ----
    jsonl = out / "responses_ts11.jsonl"
    records = eu.load_jsonl(jsonl) if args.resume else []
    if jsonl.exists() and not args.resume:
        jsonl.unlink()
    done = {(r["question_id"], r["repetition"]) for r in records}
    todo = [(q, rep) for rep in range(1, args.repetitions + 1) for q in questions if (q["id"], rep) not in done]
    for q, rep in tqdm(todo, desc="Generierung", unit="Frage"):
        try:
            rec = process(q, rep, sel, cfg)
            records.append(rec)
            eu.append_jsonl(jsonl, rec)
        except Exception:  # noqa: BLE001
            eu.log.exception("Fehler bei Frage %s (Rep %d) - übersprungen", q["id"], rep)

    # ---- Phase 2: Judge ----
    qmap = {q["id"]: q for q in questions}
    cache = {}
    todo_j = [(r, k) for r in records for k in r["keys"] if "antwortqualitaet" not in r["variants"][k] and r["question_id"] in qmap]
    for r, k in tqdm(todo_j, desc="Judge", unit="Bewertung", disable=args.no_judge or not todo_j):
        v, q = r["variants"][k], qmap[r["question_id"]]
        if args.no_judge or not v["answer"].strip():
            v.update({"antwortqualitaet": None, "deutschqualitaet": None, "faithfulness": None, "korrekt": None})
            continue
        key = (q["id"], v["answer"])
        if key not in cache:
            cache[key] = eu.judge_answer(args.judge_model, q["question"], v["answer"], q["reference_answer"], q["context"])
        v.update({k2: cache[key][k2] for k2 in ("antwortqualitaet", "deutschqualitaet", "faithfulness", "korrekt")})
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records), encoding="utf-8")

    # ---- Details ----
    rows = []
    for r in records:
        for k in r["keys"]:
            v = r["variants"][k]
            hall = np.nan
            if v.get("antwortqualitaet") is not None:
                hall = float(v.get("korrekt") is False or (v.get("faithfulness") is not None and v["faithfulness"] <= 2))
            rows.append({"variante": k, "question_id": r["question_id"], "repetition": r["repetition"], "antwort": v["answer"],
                         "gueltig": bool(v["answer"].strip()), "tokens": v["cost"]["tokens"], "latenz_ms": v["cost"]["latency_ms"],
                         "llm_aufrufe": v["cost"]["calls"], "llm_fehler": v["cost"]["errors"], "cos_ref": v.get("cos_ref", np.nan),
                         "korrektheit": v.get("antwortqualitaet"), "deutschqualitaet": v.get("deutschqualitaet"),
                         "faithfulness": v.get("faithfulness"), "korrekt": v.get("korrekt"), "halluzination": hall, **v["extra"]})
    dfa = pd.DataFrame(rows)
    if dfa.empty:
        print("Keine Daten - Abbruch.")
        return
    for c in ("korrektheit", "deutschqualitaet", "faithfulness"):
        dfa[c] = pd.to_numeric(dfa[c], errors="coerce")
    dfa["korrekt_num"] = pd.to_numeric(dfa["korrekt"].map({True: 1.0, False: 0.0}), errors="coerce")
    df = dfa[dfa["variante"].isin(args.variants)].copy()
    eu.write_csv(df.drop(columns=["korrekt_num"]), out / "results_details.csv")

    # ---- Zusammenfassung je Variante ----
    summ = []
    for k in [x for x in "ABCDE" if x in args.variants]:
        g = df[df["variante"] == k]
        gv = g[g["gueltig"]]
        facts = gv["korrekt_num"].dropna()
        summ.append({"variante": k, "bezeichnung": LABELS[k].replace("\n", " "), "label": LABELS[k], "n": len(g),
                     "ungueltig_pct": 100 * (1 - len(gv) / len(g)), "korrektheit_mean": gv["korrektheit"].mean(),
                     "korrektheit_std": gv["korrektheit"].std(), "deutschqualitaet_mean": gv["deutschqualitaet"].mean(),
                     "halluzinationsrate_pct": 100 * gv["halluzination"].mean(), "faktentreffer_pct": 100 * facts.mean() if len(facts) else np.nan,
                     "cos_zur_referenz_mean": gv["cos_ref"].mean(), "tokens_mean": g["tokens"].mean(), "tokens_std": g["tokens"].std(),
                     "latenz_ms_mean": g["latenz_ms"].mean(), "latenz_ms_median": g["latenz_ms"].median(),
                     "latenz_ms_p95": pd_.pct(g["latenz_ms"], 95), "llm_aufrufe_mean": g["llm_aufrufe"].mean(),
                     "konsens_quote_pct": 100 * g["konsens"].mean() if "konsens" in g and g["konsens"].notna().any() else np.nan,
                     "intra_aehnlichkeit_mean": g["intra_aehnlichkeit"].mean() if "intra_aehnlichkeit" in g else np.nan})
    summary = pd.DataFrame(summ)
    parts, nw = {}, []
    for _, r in summary.iterrows():
        n_, _, p_ = eu.nutzwert(r, NUTZWERT)
        nw.append(n_)
        parts[r["label"]] = [p_[c] for c in RADAR]
    summary["nutzwert"] = nw
    eu.write_csv(summary.drop(columns=["label"]).round(3), out / "results_summary.csv")

    # ---- Critic: Recall / Precision (Referenz: Judge-Urteil über Antwort A) ----
    crit_row = None
    if "C" in args.variants:
        c = dfa[dfa["variante"] == "C"].merge(dfa[dfa["variante"] == "A"][["question_id", "repetition", "korrekt_num", "korrektheit"]],
                                              on=["question_id", "repetition"], suffixes=("", "_a"))
        c = c[c["critic_gueltig"].fillna(False).astype(bool) & c["korrekt_num_a"].notna()]
        fehler, ablehn = (c["korrekt_num_a"] == 0), c["critic_ablehnung"].astype(bool)
        tp, fp, fn, tn = int((ablehn & fehler).sum()), int((ablehn & ~fehler).sum()), int((~ablehn & fehler).sum()), int((~ablehn & ~fehler).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec_ = tp / (tp + fn) if tp + fn else np.nan
        rej = c[ablehn]
        crit_row = {"critic": args.critic, "n_bewertet": len(c), "antworten_a_fehlerhaft": int(fehler.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "precision_pct": 100 * prec, "recall_pct": 100 * rec_,
                    "f1_pct": 100 * 2 * prec * rec_ / (prec + rec_) if (prec == prec and rec_ == rec_ and prec + rec_ > 0) else np.nan,
                    "genauigkeit_pct": 100 * (tp + tn) / len(c) if len(c) else np.nan,
                    "ungueltige_critic_ausgaben": int((~dfa[dfa["variante"] == "C"]["critic_gueltig"].fillna(False).astype(bool)).sum()),
                    "korrektheit_gewinn_durch_ueberarbeitung": (rej["korrektheit"] - rej["korrektheit_a"]).mean() if len(rej) else np.nan}
        eu.write_csv(pd.DataFrame([crit_row]).round(3), out / "results_critic.csv")

    # ---- Cosine-Similarity-Matrix (Mittel über Fragen/Wiederholungen) ----
    keys = [k for k in "ABCDE" if k in args.variants]
    mats = []
    for r in records:
        if r["sim_matrix"] is not None and r["question_id"] in set(df["question_id"]):
            M = np.array(r["sim_matrix"])
            idx = [r["keys"].index(k) for k in keys if k in r["keys"]]
            mats.append(M[np.ix_(idx, idx)])
    simdf = pd.DataFrame(np.mean(mats, axis=0), index=[LABELS[k] for k in keys], columns=[LABELS[k] for k in keys]) if mats else None
    if simdf is not None:
        out_sim = simdf.copy()
        out_sim.index, out_sim.columns = [i.replace("\n", " ") for i in simdf.index], [i.replace("\n", " ") for i in simdf.columns]
        eu.write_csv(out_sim.round(4).reset_index().rename(columns={"index": "variante"}), out / "results_cosine_matrix.csv")

    # ---- Diagramme ----
    eu.plot_radar(parts, list(RADAR.values()), "TS-11: Vergleich der fünf Varianten (außen = besser)", out / "diagramm_radar.png")
    if simdf is not None:
        eu.plot_heatmap(simdf, "TS-11: Wie ähnlich sind die Endantworten der Varianten? (Cosine-Similarity)",
                        out / "diagramm_heatmap_cosine.png", "Cosine-Similarity (1 = inhaltlich gleich)", fmt="{:.2f}",
                        cmap="YlGnBu", vmin=float(max(0, np.nanmin(simdf.values) - 0.05)), vmax=1.0)
    eu.plot_grouped_bars(summary, "label", {"Korrektheit (1–5)": "korrektheit_mean"}, "TS-11: Korrektheit der Endantworten",
                         "Bewertung (1 = falsch, 5 = sehr gut)", out / "diagramm_korrektheit.png",
                         err_cols={"Korrektheit (1–5)": "korrektheit_std"}, ylim=(0, 5.8))
    eu.plot_grouped_bars(summary, "label", {"Halluzinationsrate (%)": "halluzinationsrate_pct"}, "TS-11: Halluzinationsrate",
                         "Anteil der Antworten (%, niedriger = besser)", out / "diagramm_halluzination.png", ylim=(0, 115))
    eu.plot_box({LABELS[k]: (df[df["variante"] == k]["latenz_ms"] / 1000).tolist() for k in keys}, "Gesamtlatenz je Frage (s)",
                "TS-11: Antwortzeit (alle LLM-Aufrufe der Variante)", out / "diagramm_latenz_boxplot.png")
    eu.plot_single_bars(summary["label"].tolist(), summary["tokens_mean"], "TS-11: Token-Verbrauch je Frage",
                        "Ausgabe-Token je Frage (niedriger = besser)", out / "diagramm_token.png", fmt="{:.0f}")
    if crit_row and crit_row["n_bewertet"]:
        eu.plot_grouped_bars(pd.DataFrame([crit_row]).assign(c=f"Critic ({args.critic})"), "c",
                             {"Precision": "precision_pct", "Recall": "recall_pct", "Genauigkeit": "genauigkeit_pct"},
                             "TS-11: Wie gut erkennt der Critic fehlerhafte Antworten?", "Anteil (%)",
                             out / "diagramm_critic.png", ylim=(0, 118))

    pd.set_option("display.width", 220)
    print("\n=== Ergebnis TS-11 ===")
    print(summary[["bezeichnung", "korrektheit_mean", "halluzinationsrate_pct", "cos_zur_referenz_mean", "tokens_mean",
                   "latenz_ms_median", "nutzwert"]].round(2).to_string(index=False))
    if crit_row:
        print(f"\nCritic {args.critic}: Precision {crit_row['precision_pct']:.0f} %, Recall {crit_row['recall_pct']:.0f} % "
              f"(n={crit_row['n_bewertet']}, fehlerhafte A-Antworten: {crit_row['antworten_a_fehlerhaft']})")
    print(f"Dateien in: {out}")


if __name__ == "__main__":
    main()