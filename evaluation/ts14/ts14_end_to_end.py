#!/usr/bin/env python3
"""
TS-14: Ende-zu-Ende-Test - KAISER-Gesamtstack (Variante B) vs. Primary allein (Variante A)

  A  Primary allein:  qwen3 beantwortet die Frage (allgemeiner System-Prompt).
  B  Gesamtstack:     1. Router (qwen3) wählt die Kategorie -> kategoriespezifischer Hinweis im System-Prompt
                      2. Primary (qwen3) antwortet            3. Secondary (gemma3) antwortet
                      4. Critic (prometheus2) bewertet die Antwort des Primary (Score 1-5, akzeptiert ab ACCEPT_THRESHOLD)
                         -> akzeptiert: finale Antwort = Primary; abgelehnt: finale Antwort = Secondary (Rückfallebene)
                      5. Judge (llama3.1) bewertet die finale Antwort (1-5)
  Der Judge bewertet auch Variante A (gleicher Maßstab für beide). Alle Stufen laufen sequenziell (auf reiner CPU bringt
  Parallelisierung laut TS-10 nichts).

Metriken je Variante:
  * Latenz (Median, P95, Mittel): A = ein Aufruf; B = Router+Primary+Secondary+Critic (Produktivbetrieb, ohne Judge).
    Zusätzlich B inkl. Judge (latenz_inkl_judge), weil der Judge laut Architektur Stufe 5 ist.
  * Token-Verbrauch = Summe der AUSGABE-Token aller LLM-Aufrufe (query_model meldet nur diese).
  * RAM-Peak der Ollama-Prozesse (psutil, absolut, während der Generierung; Judge läuft danach und zählt nicht)
  * Antwortqualität 1-5 (Judge), Deutschqualität, Trefferquote (Judge: korrekt)
  * Fehlerrate = Anteil der Läufe mit Aufruffehler oder leerer finaler Antwort
Zusatz (B): Routing-Genauigkeit, Critic-Ablehnungsquote, Anteil Secondary als finale Antwort.

WICHTIG (PowerShell, vor 'ollama serve'): damit keine Ladezeiten in der Latenz landen, müssen alle drei Modelle gleichzeitig
im RAM bleiben:   $env:OLLAMA_MAX_LOADED_MODELS = "3"; $env:OLLAMA_KEEP_ALIVE = "2h"
Laufzeit auf CPU: ca. 2-3 Minuten je Frage und Wiederholung für B -> zuerst  --limit 2 --repetitions 1  testen; --resume setzt fort.

Beispiel:  python ts14_end_to_end.py --repetitions 3
Trockenlauf: python ts14_end_to_end.py --dry-run --out .\\dryrun
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

import eval_utils as eu  # noqa: E402
import eval_utils_extension as ex  # noqa: E402
import llm_client  # noqa: E402
from llm_client import query_model  # noqa: E402

# =========================== KONFIGURATION ================================== #
SYSTEM_PROMPT = ("Du bist ein hilfreicher, präziser Assistent. Antworte auf Deutsch, sachlich und knapp. "
                 "Wenn ein Kontext angegeben ist, stütze dich ausschließlich darauf und erfinde nichts dazu. "
                 "Wenn die Antwort nicht im Kontext steht, sage das ausdrücklich.")
MAX_TOKENS, CRITIC_MAX_TOKENS, ROUTER_MAX_TOKENS = 300, 400, 16
ACCEPT_THRESHOLD = 4                         # Critic-Score >= 4 = Primary-Antwort akzeptiert
PROCESS_NAMES = ["ollama"]
ROUTE_DESCRIPTIONS = {"fakten": "kurze Faktenfrage (Zahl, Jahr, Name, Ort)",
                      "logik": "Rechen- oder Logikaufgabe",
                      "erklaerung": "Bitte um Erklärung, Vergleich oder Zusammenfassung",
                      "rag_kontext": "Frage zu einer internen Regelung oder einem vorliegenden Dokument"}
ROUTE_HINTS = {"fakten": "Nenne die konkrete Tatsache in einem Satz.",
               "logik": "Rechne kurz Schritt für Schritt und nenne das Ergebnis.",
               "erklaerung": "Erkläre in höchstens drei Sätzen.",
               "rag_kontext": "Stütze dich ausschließlich auf den gegebenen Kontext."}

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
NAN = float("nan")


def call(key, prompt, system=None, max_tokens=MAX_TOKENS):
    try:
        return query_model(key, prompt, stream=False, temperature=0.0, system_prompt=system, max_tokens=max_tokens, retries=1)
    except Exception as exc:  # noqa: BLE001 - Lauf darf nie abbrechen
        return {"response_text": "", "error": f"{type(exc).__name__}: {exc}", "total_latency_ms": NAN, "token_count": 0}


def stage(rec, name, res):
    rec[f"{name}_ms"] = res["total_latency_ms"]
    rec[f"{name}_tokens"] = res["token_count"]
    if res.get("error"):
        rec["fehler_text"] = (rec.get("fehler_text", "") + f"{name}: {res['error']}; ")[:400]


def base_record(variante, q, rep):
    return {"variante": variante, "question_id": q["id"], "category": q["category"], "repetition": rep,
            "question": q["question"], "reference_answer": q["reference_answer"], "context": q["context"],
            "fehler_text": ""}


def run_a(q, rep, cfg):
    rec = base_record("A", q, rep)
    with ex.RAMMonitor(PROCESS_NAMES, interval=0.25) as mon:
        t0 = time.perf_counter()
        r = call(cfg["primary"], eu.build_prompt(q), SYSTEM_PROMPT)
        wall = (time.perf_counter() - t0) * 1000
    stage(rec, "primary", r)
    rec.update({"final_answer": r["response_text"], "final_quelle": "primary", "latenz_ms": wall,
                "tokens": r["token_count"], **mon.meta})
    return rec


def run_b(q, rep, cfg, routes):
    rec = base_record("B", q, rep)
    with ex.RAMMonitor(PROCESS_NAMES, interval=0.25) as mon:
        t0 = time.perf_counter()
        # 1. Router
        listing = "\n".join(f"- {r}: {ROUTE_DESCRIPTIONS.get(r, r)}" for r in routes)
        rr = call(cfg["router"], f"Ordne die Frage genau einer Kategorie zu.\nKategorien:\n{listing}\n\nFrage: {q['question']}\n\n"
                                 "Antworte NUR mit dem Kategorienamen.", None, ROUTER_MAX_TOKENS)
        text = rr["response_text"].lower()
        hits = [(text.find(x), x) for x in routes if x in text]
        kat = min(hits)[1] if hits else None
        stage(rec, "router", rr)
        rec.update({"router_kategorie": kat, "router_korrekt": kat == q["category"]})
        system_b = SYSTEM_PROMPT + " " + ROUTE_HINTS.get(kat, "")
        prompt = eu.build_prompt(q)
        # 2. Primary, 3. Secondary
        rp = call(cfg["primary"], prompt, system_b)
        stage(rec, "primary", rp)
        rs = call(cfg["secondary"], prompt, system_b)
        stage(rec, "secondary", rs)
        # 4. Critic bewertet die Primary-Antwort
        score = None
        if rp["response_text"].strip():
            instr = prompt if not q["context"] else f"Context:\n{q['context']}\n\nQuestion: {q['question']}"
            rc = call(cfg["critic"], PROMETHEUS.format(instruction=instr, response=rp["response_text"],
                                                       reference=q["reference_answer"] or "(no reference answer available)"),
                      None, CRITIC_MAX_TOKENS)
            stage(rec, "critic", rc)
            m = RESULT_RE.search(rc["response_text"])
            score = int(m.group(1)) if m else None
            if score is None:
                rec["critic_hinweis"] = "kein [RESULT]-Score"
        wall = (time.perf_counter() - t0) * 1000
    reject = (score is not None and score < ACCEPT_THRESHOLD) or not rp["response_text"].strip()
    final, quelle = (rs["response_text"], "secondary") if reject else (rp["response_text"], "primary")
    if reject and not rs["response_text"].strip():
        final, quelle = rp["response_text"], "primary"          # Rückfallebene leer -> Primary behalten
    rec.update({"critic_score": score, "critic_gueltig": score is not None, "critic_ablehnung": bool(reject),
                "final_answer": final, "final_quelle": quelle, "latenz_ms": wall,
                "tokens": sum(rec.get(f"{s}_tokens", 0) or 0 for s in ("router", "primary", "secondary", "critic")), **mon.meta})
    return rec


class CallRecorder:
    """Zeichnet Latenz/Token der Judge-Aufrufe auf (eu.judge_answer verwirft sie sonst), ohne dessen Prompt zu duplizieren."""

    def __enter__(self):
        self.calls, self._orig = [], getattr(eu, "query_model", None)
        if self._orig:
            def wrapped(*a, **k):
                r = self._orig(*a, **k)
                self.calls.append(r)
                return r
            eu.query_model = wrapped
        return self

    def __exit__(self, *exc):
        if self._orig:
            eu.query_model = self._orig

    @property
    def ms(self):
        return sum(c["total_latency_ms"] for c in self.calls if c["total_latency_ms"] == c["total_latency_ms"]) if self.calls else NAN

    @property
    def tokens(self):
        return sum(c["token_count"] for c in self.calls) if self.calls else NAN


def loaded_models(host):
    try:
        return [m["name"] for m in requests.get(host + "/api/ps", timeout=5).json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def warm(models, host, dry):
    for m in models:
        w = query_model(m, "Antworte mit einem Wort: Hallo.", stream=False, max_tokens=8, retries=0)
        if w["error"]:
            raise SystemExit(f"Warm-up für '{m}' fehlgeschlagen: {w['error']}")
    if not dry:
        names = loaded_models(host)
        for m in models:
            mid = llm_client.get_model_spec(m).get("model_id", m)
            if not any(mid == n or f"{mid}:latest" == n or n.startswith(mid) for n in names):
                eu.log.warning("Modell %s ist laut /api/ps nicht geladen - OLLAMA_MAX_LOADED_MODELS prüfen (Latenz enthält sonst Ladezeit).", m)


def main():
    ex.fix_console()
    parser = ex.common_parser(__doc__)
    parser.add_argument("--secondary", default="gemma3")
    parser.add_argument("--critic", default="prometheus2")
    parser.add_argument("--resume", action="store_true", help="Bereits berechnete Läufe überspringen")
    args = parser.parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    routes = list(ROUTE_DESCRIPTIONS) + sorted({q["category"] for q in questions} - set(ROUTE_DESCRIPTIONS))
    cfg = {"primary": args.model, "secondary": args.secondary, "critic": args.critic, "router": args.model}
    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens or MAX_TOKENS
    if args.dry_run:
        ex.setup_dry_run(args.judge_model)
        for k, fam, resp in ((args.model, "qwen", "Neun Bundesländer."), (args.secondary, "google", "Zehn Bundesländer."),
                             (args.critic, "mistral", "Feedback: unklar. [RESULT] 3")):
            ex.register_model(k, {"provider": "mock", "family": fam, "mock_response": resp, "mock_delay_s": 0.03})
        ex.register_model("ts14_router", {"provider": "mock", "family": "qwen", "mock_response": "fakten", "mock_delay_s": 0.02})
        cfg["router"] = "ts14_router"
        host = ""
    else:
        host = ex.ollama_host(llm_client.get_model_spec(args.model).get("base_url", "http://localhost:11434/v1"))
    print(f"TS-14 | {len(questions)} Fragen | Wiederholungen: {args.repetitions} | Primary/Router: {args.model} | "
          f"Secondary: {args.secondary} | Critic: {args.critic} | Judge: {args.judge_model}")

    # ---- Phase 1: Generierung (Judge noch nicht geladen -> RAM-Peak sauber) ----
    jsonl = out / "responses_ts14.jsonl"
    records = eu.load_jsonl(jsonl) if args.resume else []
    if jsonl.exists() and not args.resume:
        jsonl.unlink()
    done = {(r["variante"], r["question_id"], r["repetition"]) for r in records}
    for variante, models, fn in (("A", [cfg["primary"]], lambda q, r: run_a(q, r, cfg)),
                                 ("B", [cfg["router"], cfg["primary"], cfg["secondary"], cfg["critic"]], lambda q, r: run_b(q, r, cfg, routes))):
        todo = [(q, rep) for rep in range(1, args.repetitions + 1) for q in questions if (variante, q["id"], rep) not in done]
        if not todo:
            continue
        if not args.dry_run:
            ex.ollama_unload_all(host)              # gleiche Startbedingungen: nur die Modelle der Variante geladen
        warm(sorted(set(models)), host, args.dry_run)
        for q, rep in tqdm(todo, desc=f"Variante {variante}", unit="Frage"):
            try:
                rec = fn(q, rep)
            except Exception as exc:  # noqa: BLE001
                eu.log.exception("Fehler | %s | %s", variante, q["id"])
                rec = {**base_record(variante, q, rep), "final_answer": "", "final_quelle": "", "latenz_ms": NAN, "tokens": 0,
                       "fehler_text": f"{type(exc).__name__}: {exc}"}
            rec["fehler"] = bool(rec["fehler_text"].strip()) or not rec["final_answer"].strip()   # Aufruffehler oder leere Antwort
            records.append(rec)
            eu.append_jsonl(jsonl, rec)

    # ---- Phase 2: Judge (A mit Cache, B immer echter Aufruf, damit Latenz/Token der Stufe 5 gemessen werden) ----
    cache = {}
    todo_j = [r for r in records if "antwortqualitaet" not in r and r["final_answer"].strip() and not args.no_judge]
    for r in tqdm(todo_j, desc="Judge", unit="Bewertung", disable=not todo_j):
        key = (r["question_id"], r["final_answer"])
        if r["variante"] == "A" and key in cache:
            j, ms, tok = cache[key]
        else:
            with CallRecorder() as rec_:
                j = eu.judge_answer(args.judge_model, r["question"], r["final_answer"], r["reference_answer"], r["context"])
            ms, tok = rec_.ms, rec_.tokens
            if r["variante"] == "A":
                cache[key] = (j, ms, tok)
        r.update({k: j[k] for k in ("antwortqualitaet", "deutschqualitaet", "faithfulness", "korrekt")})
        r["judge_ms"], r["judge_tokens"] = ms, tok
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records), encoding="utf-8")

    # ---- Auswertung ----
    df = pd.DataFrame([r for r in records if r["repetition"] <= args.repetitions])
    if df.empty:
        print("Keine Daten - Abbruch.")
        return
    for c in ("antwortqualitaet", "deutschqualitaet", "faithfulness", "judge_ms", "judge_tokens", "latenz_ms", "tokens", "ram_peak_gb"):
        df[c] = pd.to_numeric(df.get(c, NAN), errors="coerce")
    df["korrekt_num"] = pd.to_numeric(df["korrekt"].map({True: 1.0, False: 0.0}), errors="coerce") if "korrekt" in df else NAN
    df["halluzination"] = (df["korrekt"] == False) | (df["faithfulness"] <= 2)
    df["latenz_inkl_judge_ms"] = np.where(df["variante"] == "B", df["latenz_ms"] + df["judge_ms"], np.nan)
    df["tokens_inkl_judge"] = np.where(df["variante"] == "B", df["tokens"] + df["judge_tokens"], np.nan)
    eu.write_csv(df.drop(columns=["korrekt_num", "prompt"], errors="ignore"), out / "results_details.csv")

    names = {"A": "A: Primary allein", "B": "B: Gesamtstack"}
    summ = []
    for v, g in df.groupby("variante"):
        q = g.dropna(subset=["antwortqualitaet"])
        row = {"variante": v, "bezeichnung": names[v], "n_laeufe": len(g), "fehlerrate_pct": 100 * g["fehler"].mean(),
               "latenz_ms_median": g["latenz_ms"].median(), "latenz_ms_mean": g["latenz_ms"].mean(),
               "latenz_ms_std": g["latenz_ms"].std(), "latenz_ms_p95": float(np.nanpercentile(g["latenz_ms"], 95)),
               "tokens_mean": g["tokens"].mean(), "ram_peak_gb_mean": g["ram_peak_gb"].mean(), "ram_peak_gb_max": g["ram_peak_gb"].max(),
               "antwortqualitaet_mean": q["antwortqualitaet"].mean(), "antwortqualitaet_std": q["antwortqualitaet"].std(),
               "deutschqualitaet_mean": q["deutschqualitaet"].mean(), "trefferquote_pct": 100 * q["korrekt_num"].mean() if q["korrekt_num"].notna().any() else NAN,                "halluzinationsrate_pct": 100 * g["halluzination"].mean()}
        if v == "B":
            row.update({"latenz_inkl_judge_ms_median": g["latenz_inkl_judge_ms"].median(),
                        "latenz_inkl_judge_ms_p95": float(np.nanpercentile(g["latenz_inkl_judge_ms"], 95)) if g["latenz_inkl_judge_ms"].notna().any() else NAN,
                        "tokens_inkl_judge_mean": g["tokens_inkl_judge"].mean(),
                        "routing_genauigkeit_pct": 100 * g["router_korrekt"].astype(float).mean(),
                        "critic_ablehnung_pct": 100 * g["critic_ablehnung"].astype(float).mean(),
                        "critic_ungueltig_pct": 100 * (1 - g["critic_gueltig"].astype(float).mean()),
                        "secondary_als_final_pct": 100 * (g["final_quelle"] == "secondary").mean()})
        summ.append(row)
    summary = pd.DataFrame(summ)
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # gepaarter Vergleich B gegen A (je Frage/Wiederholung)
    wide = df.pivot_table(index=["question_id", "repetition"], columns="variante", values=["latenz_ms", "tokens", "antwortqualitaet", "ram_peak_gb"])
    comp = {}
    if {"A", "B"} <= set(df["variante"]):
        def paired(metric, ratio=False):
            try:
                b, a = wide[(metric, "B")], wide[(metric, "A")]
            except KeyError:                       # z. B. --no-judge: keine Qualitätswerte vorhanden
                return pd.Series(dtype=float)
            return ((b / a).replace([np.inf, -np.inf], np.nan) if ratio else (b - a)).dropna()
        for nm, ser in (("qualitaet_delta_b_minus_a", paired("antwortqualitaet")),
                        ("latenz_faktor_b_durch_a", paired("latenz_ms", True)),
                        ("token_faktor_b_durch_a", paired("tokens", True))):
            m_, lo, hi = ex.bootstrap_ci(ser.values)
            comp.update({nm: m_, f"{nm}_ci95_unten": lo, f"{nm}_ci95_oben": hi})
        comp["ram_peak_delta_gb_b_minus_a"] = paired("ram_peak_gb").mean()
        eu.write_csv(pd.DataFrame([comp]).round(3), out / "results_vergleich.csv")

    # ---- Diagramme ----
    s = summary.set_index("variante")
    lat = pd.DataFrame([
        {"label": "A: Primary allein", "Median": s.loc["A", "latenz_ms_median"] / 1000, "P95": s.loc["A", "latenz_ms_p95"] / 1000},
        *([{"label": "B: Gesamtstack\n(ohne Judge)", "Median": s.loc["B", "latenz_ms_median"] / 1000, "P95": s.loc["B", "latenz_ms_p95"] / 1000},
           {"label": "B: Gesamtstack\ninkl. Judge", "Median": s.loc["B", "latenz_inkl_judge_ms_median"] / 1000,
            "P95": s.loc["B", "latenz_inkl_judge_ms_p95"] / 1000}] if "B" in s.index else [])]) if "A" in s.index else None
    if lat is not None:
        eu.plot_grouped_bars(lat, "label", {"Median": "Median", "95. Perzentil (P95)": "P95"},
                             "TS-14: Antwortzeit - Primary allein vs. Gesamtstack", "Latenz je Frage (s, niedriger = besser)",
                             out / "diagramm_latenz.png")
    if "B" in s.index:
        import matplotlib.pyplot as plt
        stages = [("router", "Router"), ("primary", "Primary"), ("secondary", "Secondary"), ("critic", "Critic"), ("judge", "Judge")]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars = {"A: Primary allein": {"primary": df[df["variante"] == "A"]["primary_ms"].median()},
                "B: Gesamtstack": {k: df[df["variante"] == "B"][f"{k}_ms" if k != "judge" else "judge_ms"].median() for k, _ in stages}}
        for xi, (lab, parts) in enumerate(bars.items()):
            bottom = 0.0
            for i, (k, nm) in enumerate(stages):
                v = (parts.get(k, 0) or 0) / 1000
                v = 0 if np.isnan(v) else v
                ax.bar(xi, v, bottom=bottom, color=eu.PALETTE[i], label=nm if xi == len(bars) - 1 else None, width=0.55)
                bottom += v
            ax.annotate(f"{bottom:.1f} s".replace(".", ","), (xi, bottom), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(list(bars))
        ax.set_ylabel("Median-Latenz je Stufe (s, gestapelt)")
        ax.set_title("TS-14: Wo geht die Zeit hin?")
        ax.legend(title="Stufe", frameon=False, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        eu._save(fig, out / "diagramm_latenz_stufen.png")
    eu.plot_grouped_bars(summary, "bezeichnung", {"Antwortqualität": "antwortqualitaet_mean", "Deutschqualität": "deutschqualitaet_mean"},
                         "TS-14: Bewertung der finalen Antworten (Judge, 1–5)", "Bewertung (1 = schlecht, 5 = sehr gut)",
                         out / "diagramm_qualitaet.png", err_cols={"Antwortqualität": "antwortqualitaet_std"}, ylim=(0, 5.8))
    eu.plot_single_bars(summary["bezeichnung"].tolist(), summary["ram_peak_gb_mean"], "TS-14: RAM-Spitzenwert der Ollama-Prozesse",
                        "RAM-Peak (GB, Mittel)", out / "diagramm_ram.png", fmt="{:.1f}")

    pd.set_option("display.width", 220)
    print("\n=== Ergebnis TS-14 ===")
    print(summary[["bezeichnung", "n_laeufe", "fehlerrate_pct", "latenz_ms_median", "latenz_ms_p95", "tokens_mean", "ram_peak_gb_mean",
                   "antwortqualitaet_mean", "halluzinationsrate_pct"]].round(2).to_string(index=False))
    if comp:
        print(f"\nB gegen A: Qualität {comp['qualitaet_delta_b_minus_a']:+.2f} Punkte "
              f"(95-%-KI [{comp['qualitaet_delta_b_minus_a_ci95_unten']:+.2f}; {comp['qualitaet_delta_b_minus_a_ci95_oben']:+.2f}]), "
              f"Latenz x{comp['latenz_faktor_b_durch_a']:.1f}, Token x{comp['token_faktor_b_durch_a']:.1f}")
    print(f"Dateien in: {out}")


if __name__ == "__main__":
    main()
