#!/usr/bin/env python3
"""
TS-04: Critic-Modelle (Prometheus 2 vs. M-Prometheus) und Paarentscheidung

Ablauf: Beide Critics bewerten dieselben ~50 Antworten (Primary-Antworten aus TS-02 + dokumentierte, bewusst
fehlerhafte Antworten) mit identischem Bewertungs-Prompt (Prometheus-"Absolute Grading"-Format, Skala 1–5).
Score >= SCHWELLE gilt als "akzeptiert". Vergleich gegen die manuelle Referenzbewertung.
Danach: Nutzwert je Critic und Paarentscheidung aus den Nutzwerten von TS-02 und TS-04.

Eingaben (im Skriptordner):
  flawed_answers.json     bewusst fehlerhafte Antworten (manuell dokumentiert)
  reference_labels.json   manuelle Referenzbewertung der Modellantworten, Schlüssel = answer_id
                          {"qwen3|F01|1": {"label": "korrekt|fehlerhaft|halluzination", "score": 1-5}}
  ../ts02/responses_ts02.jsonl, ../ts02/results_summary.csv
Fehlt ein Label, wird ersatzweise das Judge-Urteil aus TS-02 genutzt (WARNUNG, bitte manuell prüfen);
reference_labels_template.json wird zum Ausfüllen erzeugt.

Beispiel:  python ts04_critic.py --repetitions 1
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
from llm_client import query_model  # noqa: E402

# =========================== KONFIGURATION ================================== #
DEFAULT_CRITICS = ["prometheus2", "m_prometheus"]
PRIMARY_CANDIDATES = ["qwen3", "mistral_small_4"]
N_PER_PRIMARY = 25                 # ca. 50 Modellantworten insgesamt (+ fehlerhafte Antworten)
ACCEPT_THRESHOLD = 4               # Score >= 4 = Antwort akzeptiert

# Paare laut Rahmenbedingungen (Primary, Critic)
PAIRS = {"Qwen3 + Prometheus 2": ("qwen3", "prometheus2"),
         "Mistral Small 4 + M-Prometheus": ("mistral_small_4", "m_prometheus")}
PAIR_WEIGHT_PRIMARY = 0.5          # Gewicht des Primary-Nutzwerts (Rest = Critic)

# TODO: Schwellen an LF40/LF60/LF65/LL70 anpassen (PLATZHALTER)
KO_KRITERIEN = {
    "bewertungsgenauigkeit_pct": {"min": 70.0,   "label": "LF40 Bewertungsgenauigkeit (%)"},
    "total_latency_ms_mean":     {"max": 60000.0, "label": "LF60 Latenz (ms)"},
    "ungueltig_quote_pct":       {"max": 10.0,   "label": "LF65 Max. ungültige Bewertungen (%)"},
}
NUTZWERT_KRITERIEN = {
    "bewertungsgenauigkeit_pct":  {"weight": 0.30, "worst": 50.0,    "best": 100.0},
    "halluzination_erkennung_pct": {"weight": 0.30, "worst": 0.0,     "best": 100.0},
    "falsch_positiv_rate_pct":    {"weight": 0.20, "worst": 50.0,    "best": 0.0},
    "total_latency_ms_mean":      {"weight": 0.10, "worst": 15000.0, "best": 500.0},
    "token_count_mean":           {"weight": 0.10, "worst": 1000.0,  "best": 50.0},
}
RADAR_LABELS = {"bewertungsgenauigkeit_pct": "Genauigkeit", "halluzination_erkennung_pct": "Halluzinations-\nerkennung",
                "falsch_positiv_rate_pct": "Wenig Falsch-Positive", "total_latency_ms_mean": "Geschwindigkeit",
                "token_count_mean": "Token-Sparsamkeit"}

# Identischer Bewertungs-Prompt für alle Critics (Prometheus-2-Format; englische Vorlage wie im Training,
# deutsche Inhalte). Rubrik misst Korrektheit, Halluzinationsfreiheit und Kontexttreue.
PROMPT_TEMPLATE = """###Task Description:
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


def parse_score(text: str):
    m = RESULT_RE.search(text or "")
    return int(m.group(1)) if m else None


def build_critic_prompt(item: dict) -> str:
    instruction = item["question"] if not item.get("context") else f"Context:\n{item['context']}\n\nQuestion: {item['question']}"
    return PROMPT_TEMPLATE.format(instruction=instruction, response=item["answer"],
                                  reference=item.get("reference_answer") or "(no reference answer available)")


def build_eval_set(ts02_dir: Path, flawed_path: Path, labels_path: Path, questions_by_id: dict) -> list[dict]:
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    labels = {k: v for k, v in labels.items() if not k.startswith("_")}
    recs = eu.prepare_df([r for r in eu.load_jsonl(ts02_dir / "responses_ts02.jsonl") if r["model"] in PRIMARY_CANDIDATES])
    items, auto = [], 0
    for pm in PRIMARY_CANDIDATES:
        g = recs[(recs["model"] == pm) & recs["valid"] & (recs["repetition"] == 1)].sort_values("question_id") \
            if not recs.empty else pd.DataFrame()
        for _, r in g.head(N_PER_PRIMARY).iterrows():
            aid = f"{pm}|{r['question_id']}|{r['repetition']}"
            lab = labels.get(aid)
            if lab:
                label, score, src = lab["label"], lab.get("score"), "manuell"
            else:
                ok = r["korrekt"]
                label = "korrekt" if ok == 1.0 else ("fehlerhaft" if ok == 0.0 else None)
                score, src = None, "auto_ts02_judge"
                auto += 1
            if label is None:
                eu.log.warning("Keine Referenzbewertung für %s – übersprungen.", aid)
                continue
            items.append({"answer_id": aid, "question_id": r["question_id"], "question": r["question"],
                          "context": r["context"], "reference_answer": r["reference_answer"], "answer": r["response_text"],
                          "source_model": pm, "source_family": eu.model_family(pm), "ref_label": label,
                          "ref_score": score, "ref_source": src, "error_type": None})
    if auto:
        eu.log.warning("%d Referenzlabels stammen vom TS-02-Judge, NICHT aus manueller Prüfung. "
                       "reference_labels_template.json ausfüllen und erneut starten.", auto)
        print(f"WARNUNG: {auto} Referenzlabels automatisch (Judge) – für die Arbeit manuell prüfen!")
    if flawed_path.exists():
        data = json.loads(flawed_path.read_text(encoding="utf-8"))
        for f in data["answers"]:
            q = questions_by_id.get(str(f["question_id"]))
            if not q:
                eu.log.warning("Fehlerhafte Antwort %s: question_id unbekannt.", f["id"])
                continue
            items.append({"answer_id": f"flawed|{f['id']}", "question_id": q["id"], "question": q["question"],
                          "context": q["context"], "reference_answer": q["reference_answer"],
                          "answer": f["answer_text"], "source_model": "manuell_fehlerhaft", "source_family": None,
                          "ref_label": f["expected_label"], "ref_score": f.get("ref_score"),
                          "ref_source": "dokumentiert", "error_type": f.get("error_type")})
    return items


def main():
    parser = eu.base_parser(__doc__, DEFAULT_CRITICS)
    parser.add_argument("--ts02-dir", default=str(HERE.parent / "ts02"), help="Ordner mit TS-02-Ergebnissen")
    args = parser.parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    ts02_dir = Path(args.ts02_dir)
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE.parent / "ts02" / "testset.json")
    items = build_eval_set(ts02_dir, HERE / "flawed_answers.json", HERE / "reference_labels.json",
                           {q["id"]: q for q in questions})
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("Keine Bewertungsobjekte – TS-02 ausführen bzw. Eingabedateien prüfen.")
        return
    (out / "reference_labels_template.json").write_text(json.dumps(
        {"_hinweis": "label: korrekt | fehlerhaft | halluzination; score: 1-5 (optional). Datei zu reference_labels.json umbenennen.",
         **{i["answer_id"]: {"label": i["ref_label"], "score": i["ref_score"], "frage": i["question"],
                             "antwort": i["answer"][:300]} for i in items if i["answer_id"][:6] != "flawed"}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TS-04 | {len(items)} Bewertungsobjekte | Critics: {args.models} | Wiederholungen: {args.repetitions}")

    # ---- Bewertungsläufe ----
    rows = []
    for critic in args.models:
        for rep in range(1, args.repetitions + 1):
            for it in tqdm(items, desc=f"{eu.model_label(critic):<14} Rep {rep}", unit="Bewertung"):
                rec = {"critic": critic, "critic_name": eu.model_label(critic), "repetition": rep,
                       **{k: it[k] for k in ("answer_id", "question_id", "source_model", "source_family", "ref_label",
                                             "ref_score", "ref_source", "error_type")}}
                rec["familienbezug"] = ("manuell fehlerhaft" if it["source_family"] is None else
                                        "gleiche Familie" if it["source_family"] == eu.model_family(critic) else "andere Familie")
                rec["ref_akzeptiert"] = it["ref_label"] == "korrekt"
                try:
                    res = query_model(critic, build_critic_prompt(it), stream=not args.no_stream, temperature=0.0,
                                      max_tokens=1024)
                    score = parse_score(res["response_text"]) if not res["error"] else None
                    rec.update({"critic_output": res["response_text"], "critic_score": score,
                                "ttft_ms": res["ttft_ms"], "total_latency_ms": res["total_latency_ms"],
                                "token_count": res["token_count"], "error": res["error"]})
                    rec["valid"] = score is not None
                    if not rec["valid"]:
                        eu.log.warning("Ungültige Critic-Antwort | %s | %s | %s", critic, it["answer_id"],
                                       res["error"] or "kein [RESULT]-Score gefunden")
                        rec["error"] = res["error"] or "kein [RESULT]-Score gefunden"
                except Exception as exc:  # noqa: BLE001
                    eu.log.exception("Fehler | %s | %s", critic, it["answer_id"])
                    rec.update({"valid": False, "critic_score": None, "error": f"{type(exc).__name__}: {exc}"})
                rec["verdict_akzeptiert"] = (rec["critic_score"] is not None and rec["critic_score"] >= ACCEPT_THRESHOLD) \
                    if rec["valid"] else None
                rows.append(rec)
    df = pd.DataFrame(rows)
    df["valid"] = df["valid"].astype(bool)
    eu.write_csv(df, out / "results_details.csv")

    # ---- Kennzahlen je Critic ----
    summ = []
    for critic, g in df.groupby("critic", sort=False):
        v = g[g["valid"]].copy()
        v["richtig"] = v["verdict_akzeptiert"].astype(bool) == v["ref_akzeptiert"]
        hall, korr = v[v["ref_label"] == "halluzination"], v[v["ref_label"] == "korrekt"]
        row = {"critic": critic, "modell": eu.model_label(critic), "familie": eu.model_family(critic),
               "n_bewertungen": len(g), "n_gueltig": len(v),
               "ungueltig_quote_pct": 100 * (1 - len(v) / len(g)),
               "bewertungsgenauigkeit_pct": 100 * v["richtig"].mean() if len(v) else np.nan,
               "halluzination_erkennung_pct": 100 * (~hall["verdict_akzeptiert"].astype(bool)).mean() if len(hall) else np.nan,
               "falsch_positiv_rate_pct": 100 * (~korr["verdict_akzeptiert"].astype(bool)).mean() if len(korr) else np.nan,
               "token_count_mean": v["token_count"].mean(), "total_latency_ms_mean": v["total_latency_ms"].mean(),
               "total_latency_ms_std": v["total_latency_ms"].std(), "ttft_ms_mean": v["ttft_ms"].mean()}
        for fb in ("gleiche Familie", "andere Familie", "manuell fehlerhaft"):
            s = v[v["familienbezug"] == fb]
            row[f"genauigkeit_{fb.replace(' ', '_')}_pct"] = 100 * s["richtig"].mean() if len(s) else np.nan
        sc = v.dropna(subset=["ref_score"])
        if len(sc) > 2 and sc["critic_score"].nunique() > 1 and sc["ref_score"].astype(float).nunique() > 1:
            row["score_mae_zur_referenz"] = (sc["critic_score"] - sc["ref_score"].astype(float)).abs().mean()
            row["score_rangkorrelation"] = sc["critic_score"].rank().corr(sc["ref_score"].astype(float).rank())
        summ.append(row)
    summary = pd.DataFrame(summ)
    ko = summary.apply(lambda r: eu.check_ko(r, KO_KRITERIEN), axis=1)
    summary["ko_bestanden"], summary["ko_hinweise"] = [k[0] for k in ko], [k[1] for k in ko]
    nw = summary.apply(lambda r: eu.nutzwert(r, NUTZWERT_KRITERIEN), axis=1)
    summary["nutzwert"], summary["nutzwert_gewichtsabdeckung"] = [n[0] for n in nw], [n[1] for n in nw]
    summary["rang_nutzwert"] = summary["nutzwert"].rank(ascending=False, method="min")
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # ---- Paarentscheidung (Nutzwert TS-02 + Nutzwert TS-04) ----
    pair_rows = []
    ts02_sum_path = ts02_dir / "results_summary.csv"
    if ts02_sum_path.exists():
        ts02_sum = eu.read_csv(ts02_sum_path).set_index("model")
        for name, (pm, cm) in PAIRS.items():
            if pm not in ts02_sum.index or cm not in set(summary["critic"]):
                continue
            nw_p, nw_c = float(ts02_sum.loc[pm, "nutzwert"]), float(summary.loc[summary["critic"] == cm, "nutzwert"].iloc[0])
            ko_ok = (str(ts02_sum.loc[pm, "ko_bestanden"]).lower() == "true") and bool(summary.loc[summary["critic"] == cm, "ko_bestanden"].iloc[0])
            pair_rows.append({"paar": name, "primary": pm, "critic": cm, "nutzwert_primary_ts02": nw_p,
                              "nutzwert_critic_ts04": nw_c,
                              "paar_nutzwert": PAIR_WEIGHT_PRIMARY * nw_p + (1 - PAIR_WEIGHT_PRIMARY) * nw_c,
                              "ko_beider_bestanden": ko_ok})
        pairs = pd.DataFrame(pair_rows)
        if not pairs.empty:
            pairs["rang"] = pairs["paar_nutzwert"].rank(ascending=False, method="min")
            eu.write_csv(pairs.round(3), out / "results_paarentscheidung.csv")
    else:
        eu.log.warning("TS-02-Zusammenfassung nicht gefunden (%s) – Paarentscheidung entfällt.", ts02_sum_path)
        pairs = pd.DataFrame()

    # ---- Diagramme ----
    eu.plot_grouped_bars(summary, "modell", {"Bewertungsgenauigkeit": "bewertungsgenauigkeit_pct",
                                             "Halluzinationen erkannt": "halluzination_erkennung_pct",
                                             "Falsch-Positiv-Rate (niedriger = besser)": "falsch_positiv_rate_pct"},
                         "TS-04: Qualität der Critics", "Anteil (%)", out / "diagramm_critic_metriken.png", ylim=(0, 115))
    eu.plot_box({m: g.loc[g["valid"], "total_latency_ms"].tolist() for m, g in df.groupby("critic_name", sort=False)},
                "Latenz je Bewertung (ms)", "TS-04: Latenz der Critics", out / "diagramm_critic_latenz_boxplot.png")
    cols = {"gleiche Familie": "genauigkeit_gleiche_Familie_pct", "andere Familie": "genauigkeit_andere_Familie_pct",
            "manuell fehlerhaft": "genauigkeit_manuell_fehlerhaft_pct"}
    hm = summary.set_index("modell")[list(cols.values())]
    hm.columns = [f"Antworten:\n{k}" for k in cols]
    if hm.notna().any().any():
        eu.plot_heatmap(hm, "TS-04: Bewertungsgenauigkeit nach Herkunft der Antwort", out / "diagramm_critic_heatmap_familie.png",
                        "Genauigkeit (%)", fmt="{:.0f} %", vmin=0, vmax=100, ylabel="Critic")
    radar = {r["modell"]: [eu.nutzwert(r, NUTZWERT_KRITERIEN)[2][c] for c in RADAR_LABELS] for _, r in summary.iterrows()}
    eu.plot_radar(radar, list(RADAR_LABELS.values()), "TS-04: Stärken-Schwächen-Profil der Critics (1 = bestes Niveau)",
                  out / "diagramm_critic_radar.png")
    if not pairs.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.5))
        wp = PAIR_WEIGHT_PRIMARY
        b1 = ax.bar(pairs["paar"], wp * pairs["nutzwert_primary_ts02"], color=eu.PALETTE[0], label="Anteil Primary (TS-02)")
        ax.bar(pairs["paar"], (1 - wp) * pairs["nutzwert_critic_ts04"], bottom=wp * pairs["nutzwert_primary_ts02"],
               color=eu.PALETTE[1], label="Anteil Critic (TS-04)")
        for x, v, ok in zip(range(len(pairs)), pairs["paar_nutzwert"], pairs["ko_beider_bestanden"]):
            ax.annotate(f"{v:.1f}" + ("" if ok else "\n(KO verletzt)"), (x, v), ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Paar-Nutzwert (0–100 Punkte)")
        ax.set_title("TS-04: Paarentscheidung Primary Actor + Critic")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.grid(axis="y", alpha=0.3)
        eu._save(fig, out / "diagramm_paarentscheidung.png")

    pd.set_option("display.width", 220)
    print("\n=== Ergebnis TS-04 (Critics) ===")
    print(summary[["modell", "n_gueltig", "bewertungsgenauigkeit_pct", "halluzination_erkennung_pct", "falsch_positiv_rate_pct",
                   "token_count_mean", "total_latency_ms_mean", "ko_bestanden", "nutzwert"]].round(2).to_string(index=False))
    if not pairs.empty:
        print("\n=== Paarentscheidung ===")
        print(pairs.round(2).to_string(index=False))
    print(f"\nDateien in: {out}")


if __name__ == "__main__":
    main()
