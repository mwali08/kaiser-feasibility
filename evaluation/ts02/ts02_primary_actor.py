#!/usr/bin/env python3
"""
TS-02: Primary-Actor-Modelle (Qwen3 vs. Mistral Small 4)

Ablauf: gleiche Fragen, gleicher System-Prompt, gleiche Parameter (Temperatur 0.0) ->
Judge bewertet Antwortqualität, Deutschqualität, Faithfulness (bei Kontext) ->
KO-Kriterien -> Nutzwertanalyse -> CSV + Diagramme.
Alle Antworten liegen in responses_ts02.jsonl und werden in TS-03/TS-04 weiterverwendet.

Beispiel:  python ts02_primary_actor.py --repetitions 3
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import pandas as pd  # noqa: E402

import eval_utils as eu  # noqa: E402

# =========================== KONFIGURATION ================================== #
DEFAULT_MODELS = ["qwen3", "mistral_small_4"]

SYSTEM_PROMPT = ("Du bist ein hilfreicher, präziser Assistent. Antworte auf Deutsch, sachlich und knapp. "
                 "Wenn ein Kontext angegeben ist, stütze dich ausschließlich darauf und erfinde nichts dazu. "
                 "Wenn die Antwort nicht im Kontext steht, sage das ausdrücklich.")

# KO-Kriterien. TODO: Schwellen an die Anforderungen LF25/LF35/LF45/LL70 der Arbeit anpassen
# (die Werte hier sind PLATZHALTER).
KO_KRITERIEN = {
    "ttft_ms_mean":          {"max": 12000.0, "label": "LF25 TTFT (ms)"},  # 8s real + Puffer
    "tps_mean":              {"min": 5.0,    "label": "LF35 Token/s"},
    "vram_gb":               {"max": 48.0,   "label": "LF45 VRAM (GB)"},
    "antwortqualitaet_mean": {"min": 3.5,    "label": "LL70 Mindest-Antwortqualität"},
    "ungueltig_quote_pct":   {"max": 5.0,    "label": "Max. Quote ungültiger Antworten (%)"},
}

# Nutzwertanalyse: Gewichte (Summe 1.0) und Ankerwerte (worst -> 0 Punkte, best -> 100 %)
NUTZWERT_KRITERIEN = {
    "antwortqualitaet_mean": {"weight": 0.35, "worst": 1.0,    "best": 5.0},
    "deutschqualitaet_mean": {"weight": 0.25, "worst": 1.0,    "best": 5.0},
    "ttft_ms_mean":          {"weight": 0.20, "worst": 6000.0, "best": 500.0},
    "tps_mean":              {"weight": 0.20, "worst": 5.0,    "best": 30.0},
}
RADAR_LABELS = {"antwortqualitaet_mean": "Antwortqualität", "deutschqualitaet_mean": "Deutschqualität",
                "ttft_ms_mean": "Antwortstart (TTFT)", "tps_mean": "Geschwindigkeit (TPS)"}
# ============================================================================ #


def main():
    args = eu.base_parser(__doc__, DEFAULT_MODELS).parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    print(f"TS-02 | {len(questions)} Fragen | Modelle: {args.models} | Wiederholungen: {args.repetitions}")

    jsonl, vram_path = out / "responses_ts02.jsonl", out / "vram_messung.json"
    records = eu.evaluate_models(args.models, questions, args.repetitions, SYSTEM_PROMPT, args.judge_model,
                                 not args.no_judge, not args.no_stream, jsonl, vram_path, args.resume)
    df = eu.prepare_df([r for r in records if r["model"] in args.models and r["repetition"] <= args.repetitions])
    if df.empty:
        print("Keine Daten – Abbruch.")
        return

    # ---- Details ----
    details = df.drop(columns=["prompt", "context"]).rename(columns={"model": "modell_key", "model_name": "modell"})
    eu.write_csv(details, out / "results_details.csv")

    # ---- Zusammenfassung, KO, Nutzwert ----
    summary = eu.summarize_generation(df, eu.load_vram(vram_path))
    ko = summary.apply(lambda r: eu.check_ko(r, KO_KRITERIEN), axis=1)
    summary["ko_bestanden"] = [k[0] for k in ko]
    summary["ko_hinweise"] = [k[1] for k in ko]
    nw = summary.apply(lambda r: eu.nutzwert(r, NUTZWERT_KRITERIEN), axis=1)
    summary["nutzwert"] = [n[0] for n in nw]
    summary["nutzwert_gewichtsabdeckung"] = [n[1] for n in nw]
    summary["rang_nutzwert"] = summary["nutzwert"].rank(ascending=False, method="min")
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # ---- Diagramme ----
    eu.plot_grouped_bars(summary, "modell",
                         {"Antwortqualität": "antwortqualitaet_mean", "Deutschqualität": "deutschqualitaet_mean",
                          "Faithfulness": "faithfulness_mean"},
                         "TS-02: Bewertung der Antworten (Mittelwert, 1–5)", "Bewertung (1 = schlecht, 5 = sehr gut)",
                         out / "diagramm_qualitaet.png",
                         err_cols={"Antwortqualität": "antwortqualitaet_std", "Deutschqualität": "deutschqualitaet_std",
                                   "Faithfulness": "faithfulness_std"}, ylim=(0, 5.6))
    valid = df[df["valid"]]
    eu.plot_box({m: g["ttft_ms"].tolist() for m, g in valid.groupby("model_name", sort=False)},
                "Zeit bis zum ersten Token (ms)", "TS-02: Antwortstart (TTFT) je Modell", out / "diagramm_ttft_boxplot.png")
    eu.plot_box({m: g["tps"].tolist() for m, g in valid.groupby("model_name", sort=False)},
                "Token pro Sekunde", "TS-02: Generierungsgeschwindigkeit je Modell", out / "diagramm_tps_boxplot.png")
    pivot = valid.assign(q=pd.to_numeric(valid["antwortqualitaet"], errors="coerce")) \
        .pivot_table(index="category", columns="model_name", values="q", aggfunc="mean")
    if not pivot.empty and pivot.notna().any().any():
        eu.plot_heatmap(pivot, "TS-02: Antwortqualität nach Fragekategorie", out / "diagramm_heatmap_kategorien.png",
                        "Ø Antwortqualität (1–5)", vmin=1, vmax=5, xlabel="Modell", ylabel="Fragekategorie")
    radar = {r["modell"]: [eu.nutzwert(r, NUTZWERT_KRITERIEN)[2][c] for c in RADAR_LABELS] for _, r in summary.iterrows()}
    eu.plot_radar(radar, list(RADAR_LABELS.values()), "TS-02: Stärken-Schwächen-Profil (1 = bestes Niveau)",
                  out / "diagramm_radar.png")
    eu.plot_single_bars(summary["modell"], summary["nutzwert"], "TS-02: Nutzwert (0–100 Punkte)", "Nutzwert",
                        out / "diagramm_nutzwert.png", ylim=(0, 100))

    pd.set_option("display.width", 200)
    print("\n=== Ergebnis TS-02 ===")
    print(summary[["modell", "n_gueltig", "antwortqualitaet_mean", "deutschqualitaet_mean", "faithfulness_mean",
                   "ttft_ms_mean", "tps_mean", "vram_gb", "ko_bestanden", "nutzwert"]].round(2).to_string(index=False))
    print(f"\nDateien in: {out}")


if __name__ == "__main__":
    main()
