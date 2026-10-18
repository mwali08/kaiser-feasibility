#!/usr/bin/env python3
"""
TS-03: Secondary-Actor-Modelle (Gemma 3 vs. Llama 3)

Ablauf: gleiche Fragen wie TS-02 -> Judge bewertet Qualität/Deutsch -> auf den Faktenfragen wird gezählt,
wie oft Secondary und jeder Primary-Kandidat (aus TS-02) dieselbe FALSCHE Antwort geben ->
Familien-Check -> Nutzwertanalyse (hohe Qualität, geringe Fehler-Korrelation) -> CSV + Diagramme.

Voraussetzung: TS-02 wurde ausgeführt (../ts02/responses_ts02.jsonl).
Beispiel:  python ts03_secondary_actor.py --repetitions 3
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

import eval_utils as eu  # noqa: E402

# =========================== KONFIGURATION ================================== #
DEFAULT_MODELS = ["gemma3", "llama3"]
PRIMARY_CANDIDATES = ["qwen3", "mistral_small_4"]
TS02_DIR = HERE.parent / "ts02"

# Muss identisch zu TS-02 sein (gleicher System-Prompt!)
SYSTEM_PROMPT = ("Du bist ein hilfreicher, präziser Assistent. Antworte auf Deutsch, sachlich und knapp. "
                 "Wenn ein Kontext angegeben ist, stütze dich ausschließlich darauf und erfinde nichts dazu. "
                 "Wenn die Antwort nicht im Kontext steht, sage das ausdrücklich.")

# Angepasst an CPU-Inferenz (kein GPU).
# LF30 = TTFT, LF35 = TPS, LF60 = VRAM (entfaellt), LL70 = Antwortqualitaet
KO_KRITERIEN = {
    "ttft_ms_mean":           {"max": 12000.0, "label": "LF30 TTFT (ms)"},
    "tps_mean":               {"min": 5.0,     "label": "LF35 Token/s"},
    "antwortqualitaet_mean":  {"min": 3.5,     "label": "LL70 Mindest-Antwortqualität"},
    "ungueltig_quote_pct":    {"max": 5.0,     "label": "Max. Quote ungültiger Antworten (%)"},
}

# fehlerkorr_pct = Anteil der Faktenfragen, bei denen Secondary und Primary
# dieselbe falsche Antwort geben. Niedriger = besser (maximale Diversitaet).
NUTZWERT_KRITERIEN = {
    "antwortqualitaet_mean": {"weight": 0.25, "worst": 1.0,     "best": 5.0},
    "deutschqualitaet_mean": {"weight": 0.15, "worst": 1.0,     "best": 5.0},
    "fehlerkorr_pct":        {"weight": 0.30, "worst": 30.0,    "best": 0.0},
    "ttft_ms_mean":          {"weight": 0.15, "worst": 15000.0, "best": 500.0},
    "tps_mean":              {"weight": 0.15, "worst": 3.0,     "best": 20.0},
}
RADAR_LABELS = {
    "antwortqualitaet_mean": "Antwortqualität",
    "deutschqualitaet_mean": "Deutschqualität",
    "fehlerkorr_pct":        "Fehler-Unabhängigkeit",
    "ttft_ms_mean":          "Antwortstart (TTFT)",
    "tps_mean":              "Geschwindigkeit (TPS)",
}
# ============================================================================ #


def phi_coefficient(a, b, c, d):
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / den if den > 0 else float("nan")


def error_correlation(sec_df, prim_df, judge_model, use_judge):
    """Paarweise Fehleranalyse (gleiche Frage + gleiche Wiederholung) auf den Faktenfragen."""
    pairs, summary = [], []
    for sm in sec_df["model"].unique():
        for pm in prim_df["model"].unique():
            cols = ["question_id", "repetition", "question", "reference_answer", "response_text", "korrekt"]
            s = sec_df[(sec_df["model"] == sm) & sec_df["valid"] & sec_df["is_fact"] & sec_df["korrekt"].notna()][cols]
            p = prim_df[(prim_df["model"] == pm) & prim_df["valid"] & prim_df["is_fact"] & prim_df["korrekt"].notna()][cols]
            m = s.merge(p, on=["question_id", "repetition"], suffixes=("_s", "_p"))
            both_wrong = m[(m["korrekt_s"] == 0) & (m["korrekt_p"] == 0)]
            same = {}
            for _, r in tqdm(both_wrong.iterrows(), total=len(both_wrong), desc=f"Fehlervergleich {sm}/{pm}",
                             unit="Paar", leave=False):
                same[(r["question_id"], r["repetition"])] = (
                    eu.judge_same_error(judge_model, r["question_s"], r["reference_answer_s"],
                                        r["response_text_s"], r["response_text_p"]) if use_judge else None)
            for _, r in m.iterrows():
                pairs.append({"secondary": sm, "primary": pm, "question_id": r["question_id"],
                              "repetition": r["repetition"], "secondary_korrekt": int(r["korrekt_s"]),
                              "primary_korrekt": int(r["korrekt_p"]),
                              "beide_falsch": int(r["korrekt_s"] == 0 and r["korrekt_p"] == 0),
                              "gleicher_fehler": same.get((r["question_id"], r["repetition"]))})
            n = len(m)
            a = len(both_wrong)
            b = int(((m["korrekt_s"] == 0) & (m["korrekt_p"] == 1)).sum())
            c = int(((m["korrekt_s"] == 1) & (m["korrekt_p"] == 0)).sum())
            d = int(((m["korrekt_s"] == 1) & (m["korrekt_p"] == 1)).sum())
            n_same = sum(1 for v in same.values() if v)
            summary.append({"secondary": sm, "primary": pm, "n_fakten_paare": n, "n_beide_falsch": a,
                            "n_gleiche_falsche_antwort": n_same if use_judge else float("nan"),
                            "gleicher_fehler_pct": 100 * n_same / n if (n and use_judge) else float("nan"),
                            "p_sec_falsch_wenn_prim_falsch_pct": 100 * a / (a + c) if (a + c) else float("nan"),
                            "phi": phi_coefficient(a, b, c, d)})
    return pd.DataFrame(pairs), pd.DataFrame(summary)


def main():
    parser = eu.base_parser(__doc__, DEFAULT_MODELS)
    parser.add_argument("--ts02-dir", default=str(TS02_DIR), help="Ordner mit responses_ts02.jsonl")
    args = parser.parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    print(f"TS-03 | {len(questions)} Fragen | Modelle: {args.models} | Wiederholungen: {args.repetitions}")

    # --- Familien-Check (Rahmenbedingung: andere Familie als der Primary Actor) ---
    for sm in args.models:
        clash = [pm for pm in PRIMARY_CANDIDATES if eu.model_family(pm) == eu.model_family(sm)]
        if clash:
            eu.log.warning("Familien-Konflikt: %s gehört zur selben Familie wie %s", sm, clash)

    jsonl, vram_path = out / "responses_ts03.jsonl", out / "vram_messung.json"
    records = eu.evaluate_models(args.models, questions, args.repetitions, SYSTEM_PROMPT, args.judge_model,
                                 not args.no_judge, not args.no_stream, jsonl, vram_path, args.resume)
    df = eu.prepare_df([r for r in records if r["model"] in args.models and r["repetition"] <= args.repetitions])
    if df.empty:
        print("Keine Daten – Abbruch.")
        return
    eu.write_csv(df.drop(columns=["prompt", "context"]).rename(columns={"model": "modell_key", "model_name": "modell"}),
                 out / "results_details.csv")

    # --- Primary-Antworten aus TS-02 ---
    ts02 = eu.load_jsonl(Path(args.ts02_dir) / "responses_ts02.jsonl")
    prim_df = eu.prepare_df([r for r in ts02 if r["model"] in PRIMARY_CANDIDATES])
    ids_ts03 = {q["id"] for q in questions}
    if prim_df.empty:
        eu.log.warning("Keine TS-02-Antworten gefunden – Fehler-Korrelation entfällt. Zuerst TS-02 ausführen.")
        corr_pairs, corr_sum = pd.DataFrame(), pd.DataFrame()
    else:
        if not ids_ts03 <= set(prim_df["question_id"]):
            eu.log.warning("Testset von TS-03 und TS-02 weichen ab – nur gemeinsame Fragen werden verglichen.")
        corr_pairs, corr_sum = error_correlation(df, prim_df, args.judge_model, not args.no_judge)
        eu.write_csv(corr_pairs, out / "results_fehlerpaare.csv")
        eu.write_csv(corr_sum.round(3), out / "results_fehlerkorrelation.csv")

    # --- Zusammenfassung, KO, Nutzwert ---
    summary = eu.summarize_generation(df, eu.load_vram(vram_path))
    summary["fehlerkorr_pct"] = float("nan")
    if not corr_sum.empty:
        for sm, g in corr_sum.groupby("secondary"):
            summary.loc[summary["model"] == sm, "fehlerkorr_pct"] = g["gleicher_fehler_pct"].mean()
            for _, r in g.iterrows():
                summary.loc[summary["model"] == sm, f"gleicher_fehler_pct_vs_{r['primary']}"] = r["gleicher_fehler_pct"]
    summary["familie_ok"] = [all(eu.model_family(pm) != eu.model_family(m) for pm in PRIMARY_CANDIDATES)
                             for m in summary["model"]]
    ko = summary.apply(lambda r: eu.check_ko(r, KO_KRITERIEN), axis=1)
    summary["ko_bestanden"] = [k[0] and f for k, f in zip(ko, summary["familie_ok"])]
    summary["ko_hinweise"] = [k[1] + ("" if f else " | Familien-Konflikt mit Primary") for k, f in zip(ko, summary["familie_ok"])]
    nw = summary.apply(lambda r: eu.nutzwert(r, NUTZWERT_KRITERIEN), axis=1)
    summary["nutzwert"] = [n[0] for n in nw]
    summary["nutzwert_gewichtsabdeckung"] = [n[1] for n in nw]
    summary["rang_nutzwert"] = summary["nutzwert"].rank(ascending=False, method="min")
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # --- Diagramme ---
    eu.plot_grouped_bars(summary, "modell", {"Antwortqualität": "antwortqualitaet_mean",
                                             "Deutschqualität": "deutschqualitaet_mean"},
                         "TS-03: Bewertung der Antworten (Mittelwert, 1–5)", "Bewertung (1 = schlecht, 5 = sehr gut)",
                         out / "diagramm_qualitaet.png",
                         err_cols={"Antwortqualität": "antwortqualitaet_std", "Deutschqualität": "deutschqualitaet_std"},
                         ylim=(0, 5.6))
    if not corr_sum.empty and corr_sum["gleicher_fehler_pct"].notna().any():
        hm = corr_sum.pivot(index="secondary", columns="primary", values="gleicher_fehler_pct")
        hm.index = [eu.model_label(i) for i in hm.index]
        hm.columns = [eu.model_label(c) for c in hm.columns]
        eu.plot_heatmap(hm, "TS-03: Gleiche falsche Antworten auf Faktenfragen", out / "diagramm_heatmap_fehlerkorrelation.png",
                        "Anteil der Faktenfragen (%)", fmt="{:.1f} %", cmap="OrRd", vmin=0,
                        xlabel="Primary-Kandidat (TS-02)", ylabel="Secondary-Kandidat")
    valid = df[df["valid"]]
    eu.plot_box({m: g["ttft_ms"].tolist() for m, g in valid.groupby("model_name", sort=False)},
                "Zeit bis zum ersten Token (ms)", "TS-03: Antwortstart (TTFT) je Modell", out / "diagramm_ttft_boxplot.png")
    radar = {r["modell"]: [eu.nutzwert(r, NUTZWERT_KRITERIEN)[2][c] for c in RADAR_LABELS] for _, r in summary.iterrows()}
    eu.plot_radar(radar, list(RADAR_LABELS.values()), "TS-03: Stärken-Schwächen-Profil (1 = bestes Niveau)",
                  out / "diagramm_radar.png")
    eu.plot_single_bars(summary["modell"], summary["nutzwert"], "TS-03: Nutzwert (0–100 Punkte)", "Nutzwert",
                        out / "diagramm_nutzwert.png", ylim=(0, 100))

    pd.set_option("display.width", 200)
    print("\n=== Ergebnis TS-03 ===")
    print(summary[["modell", "n_gueltig", "antwortqualitaet_mean", "deutschqualitaet_mean", "fehlerkorr_pct",
                   "ttft_ms_mean", "tps_mean", "vram_gb", "ko_bestanden", "nutzwert"]].round(2).to_string(index=False))
    print(f"\nDateien in: {out}")


if __name__ == "__main__":
    main()
