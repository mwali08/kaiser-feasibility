#!/usr/bin/env python3
"""
TS-08: Routing-Verfahren - LLM-basiertes Routing (qwen3) vs. Embedding-basiertes Routing (nomic-embed-text)

Jede Frage wird von beiden Routern einer Kategorie zugeordnet. Soll-Kategorie = Feld "category" im Testset.
Der Router sieht NUR die Frage (kein Kontext, keine Referenzantwort).
  * LLM-Router:       Prompt mit Kategorienliste, Antwort = Kategoriename (Temperatur 0).
  * Embedding-Router: Cosine-Similarity zwischen Frage und Kategorie-Beschreibungen (Prototypen, einmalig berechnet).
Metriken: Genauigkeit (%), gültige Ausgaben (%), Latenz (Median, P95). Die einmalige Vorberechnung der
Kategorie-Embeddings wird separat als setup_ms ausgewiesen.

Beispiel:  python ts08_routing.py --repetitions 3
Trockenlauf: python ts08_routing.py --dry-run --out .\\dryrun
"""
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
# Beschreibungen der Routen (NICHT aus dem Testset abgeleitet, um Leckage zu vermeiden).
# Weitere Kategorien im Testset -> hier ergänzen, sonst dient der Kategoriename als Beschreibung.
CATEGORY_DESCRIPTIONS = {
    "vorschriften": "Fragen zu militärischen Dienstvorschriften, Aufbewahrungsfristen, Meldepflichten oder rechtlichen Regelungen.",
    "einsatzberichte": "Fragen zu den formalen Anforderungen, Inhalten oder Fristen von Einsatzberichten.",
    "logistik": "Fragen zur Nachschubkette, zu Transportkapazitäten, Paletten, Lieferzeiten oder Priorisierungen.",
    "strukturen": "Fragen zur Organisation des Bundesheeres, zu Militärkommanden oder Teilstreitkräften.",
    "meinung": "Fragen, die eine Erklärung, Definition oder Meinung zu einem militärischen Begriff verlangen (z.B. Befehl vs. Weisung).",
}
LLM_MAX_TOKENS = 16
# ============================================================================ #


def llm_route(key, question, routes):
    listing = "\n".join(f"- {r}: {CATEGORY_DESCRIPTIONS.get(r, r)}" for r in routes)
    prompt = ("Ordne die folgende Nutzerfrage genau einer Kategorie zu.\n"
              f"Kategorien:\n{listing}\n\nFrage: {question}\n\n"
              "Antworte NUR mit dem Kategorienamen, ohne weitere Wörter.")
    res = query_model(key, prompt, stream=False, temperature=0.0, max_tokens=LLM_MAX_TOKENS, retries=0)
    text = res["response_text"].lower()
    hits = [(text.find(r.lower()), r) for r in routes if r.lower() in text]
    ist = min(hits)[1] if hits else None            # frühester genannter Kategoriename
    return ist, res["total_latency_ms"], res["response_text"], res["error"]


def main():
    ex.fix_console()
    args = ex.common_parser(__doc__).parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    routes = list(CATEGORY_DESCRIPTIONS) + sorted({q["category"] for q in questions} - set(CATEGORY_DESCRIPTIONS))
    for r in routes:
        if r not in CATEGORY_DESCRIPTIONS:
            eu.log.warning("Keine Beschreibung für Kategorie '%s' – Kategoriename wird als Prototyp verwendet.", r)
    if args.dry_run:
        ex.setup_dry_run(args.judge_model)
        pd_.set_dry_run(True)
        ex.register_model("ts08_llm", {"provider": "mock", "mock_response": "fakten", "family": "qwen", "mock_delay_s": 0.03})
    else:
        pd_.register_variant("ts08_llm", args.model)
    print(f"TS-08 | {len(questions)} Fragen | Routen: {routes} | Wiederholungen: {args.repetitions}")

    # Einmaliger Setup: Embeddings der Kategorie-Beschreibungen (+ Warm-up des Embedding-Modells)
    pd_.embed_texts(["search_query: Aufwärmen"])
    proto, setup_ms = pd_.embed_texts([f"search_document: {r}: {CATEGORY_DESCRIPTIONS.get(r, r)}" for r in routes])
    query_model("ts08_llm", "Antworte mit einem Wort: Hallo.", stream=False, max_tokens=8, retries=0)   # Warm-up

    rows = []
    for rep in range(1, args.repetitions + 1):
        for q in tqdm(questions, desc=f"Routing Lauf {rep}", unit="Frage"):
            for router in ("LLM-Router (qwen3)", "Embedding-Router (nomic)"):
                rec = {"router": router, "question_id": q["id"], "repetition": rep, "frage": q["question"],
                       "soll": q["category"], "ist": None, "latenz_ms": np.nan, "rohausgabe": "", "error": None}
                try:
                    if router.startswith("LLM"):
                        ist, lat, raw, err = llm_route("ts08_llm", q["question"], routes)
                    else:
                        vec, lat = pd_.embed_texts([f"search_query: {q['question']}"])
                        sims = pd_.cosine_matrix(vec, proto)[0]
                        ist, raw, err = routes[int(np.argmax(sims))], f"max cos={sims.max():.3f}", None
                    rec.update({"ist": ist, "latenz_ms": lat, "rohausgabe": raw, "error": err})
                except Exception as exc:  # noqa: BLE001 - Lauf darf nie abbrechen
                    eu.log.exception("Router-Fehler | %s | %s", router, q["id"])
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["gueltig"] = rec["ist"] in routes
                rec["korrekt"] = rec["ist"] == rec["soll"]
                if not rec["gueltig"]:
                    eu.log.warning("Ungültige Router-Ausgabe | %s | %s | %r | %s", router, q["id"], rec["rohausgabe"], rec["error"])
                rows.append(rec)
    df = pd.DataFrame(rows)
    eu.write_csv(df, out / "results_details.csv")

    # ---- Zusammenfassung ----
    summ = []
    for router, g in df.groupby("router", sort=False):
        lo, hi = pd_.wilson_ci(int(g["korrekt"].sum()), len(g))
        lat = g[g["gueltig"]]["latenz_ms"]
        row = {"router": router, "n_anfragen": len(g), "gueltig_pct": 100 * g["gueltig"].mean(),
               "genauigkeit_pct": 100 * g["korrekt"].mean(), "genauigkeit_ci95_unten": lo, "genauigkeit_ci95_oben": hi,
               "latenz_median_ms": lat.median(), "latenz_mean_ms": lat.mean(), "latenz_p95_ms": pd_.pct(lat, 95),
               "setup_ms": setup_ms if router.startswith("Emb") else 0.0}
        for cat in routes:
            s = g[g["soll"] == cat]
            row[f"genauigkeit_{cat}_pct"] = 100 * s["korrekt"].mean() if len(s) else np.nan
        summ.append(row)
    summary = pd.DataFrame(summ)
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # ---- Diagramme ----
    eu.plot_grouped_bars(summary, "router", {"Genauigkeit (richtige Kategorie)": "genauigkeit_pct",
                                             "Gültige Ausgaben": "gueltig_pct"},
                         "TS-08: Wie oft wählt der Router die richtige Kategorie?", "Anteil der Anfragen (%)",
                         out / "diagramm_genauigkeit.png", ylim=(0, 118))
    eu.plot_box({r: g["latenz_ms"].tolist() for r, g in df.groupby("router", sort=False)},
                "Latenz je Routing-Entscheidung (ms)", "TS-08: Geschwindigkeit der Router", out / "diagramm_latenz_boxplot.png")
    hm = summary.set_index("router")[[f"genauigkeit_{c}_pct" for c in routes]]
    hm.columns = routes
    eu.plot_heatmap(hm, "TS-08: Genauigkeit nach Fragekategorie", out / "diagramm_heatmap_kategorien.png",
                    "Genauigkeit (%)", fmt="{:.0f} %", vmin=0, vmax=100, xlabel="Soll-Kategorie", ylabel="Router")
    for i, (router, g) in enumerate(df.groupby("router", sort=False)):
        cm = pd.crosstab(g["soll"], g["ist"].fillna("ungültig")).reindex(index=routes, fill_value=0)
        cm = cm.reindex(columns=routes + (["ungültig"] if "ungültig" in cm.columns else []), fill_value=0)
        eu.plot_heatmap(cm, f"TS-08: Verwechslungen - {router}", out / f"diagramm_konfusion_{'llm' if i == 0 else 'embedding'}.png",
                        "Anzahl", fmt="{:.0f}", cmap="Blues", xlabel="Vom Router gewählt", ylabel="Soll-Kategorie")

    pd.set_option("display.width", 200)
    print("\n=== Ergebnis TS-08 ===")
    print(summary[["router", "genauigkeit_pct", "gueltig_pct", "latenz_median_ms", "latenz_p95_ms", "setup_ms"]].round(1).to_string(index=False))
    print(f"\nDateien in: {out}")


if __name__ == "__main__":
    main()