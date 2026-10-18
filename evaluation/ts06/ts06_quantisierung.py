#!/usr/bin/env python3
"""
TS-06: Quantisierungsstufen (Engine aus TS-05: Ollama)
Stufen: q4_K_M, q5_K_M, q8_0 (soweit als Ollama-Tag für qwen3:4b-instruct-2507 verfügbar) + fp16 als
unquantisierte Referenz (falls sie auf dem Laptop läuft).

ABWEICHUNGEN VOM URSPRÜNGLICHEN PLAN (siehe auch shared/eval_utils_extension.py):
  2. Qwen3 standard -> qwen3:4b-instruct-2507 (Thinking-Mode ungeeignet für den Router).
  3. VRAM nicht messbar -> RAM-Peak der Ollama-Prozesse via psutil.
  4. Perplexity über Ollama nicht verfügbar -> Ersatzmetriken: Judge-Qualität, Antwortlänge, Textähnlichkeit zur
     Referenzstufe. (Echte Perplexity wäre mit dem Tool 'llama-perplexity' aus llama.cpp auf den GGUF-Dateien
     möglich – außerhalb dieses Skripts.)

Nicht vorhandene Tags werden übersprungen (Hinweis mit 'ollama pull <tag>' bzw. Import der GGUF aus einer
Offline-Quelle). Für q5_K_M ggf. GGUF importieren (Modelfile) und den Tag unter TAG_OVERRIDES eintragen.

ENTSCHEIDUNG: Kleinste Stufe, deren Qualitätsverlust gegenüber der Referenz unter der Schwelle bleibt
(Qualitätsverlust = Referenz-Qualität minus Stufen-Qualität, gepaart je Frage; Bootstrap-95-%-KI wird ausgewiesen).
HINWEIS: Mit nur wenigen Fragen ist das Judge-Rauschen größer als der echte Unterschied zwischen q5 und q8 –
für belastbare Aussagen ein Testset mit >= 30 Fragen und >= 3 Wiederholungen verwenden.

Beispiel:  python ts06_quantisierung.py --repetitions 3
Trockenlauf: python ts06_quantisierung.py --dry-run --out .\\dryrun
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eval_utils as eu  # noqa: E402
import eval_utils_extension as ex  # noqa: E402
import llm_client  # noqa: E402

# =========================== KONFIGURATION ================================== #
SYSTEM_PROMPT = ("Du bist ein hilfreicher, präziser Assistent. Antworte auf Deutsch, sachlich und knapp. "
                 "Wenn ein Kontext angegeben ist, stütze dich ausschließlich darauf und erfinde nichts dazu. "
                 "Wenn die Antwort nicht im Kontext steht, sage das ausdrücklich.")
MAX_TOKENS = 300
# Stufen aufsteigend nach Größe; bpw = ungefähre Bits pro Gewicht (für Diagramme/Reihenfolge)
STUFEN = [{"stufe": "q4_K_M", "bpw": 4.8}, {"stufe": "q6_K", "bpw": 6.5},
          {"stufe": "q8_0", "bpw": 8.5}, {"stufe": "fp16", "bpw": 16.0}]
TAG_OVERRIDES = {"q6_K": "qwen3-q6k"}             # z. B. {"q5_K_M": "qwen3-4b-2507-q5km-lokal:latest"} bei Offline-Import
QUALITAETSVERLUST_MAX = 0.3      # Skalenpunkte (Judge 1-5) – ANNAHME, an die Arbeit anpassen
FAKTENVERLUST_MAX_PP = 10.0      # max. Rückgang der Faktentreffer in Prozentpunkten – ANNAHME
ENTSCHEIDUNG_NACH = "mittelwert"  # oder "ci_obergrenze" (konservativ: obere KI-Grenze des Verlusts <= Schwelle)
# ============================================================================ #


def base_tag(model_id: str) -> str:
    return re.sub(r"-(?:q\d\w*|fp16|bf16|f16)$", "", model_id)


def main():
    ex.fix_console()
    parser = ex.common_parser(__doc__)
    parser.add_argument("--stufen", nargs="+", default=None, choices=[s["stufe"] for s in STUFEN],
                        help="Nur diese Stufen messen (Standard: alle)")
    parser.add_argument("--base-tag", default=None, help="Basis-Tag ohne Stufe, z. B. qwen3:4b-instruct-2507")
    parser.add_argument("--fresh", action="store_true", help="Alte Ergebnisse verwerfen")
    args = parser.parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    max_tokens = args.max_tokens or MAX_TOKENS
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    if len(questions) < 30:
        eu.log.warning("Nur %d Fragen: Qualitätsunterschiede zwischen Stufen sind statistisch kaum belastbar.", len(questions))
    if args.dry_run:
        ex.setup_dry_run(args.judge_model)

    try:
        reg = llm_client.get_model_spec(args.model)
    except Exception:  # noqa: BLE001
        reg = {"model_id": "qwen3:4b-instruct-2507-q8_0", "base_url": "http://localhost:11434/v1"}
    base = args.base_tag or base_tag(reg["model_id"])
    base_url = "http://localhost:11434/v1"
    host = ex.ollama_host(base_url)
    tags = [] if args.dry_run else ex.ollama_list_tags(host)
    print(f"TS-06 | {len(questions)} Fragen | Wiederholungen: {args.repetitions} | Basis: {base}")

    stufen = [s for s in STUFEN if not args.stufen or s["stufe"] in args.stufen]
    jsonl, meta_path = out / "responses_ts06.jsonl", out / "meta_ts06.json"
    records = [] if args.fresh else eu.load_jsonl(jsonl)
    meta = {} if (args.fresh or not meta_path.exists()) else json.loads(meta_path.read_text(encoding="utf-8"))
    status = {}

    # Registry für ALLE Stufen (auch bereits gemessene aus früheren Läufen)
    for i, s in enumerate(STUFEN):
        s["tag"] = TAG_OVERRIDES.get(s["stufe"], f"{base}-{s['stufe']}")
        s["reg_key"] = f"ts06_{s['stufe']}"
        spec = ({"provider": "mock", "mock_response": f"Testantwort Stufe {s['stufe']}. Neun Bundesländer.",
                 "mock_delay_s": 0.02 + 0.01 * (3 - i)} if args.dry_run else
                {"provider": "openai_compatible", "model_id": s["tag"], "base_url": base_url})
        spec.update({"family": "qwen", "display_name": s["stufe"]})
        ex.register_model(s["reg_key"], spec)

    for s in stufen:
        if not args.dry_run and not ex.server_reachable(base_url):
            raise SystemExit(f"Ollama nicht erreichbar ({base_url}). 'ollama serve' starten.")
        if not args.dry_run and not ex.ollama_has_tag(tags, s["tag"]):
            status[s["stufe"]] = f"übersprungen: Tag nicht vorhanden ({s['tag']})"
            eu.log.warning("%s – bitte 'ollama pull %s' ausführen oder Tag unter TAG_OVERRIDES eintragen.",
                           status[s["stufe"]], s["tag"])
            print(f"WARNUNG – {status[s['stufe']]}")
            continue
        if not args.dry_run:
            ex.ollama_unload_all(host)
        recs, m = ex.run_generation_block(s["reg_key"], questions, args.repetitions, SYSTEM_PROMPT, max_tokens,
                                          args.judge_model, not args.no_judge, not args.no_stream, ["ollama"],
                                          desc=f"{s['stufe']:<8}", extra_fields={"stufe": s["stufe"]})
        if not args.dry_run:
            ex.ollama_unload_all(host)
        if m.get("warmup_error"):
            status[s["stufe"]] = f"läuft nicht: {m['warmup_error']}"
            eu.log.warning("%s: %s", s["stufe"], status[s["stufe"]])
            print(f"WARNUNG – {s['stufe']}: {status[s['stufe']]}")
            continue
        records = [r for r in records if r.get("stufe") != s["stufe"]] + recs
        meta[s["stufe"]] = m
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    df = eu.prepare_df(records)
    if df.empty:
        print("Keine Daten – Abbruch.")
        return
    eu.write_csv(df.drop(columns=["prompt", "context"]).rename(columns={"model": "modell_key", "model_name": "modell"}),
                 out / "results_details.csv")
    tested = [s for s in STUFEN if s["stufe"] in set(df["stufe"])]
    ref = tested[-1]["stufe"]                      # höchste getestete Stufe = Referenz
    print(f"Referenzstufe: {ref} (fp16 falls getestet, sonst höchste getestete Stufe)")

    # ---- Kennzahlen je Stufe ----
    summ = eu.summarize_generation(df, {})
    summ["stufe"] = summ["model"].str.replace("ts06_", "", regex=False)
    summ = summ.set_index("stufe").loc[[s["stufe"] for s in tested]].reset_index()
    summ["bpw"] = [next(s["bpw"] for s in STUFEN if s["stufe"] == x) for x in summ["stufe"]]
    v = df[df["valid"]]
    summ["antwortlaenge_zeichen_mean"] = [v[v["stufe"] == x]["antwortlaenge_zeichen"].mean() for x in summ["stufe"]]
    summ["ram_peak_gb"] = [meta[x].get("ram_peak_gb", np.nan) for x in summ["stufe"]]
    summ["ladezeit_s"] = [meta[x].get("ladezeit_ms", np.nan) / 1000 for x in summ["stufe"]]
    ref_len = summ.loc[summ["stufe"] == ref, "antwortlaenge_zeichen_mean"].iloc[0]
    summ["antwortlaenge_abw_pct"] = 100 * (summ["antwortlaenge_zeichen_mean"] - ref_len) / ref_len if ref_len else np.nan

    # ---- Qualitätsverlust (gepaart je Frage), Textähnlichkeit zur Referenz ----
    qcol = pd.to_numeric(df["antwortqualitaet"], errors="coerce")
    qdf = df.assign(q=qcol)[df["valid"]].groupby(["stufe", "question_id"])["q"].mean().unstack(0)
    refdf = df[(df["stufe"] == ref) & df["valid"]].set_index(["question_id", "repetition"])["response_text"]
    ref_fact = summ.loc[summ["stufe"] == ref, "faktentreffer_pct"].iloc[0]
    cols = {"verlust": [], "ci_lo": [], "ci_hi": [], "sim": [], "fakt_verlust": []}
    for x in summ["stufe"]:
        d = (qdf[ref] - qdf[x]).dropna() if (ref in qdf and x in qdf) else pd.Series(dtype=float)
        mean, lo, hi = ex.bootstrap_ci(d.values)
        cols["verlust"].append(mean if x != ref else 0.0)
        cols["ci_lo"].append(lo if x != ref else 0.0)
        cols["ci_hi"].append(hi if x != ref else 0.0)
        g = df[(df["stufe"] == x) & df["valid"]]
        sims = [ex.text_similarity(r["response_text"], refdf.get((r["question_id"], r["repetition"]), "")) for _, r in g.iterrows()]
        cols["sim"].append(np.nanmean(sims) if len(sims) else np.nan)
        cols["fakt_verlust"].append(ref_fact - summ.loc[summ["stufe"] == x, "faktentreffer_pct"].iloc[0])
    summ["qualitaetsverlust"] = cols["verlust"]
    summ["qualitaetsverlust_ci95_unten"], summ["qualitaetsverlust_ci95_oben"] = cols["ci_lo"], cols["ci_hi"]
    summ["aehnlichkeit_zur_referenz_pct"] = cols["sim"]
    summ["faktentreffer_verlust_pp"] = cols["fakt_verlust"]

    # ---- Entscheidung: kleinste Stufe unter der Schwelle ----
    metric = "qualitaetsverlust_ci95_oben" if ENTSCHEIDUNG_NACH == "ci_obergrenze" else "qualitaetsverlust"
    summ["schwelle_eingehalten"] = ((summ[metric].fillna(np.inf) <= QUALITAETSVERLUST_MAX) &
                                    (summ["faktentreffer_verlust_pp"].fillna(0) <= FAKTENVERLUST_MAX_PP))
    ok = summ[summ["schwelle_eingehalten"]]
    empf = ok.sort_values("bpw").iloc[0]["stufe"] if not ok.empty else None
    summ["empfohlen"] = summ["stufe"] == empf
    summ["referenzstufe"] = summ["stufe"] == ref
    # nicht gemessene Stufen als Zeilen mit Status ergänzen
    missing = pd.DataFrame([{"stufe": s["stufe"], "bpw": s["bpw"], "status": status.get(s["stufe"], "nicht getestet")}
                            for s in STUFEN if s["stufe"] not in set(summ["stufe"])])
    summ["status"] = "getestet"
    full = pd.concat([summ, missing], ignore_index=True).sort_values("bpw")
    eu.write_csv(full.drop(columns=["model", "modell", "familie"], errors="ignore").round(3), out / "results_summary.csv")

    # ---- Diagramme ----
    labels = summ["stufe"].tolist()
    eu.plot_grouped_bars(summ, "stufe", {"Antwortqualität": "antwortqualitaet_mean", "Deutschqualität": "deutschqualitaet_mean"},
                         "TS-06: Qualität je Quantisierungsstufe (Judge, 1–5)", "Bewertung (1 = schlecht, 5 = sehr gut)",
                         out / "diagramm_qualitaet.png", err_cols={"Antwortqualität": "antwortqualitaet_std",
                                                                   "Deutschqualität": "deutschqualitaet_std"}, ylim=(0, 5.6))
    eu.plot_single_bars(labels, summ["tps_mean"], "TS-06: Geschwindigkeit je Stufe", "Token pro Sekunde (höher = besser)",
                        out / "diagramm_tps.png", fmt="{:.1f}")
    eu.plot_single_bars(labels, summ["ram_peak_gb"], "TS-06: Arbeitsspeicher-Spitzenwert je Stufe (statt VRAM)",
                        "RAM-Peak (GB, niedriger = besser)", out / "diagramm_ram.png", fmt="{:.1f}")
    eu.plot_box({s: v[v["stufe"] == s]["tps"].tolist() for s in labels}, "Token pro Sekunde",
                "TS-06: Streuung der Geschwindigkeit", out / "diagramm_tps_boxplot.png")
    eu.plot_grouped_bars(summ, "stufe", {"Textähnlichkeit zur Referenz (%)": "aehnlichkeit_zur_referenz_pct"},
                         f"TS-06: Ähnlichkeit der Antworten zur Referenzstufe ({ref})", "Ähnlichkeit (%)",
                         out / "diagramm_aehnlichkeit.png", ylim=(0, 115))
    pivot = v.assign(q=pd.to_numeric(v["antwortqualitaet"], errors="coerce")).pivot_table(
        index="category", columns="stufe", values="q", aggfunc="mean")[labels]
    if pivot.notna().any().any():
        eu.plot_heatmap(pivot, "TS-06: Antwortqualität nach Fragekategorie und Stufe", out / "diagramm_heatmap_kategorien.png",
                        "Ø Antwortqualität (1–5)", vmin=1, vmax=5, xlabel="Quantisierungsstufe", ylabel="Fragekategorie")
    ex.plot_scatter_labeled(summ["ram_peak_gb"], summ["qualitaetsverlust"], labels,
                            "TS-06: Abwägung Speicherbedarf gegen Qualitätsverlust", "RAM-Peak (GB)",
                            f"Qualitätsverlust gegenüber {ref} (Skalenpunkte)", out / "diagramm_abwaegung.png")

    pd.set_option("display.width", 220)
    print("\n=== Ergebnis TS-06 ===")
    show = ["stufe", "antwortqualitaet_mean", "qualitaetsverlust", "qualitaetsverlust_ci95_oben", "aehnlichkeit_zur_referenz_pct",
            "antwortlaenge_abw_pct", "tps_mean", "ram_peak_gb", "schwelle_eingehalten", "empfohlen"]
    print(summ[show].round(2).to_string(index=False))
    print(f"\nEMPFEHLUNG: {empf if empf else 'keine Stufe erfüllt die Schwelle'} "
          f"(Schwelle {QUALITAETSVERLUST_MAX} Skalenpunkte, Referenz {ref})")
    print(f"Dateien in: {out}")


if __name__ == "__main__":
    main()
