#!/usr/bin/env python3
"""
TS-05: Serving-Engine und Quantisierungsverfahren
Konfigurationen: llama.cpp (GGUF), Ollama (GGUF), vLLM (AWQ), vLLM (GPTQ)
Modell: Primary Actor aus Phase B (qwen3 = qwen3:4b-instruct-2507), gleiche Bitbreite, gleiche Prompts/Parameter.

ABWEICHUNGEN VOM URSPRÜNGLICHEN PLAN (siehe auch shared/eval_utils_extension.py):
  1. Mistral Small 4 (24B) -> Mistral 7B (Hardware) – für TS-05 nicht relevant, nur der Vollständigkeit halber.
  2. Qwen3 standard -> qwen3:4b-instruct-2507 (Thinking-Mode ungeeignet für den Router).
  3. VRAM nicht messbar (keine GPU) -> RAM-Peak der Engine-Prozesse via psutil.
  4. Perplexity über Ollama nicht verfügbar -> Judge-basierte Qualitätsbewertung + Textähnlichkeit zur Baseline.
  5. vLLM braucht eine GPU -> nicht testbar, wird mit Warnung übersprungen (erscheint trotzdem in der Ergebnistabelle).

ABLAUF / BEDIENUNG
  * Es wird je Konfiguration ein laufender Server erwartet (Ollama: ollama serve; llama.cpp: llama-server).
    Startbeispiel llama.cpp (gleiche GGUF-Datei wie in Ollama, gleiche Threads/Kontext):
        llama-server -m C:\\Modelle\\qwen3-4b-instruct-2507-Q8_0.gguf -c 4096 --port 8080 -t 8
  * Konfigurationen einzeln messen (empfohlen, damit sich die Engines nicht gegenseitig stören):
        python ts05_serving_engine.py --configs llamacpp_gguf
        python ts05_serving_engine.py --configs ollama_gguf
    Ergebnisse früherer Läufe bleiben erhalten; Summary/Diagramme werden aus ALLEN Daten neu erzeugt.
  * Ablauf je Konfiguration: Warm-up (Ladezeit) -> Generierung unter RAM-Messung -> danach Judge.
  * Trockenlauf ohne Server:  python ts05_serving_engine.py --dry-run --out .\\dryrun

KO-Kriterium: Offline-Betrieb. Danach Nutzwertanalyse (nur getestete Konfigurationen).
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
MAX_TOKENS = 300                 # identisch für alle Konfigurationen (auf CPU begrenzen, sonst Laufzeit zu lang)
BASELINE_CONFIG = "ollama_gguf"  # Referenz für Qualitätsverlust und Textähnlichkeit

# Die Einrichtungs-/Installations-Scores (1 = sehr aufwendig, 5 = sehr einfach) sind SUBJEKTIVE Einschätzungen
# des Autors und müssen an die eigenen Erfahrungen angepasst werden. Die Kommentare sind der qualitative Teil.
# bits=None -> wird aus dem Ollama-Tag abgeleitet (qwen3: ...-q8_0 -> 8). AWQ/GPTQ sind 4-Bit-Verfahren.
ENGINES = [
    {"key": "llamacpp_gguf", "engine": "llama.cpp", "format": "GGUF", "base_url": "http://localhost:8080/v1",
     "model_id": "ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF:Q8_0", "process_names": ["llama-server"], "needs_gpu": False, "bits": None,
     "offline_betrieb": True, "offline_install_score": 5, "einrichtung_score": 3,
     "einrichtung_kommentar": "Statische Binaries + GGUF-Datei, kein Installer/keine Abhängigkeiten; "
                              "Parameter (Threads, Kontext) per Kommandozeile; kein Modell-Management."},
    {"key": "ollama_gguf", "engine": "Ollama", "format": "GGUF", "base_url": None, "model_id": None,
     "process_names": ["ollama"], "needs_gpu": False, "bits": None,
     "offline_betrieb": True, "offline_install_score": 4, "einrichtung_score": 5,
     "einrichtung_kommentar": "Ein Installer, Modell per 'ollama pull' (einmalig online) oder offline per "
                              "Modelfile aus GGUF; Modell-Management und API sofort verfügbar."},
    {"key": "vllm_awq", "engine": "vLLM", "format": "AWQ", "base_url": "http://localhost:8000/v1",
     "model_id": "qwen3-awq", "process_names": ["vllm", "python"], "needs_gpu": True, "bits": 4,
     "offline_betrieb": True, "offline_install_score": 2, "einrichtung_score": 2,
     "einrichtung_kommentar": "NICHT TESTBAR (keine GPU). Erwartung: Python-Umgebung mit CUDA/torch, unter Windows nur "
                              "via WSL2/Linux; viele Abhängigkeiten -> Offline-Installation nur mit Wheelhouse."},
    {"key": "vllm_gptq", "engine": "vLLM", "format": "GPTQ", "base_url": "http://localhost:8000/v1",
     "model_id": "qwen3-gptq", "process_names": ["vllm", "python"], "needs_gpu": True, "bits": 4,
     "offline_betrieb": True, "offline_install_score": 2, "einrichtung_score": 2,
     "einrichtung_kommentar": "NICHT TESTBAR (keine GPU). Wie vLLM (AWQ); zusätzlich GPTQ-Kernel/Kompatibilität prüfen."},
]

# Nutzwertanalyse (Gewichte Summe 1.0; Ankerwerte für CPU-Betrieb – an Anforderungen anpassen)
NUTZWERT_KRITERIEN = {
    "tps_mean":              {"weight": 0.25, "worst": 2.0,     "best": 15.0},
    "ttft_ms_mean":          {"weight": 0.15, "worst": 30000.0, "best": 1000.0},
    "ram_peak_gb":           {"weight": 0.10, "worst": 24.0,    "best": 4.0},
    "antwortqualitaet_mean": {"weight": 0.25, "worst": 1.0,     "best": 5.0},
    "einrichtung_score":     {"weight": 0.15, "worst": 1.0,     "best": 5.0},
    "offline_install_score": {"weight": 0.10, "worst": 1.0,     "best": 5.0},
}
# KO: Offline-Betrieb muss möglich sein (TODO: ggf. weitere KO-Kriterien der Arbeit ergänzen)
KO_KRITERIEN = {"offline_betrieb_num": {"min": 1.0, "label": "KO: Offline-Betrieb"}}
RADAR_LABELS = {"tps_mean": "Geschwindigkeit (TPS)", "ttft_ms_mean": "Antwortstart (TTFT)",
                "ram_peak_gb": "Speichereffizienz (RAM)", "antwortqualitaet_mean": "Antwortqualität",
                "einrichtung_score": "Einrichtung", "offline_install_score": "Offline-Installation"}
# ============================================================================ #


def derive_bits(tag: str):
    m = re.search(r"q(\d)", tag.lower())
    return int(m.group(1)) if m else (16 if re.search(r"fp16|f16|bf16", tag.lower()) else None)


def build_configs(args):
    try:
        reg = llm_client.get_model_spec(args.model)
    except Exception:  # noqa: BLE001
        reg = {"model_id": "qwen3:4b-instruct-2507-q8_0", "base_url": "http://localhost:11434/v1"}
    bits = derive_bits(reg["model_id"])
    cfgs = []
    for e in ENGINES:
        c = dict(e)
        if c["key"] == "ollama_gguf":
            c["base_url"], c["model_id"] = "http://localhost:11434/v1", reg["model_id"]
        if c["bits"] is None:
            c["bits"] = bits
        c["label"] = f"{c['engine']} ({c['format']})"
        c["reg_key"] = f"ts05_{c['key']}"
        spec = ({"provider": "mock", "mock_response": f"Testantwort ({c['key']}). Neun Bundesländer.",
                 "mock_delay_s": 0.02 + 0.01 * len(cfgs)} if args.dry_run else
                {"provider": "openai_compatible", "model_id": c["model_id"], "base_url": c["base_url"]})
        spec.update({"family": "qwen", "display_name": c["label"]})
        ex.register_model(c["reg_key"], spec)
        cfgs.append(c)
    return cfgs


def main():
    ex.fix_console()
    parser = ex.common_parser(__doc__)
    parser.add_argument("--configs", nargs="+", default=None, choices=[e["key"] for e in ENGINES],
                        help="Nur diese Konfigurationen messen (Standard: alle)")
    parser.add_argument("--fresh", action="store_true", help="Alte Ergebnisse verwerfen")
    parser.add_argument("--check-internet", action="store_true",
                        help="Je Lauf prüfen/protokollieren, ob Internet erreichbar ist (Nachweis Offline-Betrieb)")
    args = parser.parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    max_tokens = args.max_tokens or MAX_TOKENS
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    if args.dry_run:
        ex.setup_dry_run(args.judge_model)
    configs = build_configs(args)
    selected = [c for c in configs if not args.configs or c["key"] in args.configs]
    print(f"TS-05 | {len(questions)} Fragen | Wiederholungen: {args.repetitions} | max_tokens: {max_tokens}")
    if len({c["bits"] for c in configs if c["bits"]}) > 1:
        eu.log.warning("Unterschiedliche Bitbreiten (%s): AWQ/GPTQ sind 4-Bit; die GGUF-Variante auf Q4 setzen, "
                       "wenn vLLM später (mit GPU) verglichen werden soll.", sorted({c['bits'] for c in configs if c['bits']}))

    jsonl, meta_path = out / "responses_ts05.jsonl", out / "meta_ts05.json"
    records = [] if args.fresh else eu.load_jsonl(jsonl)
    meta = {} if (args.fresh or not meta_path.exists()) else json.loads(meta_path.read_text(encoding="utf-8"))
    skip_reason = {}
    gpu = ex.has_gpu()

    for cfg in selected:
        if cfg["needs_gpu"] and not gpu:
            skip_reason[cfg["key"]] = "übersprungen: keine GPU (vLLM nicht testbar)"
        elif not args.dry_run and not ex.server_reachable(cfg["base_url"]):
            skip_reason[cfg["key"]] = f"übersprungen: Server nicht erreichbar ({cfg['base_url']})"
        if cfg["key"] in skip_reason:
            eu.log.warning("%s: %s", cfg["label"], skip_reason[cfg["key"]])
            print(f"WARNUNG – {cfg['label']}: {skip_reason[cfg['key']]}")
            continue
        host = ex.ollama_host(cfg["base_url"]) if cfg["engine"] == "Ollama" else None
        if host and not args.dry_run:
            ex.ollama_unload_all(host)      # gleiche Startbedingungen: kein Modell geladen
        recs, m = ex.run_generation_block(cfg["reg_key"], questions, args.repetitions, SYSTEM_PROMPT, max_tokens,
                                          args.judge_model, not args.no_judge, not args.no_stream,
                                          cfg["process_names"], desc=f"{cfg['label']:<18}",
                                          extra_fields={"konfig": cfg["key"]})
        if host and not args.dry_run:
            ex.ollama_unload_all(host)
        if m.get("warmup_error"):
            skip_reason[cfg["key"]] = f"Fehler im Warm-up: {m['warmup_error']}"
            eu.log.warning("%s: %s", cfg["label"], skip_reason[cfg["key"]])
            continue
        if args.check_internet:
            m["internet_erreichbar"] = ex.internet_reachable()
        records = [r for r in records if r.get("konfig") != cfg["key"]] + recs
        meta[cfg["key"]] = m
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    df = eu.prepare_df(records)
    tested = set(df["konfig"]) if not df.empty else set()

    # ---- Details ----
    if not df.empty:
        eu.write_csv(df.drop(columns=["prompt", "context"]).rename(columns={"model": "modell_key", "model_name": "modell"}),
                     out / "results_details.csv")
        base = eu.summarize_generation(df, {})
        base["konfig"] = base["model"].str.replace("ts05_", "", regex=False)
        base = base.set_index("konfig")
    rows = []
    for cfg in configs:
        row = {"konfig": cfg["key"], "konfiguration": cfg["label"], "engine": cfg["engine"], "format": cfg["format"],
               "bitbreite": cfg["bits"], "einrichtung_score": cfg["einrichtung_score"],
               "offline_install_score": cfg["offline_install_score"], "offline_betrieb_num": float(cfg["offline_betrieb"]),
               "einrichtung_kommentar": cfg["einrichtung_kommentar"]}
        if cfg["key"] in tested:
            row["status"] = "getestet"
            row.update({k: v for k, v in base.loc[cfg["key"]].to_dict().items() if k not in ("model", "modell", "familie")})
            m = meta.get(cfg["key"], {})
            row["ram_peak_gb"], row["ram_delta_gb"] = m.get("ram_peak_gb", np.nan), m.get("ram_delta_gb", np.nan)
            row["ram_system_delta_gb"], row["ladezeit_s"] = m.get("ram_system_delta_gb", np.nan), m.get("ladezeit_ms", np.nan) / 1000
            if "internet_erreichbar" in m:
                row["lauf_ohne_internet"] = not m["internet_erreichbar"]
        else:
            row["status"] = skip_reason.get(cfg["key"], "nicht getestet (noch nicht ausgeführt)")
        rows.append(row)
    summary = pd.DataFrame(rows)

    # ---- Qualitätsverlust und Textähnlichkeit zur Baseline ----
    summary["qualitaetsverlust_vs_baseline"] = np.nan
    summary["aehnlichkeit_zur_baseline_pct"] = np.nan
    if BASELINE_CONFIG in tested:
        bq = summary.loc[summary["konfig"] == BASELINE_CONFIG, "antwortqualitaet_mean"].iloc[0]
        bdf = df[(df["konfig"] == BASELINE_CONFIG) & df["valid"]].set_index(["question_id", "repetition"])["response_text"]
        for i, r in summary.iterrows():
            if r["status"] != "getestet":
                continue
            summary.loc[i, "qualitaetsverlust_vs_baseline"] = bq - r["antwortqualitaet_mean"]
            g = df[(df["konfig"] == r["konfig"]) & df["valid"]]
            sims = [ex.text_similarity(rec["response_text"], bdf.get((rec["question_id"], rec["repetition"]), ""))
                    for _, rec in g.iterrows()]
            summary.loc[i, "aehnlichkeit_zur_baseline_pct"] = np.nanmean(sims) if len(sims) else np.nan
    else:
        eu.log.warning("Baseline '%s' nicht getestet – Qualitätsverlust/Ähnlichkeit nicht berechenbar.", BASELINE_CONFIG)

    # ---- KO + Nutzwert (nur getestete Konfigurationen) ----
    ko = summary.apply(lambda r: eu.check_ko(r, KO_KRITERIEN), axis=1)
    summary["ko_bestanden"], summary["ko_hinweise"] = [k[0] for k in ko], [k[1] for k in ko]
    parts_rows = {}
    nw_vals, cov = [], []
    for _, r in summary.iterrows():
        if r["status"] != "getestet":
            nw_vals.append(np.nan)
            cov.append(np.nan)
            continue
        nw, c, parts = eu.nutzwert(r, NUTZWERT_KRITERIEN)
        nw_vals.append(nw)
        cov.append(c)
        parts_rows[r["konfiguration"]] = parts
    summary["nutzwert"], summary["nutzwert_gewichtsabdeckung"] = nw_vals, cov
    ok = summary["ko_bestanden"] & summary["nutzwert"].notna()
    summary["rang_nutzwert"] = summary.loc[ok, "nutzwert"].rank(ascending=False, method="min")
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # ---- Diagramme (nur getestete Konfigurationen) ----
    t = summary[summary["status"] == "getestet"]
    if t.empty:
        print("Keine getesteten Konfigurationen – keine Diagramme.")
    else:
        labels = t["konfiguration"].tolist()
        eu.plot_single_bars(labels, t["tps_mean"], "TS-05: Generierungsgeschwindigkeit (Token pro Sekunde)",
                            "Token pro Sekunde (höher = besser)", out / "diagramm_tps.png", fmt="{:.1f}")
        eu.plot_box({c["label"]: df[(df["konfig"] == c["key"]) & df["valid"]]["ttft_ms"].tolist()
                     for c in configs if c["key"] in tested},
                    "Zeit bis zum ersten Token (ms)", "TS-05: Antwortstart (TTFT) je Konfiguration",
                    out / "diagramm_ttft_boxplot.png")
        eu.plot_single_bars(labels, t["ram_peak_gb"], "TS-05: Arbeitsspeicher-Spitzenwert der Engine (statt VRAM)",
                            "RAM-Peak (GB, niedriger = besser)", out / "diagramm_ram.png", fmt="{:.1f}")
        eu.plot_grouped_bars(t, "konfiguration", {"Antwortqualität (Judge)": "antwortqualitaet_mean",
                                                  "Deutschqualität (Judge)": "deutschqualitaet_mean"},
                             "TS-05: Qualität der Antworten (Mittelwert, 1–5)", "Bewertung (1 = schlecht, 5 = sehr gut)",
                             out / "diagramm_qualitaet.png",
                             err_cols={"Antwortqualität (Judge)": "antwortqualitaet_std",
                                       "Deutschqualität (Judge)": "deutschqualitaet_std"}, ylim=(0, 5.6))
        hm = pd.DataFrame(parts_rows).T[list(RADAR_LABELS)]
        hm.columns = list(RADAR_LABELS.values())
        eu.plot_heatmap(hm, "TS-05: Erfüllungsgrad je Kriterium (0 = schlecht, 1 = bestes Niveau)",
                        out / "diagramm_heatmap_kriterien.png", "Erfüllungsgrad", fmt="{:.2f}", vmin=0, vmax=1,
                        ylabel="Konfiguration")
        eu.plot_radar({l: [parts_rows[l][c] for c in RADAR_LABELS] for l in parts_rows},
                      list(RADAR_LABELS.values()), "TS-05: Stärken-Schwächen-Profil", out / "diagramm_radar.png")
        eu.plot_single_bars(labels, t["nutzwert"], "TS-05: Nutzwert (0–100 Punkte)", "Nutzwert",
                            out / "diagramm_nutzwert.png", ylim=(0, 100))

    pd.set_option("display.width", 220)
    print("\n=== Ergebnis TS-05 ===")
    show = ["konfiguration", "status", "tps_mean", "ttft_ms_mean", "ram_peak_gb", "antwortqualitaet_mean",
            "qualitaetsverlust_vs_baseline", "ko_bestanden", "nutzwert", "rang_nutzwert"]
    print(summary[[c for c in show if c in summary]].round(2).to_string(index=False))
    print(f"\nDateien in: {out}")


if __name__ == "__main__":
    main()
