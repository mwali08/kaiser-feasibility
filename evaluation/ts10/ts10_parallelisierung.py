#!/usr/bin/env python3
"""
TS-10: Parallelisierung - sequenziell (Primary, dann Secondary) vs. parallel (beide gleichzeitig)

Erwartung auf reiner CPU: Parallel ist NICHT schneller (beide Modelle teilen sich dieselben Kerne und
dieselbe Speicherbandbreite), eher langsamer durch Kontextwechsel/Cache-Verdrängung. Das Skript misst genau das:
  * Gesamtlatenz je Durchlauf (Start bis beide Antworten da): Median, P95, Mittelwert
  * RAM-Peak der Ollama-Prozesse (psutil) je Durchlauf
  * Speedup = Latenz sequenziell / Latenz parallel je Frage und Wiederholung (> 1 = parallel schneller)
Fairness: Beide Modelle werden vorab geladen (Warm-up), die Reihenfolge der Varianten wechselt je Durchlauf.

WICHTIG: Beide Modelle müssen gleichzeitig im RAM bleiben. Vor 'ollama serve' setzen (PowerShell):
    $env:OLLAMA_MAX_LOADED_MODELS = "2"; $env:OLLAMA_KEEP_ALIVE = "30m"
Nur wenn Ollama beide Modelle parallel bedienen KANN, gibt es überhaupt Überlappung.

Beispiel:  python ts10_parallelisierung.py --repetitions 3 --primary qwen3 --secondary gemma3
Trockenlauf: python ts10_parallelisierung.py --dry-run --out .\\dryrun
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
                 "Wenn ein Kontext angegeben ist, stütze dich ausschließlich darauf und erfinde nichts dazu.")
MAX_TOKENS = 150          # kurz halten: jede Frage wird 2 x 2 Mal gestellt (CPU!)
PROCESS_NAMES = ["ollama"]
# ============================================================================ #


def call(key, prompt, max_tokens):
    try:
        return query_model(key, prompt, stream=False, temperature=0.0, system_prompt=SYSTEM_PROMPT,
                           max_tokens=max_tokens, retries=0)
    except Exception as exc:  # noqa: BLE001
        return {"response_text": "", "error": f"{type(exc).__name__}: {exc}", "total_latency_ms": np.nan, "token_count": 0}


def run_once(mode, primary, secondary, prompt, max_tokens):
    """Rückgabe: Gesamtlatenz (ms), Ergebnis Primary, Ergebnis Secondary, RAM-Meta."""
    with ex.RAMMonitor(PROCESS_NAMES, interval=0.25) as mon:
        t0 = time.perf_counter()
        if mode == "sequenziell":
            r1 = call(primary, prompt, max_tokens)
            r2 = call(secondary, prompt, max_tokens)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f1, f2 = pool.submit(call, primary, prompt, max_tokens), pool.submit(call, secondary, prompt, max_tokens)
                r1, r2 = f1.result(), f2.result()
        wall = (time.perf_counter() - t0) * 1000
    return wall, r1, r2, mon.meta


def main():
    ex.fix_console()
    parser = ex.common_parser(__doc__)
    parser.add_argument("--primary", default="qwen3", help="Primary-Modell (Kurzname)")
    parser.add_argument("--secondary", default="gemma3", help="Secondary-Modell (Kurzname, andere Familie)")
    args = parser.parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    max_tokens = args.max_tokens or MAX_TOKENS
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    if args.dry_run:
        ex.setup_dry_run(args.judge_model)
        ex.register_model(args.primary, {"provider": "mock", "family": "qwen", "mock_delay_s": 0.15, "mock_response": "Testantwort A."})
        ex.register_model(args.secondary, {"provider": "mock", "family": "google", "mock_delay_s": 0.1, "mock_response": "Testantwort B."})
    print(f"TS-10 | {len(questions)} Fragen | Primary: {args.primary} | Secondary: {args.secondary} | "
          f"Wiederholungen: {args.repetitions} | max_tokens: {max_tokens}")
    for key in (args.primary, args.secondary):           # beide Modelle laden (nicht messen)
        w = call(key, "Antworte mit einem Wort: Hallo.", 8)
        if w["error"]:
            raise SystemExit(f"Warm-up für {key} fehlgeschlagen: {w['error']}")

    rows = []
    todo = [(q, rep) for rep in range(1, args.repetitions + 1) for q in questions]
    for idx, (q, rep) in enumerate(tqdm(todo, desc="TS-10", unit="Frage")):
        prompt = eu.build_prompt(q)
        for mode in (("sequenziell", "parallel") if idx % 2 == 0 else ("parallel", "sequenziell")):
            wall, r1, r2, ram = run_once(mode, args.primary, args.secondary, prompt, max_tokens)
            err = r1["error"] or r2["error"]
            if err:
                eu.log.warning("Fehler | %s | %s | Rep %d | %s", mode, q["id"], rep, err)
            rows.append({"variante": mode, "question_id": q["id"], "repetition": rep, "gesamtlatenz_ms": wall,
                         "primary_ms": r1["total_latency_ms"], "secondary_ms": r2["total_latency_ms"],
                         "token_count": r1["token_count"] + r2["token_count"], "error": err, "valid": err is None, **ram})
    df = pd.DataFrame(rows)
    eu.write_csv(df, out / "results_details.csv")
    v = df[df["valid"]]

    # ---- Zusammenfassung ----
    names = ["sequenziell", "parallel"]
    summ = []
    for name in names:
        g = v[v["variante"] == name]
        summ.append({"variante": name, "n_durchlaeufe": len(df[df["variante"] == name]), "n_gueltig": len(g),
                     "latenz_median_ms": g["gesamtlatenz_ms"].median(), "latenz_mean_ms": g["gesamtlatenz_ms"].mean(),
                     "latenz_std_ms": g["gesamtlatenz_ms"].std(), "latenz_p95_ms": pd_.pct(g["gesamtlatenz_ms"], 95),
                     "ram_peak_gb_mean": g["ram_peak_gb"].mean(), "ram_peak_gb_max": g["ram_peak_gb"].max(),
                     "ram_system_delta_gb_max": g["ram_system_delta_gb"].max(), "token_count_mean": g["token_count"].mean()})
    summary = pd.DataFrame(summ)
    wide = v.pivot_table(index=["question_id", "repetition"], columns="variante", values="gesamtlatenz_ms").dropna()
    speed = (wide["sequenziell"] / wide["parallel"]) if len(wide) else pd.Series(dtype=float)
    mean_sp, lo, hi = ex.bootstrap_ci(speed.values)
    summary["speedup_mean_seq_durch_par"] = [np.nan, mean_sp]
    summary["speedup_ci95_unten"], summary["speedup_ci95_oben"] = [np.nan, lo], [np.nan, hi]
    summary["parallel_schneller_pct"] = [np.nan, 100 * (speed > 1).mean() if len(speed) else np.nan]
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # ---- Diagramme ----
    per_q = v.groupby(["question_id", "variante"])["gesamtlatenz_ms"].median().div(1000).unstack("variante")
    per_q = per_q.reindex(index=[q["id"] for q in questions]).dropna(how="all")
    ex.plot_lines(per_q.index.tolist(), {"Sequenziell (Primary, dann Secondary)": per_q["sequenziell"].tolist(),
                                         "Parallel (beide gleichzeitig)": per_q["parallel"].tolist()},
                  "TS-10: Gesamtlatenz je Frage - sequenziell vs. parallel", "Frage (ID)",
                  "Gesamtlatenz (s, Median über Wiederholungen)", out / "diagramm_latenz_linie.png")
    eu.plot_grouped_bars(summary, "variante", {"RAM-Peak Mittel": "ram_peak_gb_mean", "RAM-Peak Maximum": "ram_peak_gb_max"},
                         "TS-10: Arbeitsspeicher-Spitzenwert der Ollama-Prozesse", "RAM (GB, niedriger = besser)",
                         out / "diagramm_ram_peak.png")
    eu.plot_box({n: v[v["variante"] == n]["gesamtlatenz_ms"].div(1000).tolist() for n in names},
                "Gesamtlatenz (s)", "TS-10: Streuung der Gesamtlatenz", out / "diagramm_latenz_boxplot.png")
    eu.plot_grouped_bars(summary, "variante", {"Median": "latenz_median_ms", "95. Perzentil (P95)": "latenz_p95_ms"},
                         "TS-10: Typische und schlechte Antwortzeit", "Gesamtlatenz (ms)", out / "diagramm_latenz_median_p95.png")

    pd.set_option("display.width", 200)
    print("\n=== Ergebnis TS-10 ===")
    print(summary[["variante", "n_gueltig", "latenz_median_ms", "latenz_p95_ms", "ram_peak_gb_mean", "ram_peak_gb_max"]].round(1).to_string(index=False))
    print(f"\nSpeedup sequenziell/parallel: Ø {mean_sp:.2f}x, 95-%-KI [{lo:.2f}; {hi:.2f}] (>1 = parallel schneller)")
    print(f"Dateien in: {out}")


if __name__ == "__main__":
    main()