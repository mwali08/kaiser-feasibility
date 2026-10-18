#!/usr/bin/env python3
"""
TS-07: Mehrbenutzer-Skalierung (Engine aus TS-05: Ollama; optional llama.cpp via --engine llamacpp)

Aufbau: Stufenweise steigende Zahl GLEICHZEITIGER Anfragen (1, 2, 4, 8). Jeder simulierte Nutzer sendet
nacheinander `--requests-per-user` Anfragen (geschlossene Last, keine Denkpause = Worst Case je Nutzerzahl).
Alle Nutzer einer Stufe starten gleichzeitig (Barrier).

Messgrößen je Stufe: Durchsatz (Anfragen/min und Token/s gesamt), Latenz (Median, P95), TTFT (Median, P95),
Token/s je Anfrage, Timeout-/Fehlerrate (%), RAM-Peak der Engine-Prozesse.
  * Latenz = Zeit von Senden bis Antwortende aus Sicht des Clients, INKLUSIVE Wartezeit in der Server-Warteschlange.
  * Timeout = Client-Timeout oder Antwort länger als --timeout-s (gilt zugleich als Latenz-Obergrenze).
  * Latenz-Perzentile beruhen nur auf erfolgreichen Anfragen; Fehler/Timeouts stehen in der Fehlerrate.
  * Für belastbare P95-Werte mindestens ~20 erfolgreiche Anfragen je Stufe (--requests-per-user, --repetitions erhöhen).

ANNAHME (konfigurierbar, unten bzw. per CLI): erwartete Nutzerzahl = 4 gleichzeitige Nutzer; Zielwerte:
P95-Latenz <= 120 s, Fehlerrate <= 5 %, >= 1 Token/s je Anfrage (CPU-Betrieb!). Die Zielwerte sind PLATZHALTER.

WICHTIG: Ollama bedient Anfragen nur parallel, wenn der Server entsprechend konfiguriert ist (Umgebungsvariablen
OLLAMA_NUM_PARALLEL, OLLAMA_MAX_QUEUE vor 'ollama serve' setzen). Ohne Parallelität werden Anfragen
sequentiell abgearbeitet -> Latenz steigt linear (genau das soll die Messung zeigen). Variante mit anderer
Server-Konfiguration: Server neu starten und mit --label parallel4 erneut messen (eigene Ausgabedateien).

ABWEICHUNGEN VOM URSPRÜNGLICHEN PLAN: siehe shared/eval_utils_extension.py (VRAM -> RAM-Peak, qwen3:4b-instruct-2507).

Beispiel:  python ts07_skalierung.py --requests-per-user 3 --expected-users 4
Trockenlauf: python ts07_skalierung.py --dry-run --out .\\dryrun
"""
import re
import sys
import threading
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
import llm_client  # noqa: E402
from llm_client import query_model  # noqa: E402

# =========================== KONFIGURATION ================================== #
SYSTEM_PROMPT = ("Du bist ein hilfreicher, präziser Assistent. Antworte auf Deutsch, sachlich und knapp. "
                 "Wenn ein Kontext angegeben ist, stütze dich ausschließlich darauf und erfinde nichts dazu. "
                 "Wenn die Antwort nicht im Kontext steht, sage das ausdrücklich.")
MAX_TOKENS = 200                 # kurz halten, sonst dauert ein Lauf auf CPU zu lange

CONCURRENCY_LEVELS = [1, 2, 4, 8]
# HINWEIS: Höhere Stufen (25, 50 gleichzeitige Anfragen) sind bei reiner CPU-Inferenz unrealistisch
# (Laufzeit/Timeouts, RAM für KV-Caches) und bewusst auskommentiert. Mit GPU-Server wieder aktivieren:
# CONCURRENCY_LEVELS = [1, 2, 4, 8, 25, 50]

ERWARTETE_NUTZER = 4             # ANNAHME: unbekannt -> konfigurierbar (--expected-users)
ZIEL_P95_LATENZ_S = 120.0        # PLATZHALTER
ZIEL_FEHLERRATE_PCT = 5.0        # PLATZHALTER
ZIEL_MIN_TPS_PRO_ANFRAGE = 1.0   # PLATZHALTER (gefühlt noch nutzbare Lesegeschwindigkeit)
TIMEOUT_S = 180.0                # Client-Timeout = SLA-Obergrenze je Anfrage
DENKPAUSE_S = 0.0                # Pause eines Nutzers zwischen zwei Anfragen (0 = Worst Case)

ENGINES = {"ollama": {"process_names": ["ollama"], "base_url": "http://localhost:11434/v1", "model_id": None},
           "llamacpp": {"process_names": ["llama-server"], "base_url": "http://localhost:8080/v1", "model_id": "qwen3"}}
# ============================================================================ #


def classify(res: dict, wall_ms: float, timeout_s: float) -> str:
    err = res.get("error")
    if err:
        return "timeout" if re.search(r"timeout|timed out", err, re.I) else "fehler"
    return "timeout" if wall_ms > timeout_s * 1000 else "ok"


def run_level(key, questions, n_users, req_per_user, rep, timeout_s, max_tokens, stream, think_s, process_names):
    barrier = threading.Barrier(n_users)
    pbar = tqdm(total=n_users * req_per_user, desc=f"{n_users:>2} gleichzeitig (Lauf {rep})", unit="Anfrage", leave=False)

    def user(uid):
        rows = []
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            pass
        for i in range(req_per_user):
            q = questions[(uid * req_per_user + i + rep * 7) % len(questions)]
            t0 = time.perf_counter()
            try:
                res = query_model(key, eu.build_prompt(q), stream=stream, temperature=0.0, system_prompt=SYSTEM_PROMPT,
                                  max_tokens=max_tokens, retries=0)
            except Exception as exc:  # noqa: BLE001
                res = {"error": f"{type(exc).__name__}: {exc}", "response_text": "", "ttft_ms": np.nan,
                       "token_count": 0, "tps": np.nan}
            wall = (time.perf_counter() - t0) * 1000
            status = classify(res, wall, timeout_s)
            if status != "ok":
                eu.log.warning("Anfrage %s | Nutzer %d/%d | %s | %s", status, uid, n_users, q["id"], res.get("error"))
            rows.append({"stufe_nutzer": n_users, "lauf": rep, "nutzer": uid, "anfrage_nr": i, "question_id": q["id"],
                         "status": status, "latenz_ms": wall, "ttft_ms": res.get("ttft_ms", np.nan),
                         "token_count": res.get("token_count", 0) if status == "ok" else 0,
                         "tps": res.get("tps", np.nan) if status == "ok" else np.nan, "error": res.get("error")})
            pbar.update(1)
            if think_s:
                time.sleep(think_s)
        return rows

    rows = []
    with ex.RAMMonitor(process_names) as mon:
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_users) as pool:
            for f in [pool.submit(user, u) for u in range(n_users)]:
                rows += f.result()
        wall_s = time.perf_counter() - t_start
    pbar.close()
    ok = [r for r in rows if r["status"] == "ok"]
    run = {"stufe_nutzer": n_users, "lauf": rep, "wall_s": wall_s, "n_anfragen": len(rows), "n_ok": len(ok),
           "durchsatz_anfragen_min": 60 * len(ok) / wall_s if wall_s else np.nan,
           "durchsatz_token_s": sum(r["token_count"] for r in ok) / wall_s if wall_s else np.nan, **mon.meta}
    return rows, run


def pct(a, p):
    a = [x for x in a if not pd.isna(x)]
    return float(np.percentile(a, p)) if a else np.nan


def main():
    ex.fix_console()
    parser = ex.common_parser(__doc__)
    parser.set_defaults(repetitions=1)
    parser.add_argument("--engine", choices=list(ENGINES), default="ollama", help="Engine laut Ergebnis TS-05")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Stufen überschreiben, z. B. --levels 1 2 4")
    parser.add_argument("--requests-per-user", type=int, default=2, help="Anfragen je simuliertem Nutzer und Lauf")
    parser.add_argument("--expected-users", type=int, default=ERWARTETE_NUTZER, help="Annahme: erwartete Nutzerzahl")
    parser.add_argument("--timeout-s", type=float, default=TIMEOUT_S)
    parser.add_argument("--label", default="", help="Etikett der Server-Konfiguration (z. B. parallel4) – Dateisuffix")
    args = parser.parse_args()
    levels = args.levels or CONCURRENCY_LEVELS
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    max_tokens = args.max_tokens or MAX_TOKENS
    suffix = f"_{args.label}" if args.label else ""
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    if args.dry_run:
        ex.setup_dry_run(args.judge_model)
    ex.set_request_timeout(args.timeout_s)

    eng = ENGINES[args.engine]
    try:
        reg = llm_client.get_model_spec(args.model)
    except Exception:  # noqa: BLE001
        reg = {"model_id": "qwen3:4b-instruct-2507-q8_0", "base_url": "http://localhost:11434/v1"}
    base_url, model_id = eng["base_url"] or reg["base_url"], eng["model_id"] or reg["model_id"]
    key = "ts07_engine"
    spec = ({"provider": "mock", "mock_response": "Testantwort Neun Bundesländer.", "mock_delay_s": 0.05} if args.dry_run else
            {"provider": "openai_compatible", "model_id": model_id, "base_url": base_url})
    spec.update({"family": "qwen", "display_name": f"{args.engine}"})
    ex.register_model(key, spec)
    if not args.dry_run and not ex.server_reachable(base_url):
        raise SystemExit(f"Server nicht erreichbar ({base_url}). Engine starten und erneut ausführen.")
    for n in levels:
        if n > 8:
            eu.log.warning("Stufe %d auf CPU unrealistisch (Timeouts, RAM) – bitte bewusst wählen.", n)

    n_total = sum(n * args.requests_per_user * args.repetitions for n in levels)
    print(f"TS-07 | Engine: {args.engine} ({model_id}) | Stufen: {levels} | {args.requests_per_user} Anfragen/Nutzer | "
          f"Läufe: {args.repetitions} | gesamt {n_total} Anfragen | Timeout {args.timeout_s:.0f} s")
    print(f"Annahme erwartete Nutzer: {args.expected_users} | Ziele: P95 <= {ZIEL_P95_LATENZ_S:.0f} s, "
          f"Fehler <= {ZIEL_FEHLERRATE_PCT}%, >= {ZIEL_MIN_TPS_PRO_ANFRAGE} Token/s je Anfrage")

    if not args.dry_run:
        ex.ollama_unload_all(ex.ollama_host(base_url)) if args.engine == "ollama" else None
    w = query_model(key, "Antworte mit einem Wort: Hallo.", stream=False, max_tokens=16, retries=0)   # Warm-up
    if w["error"]:
        raise SystemExit(f"Warm-up fehlgeschlagen: {w['error']}")
    print(f"Warm-up/Ladezeit: {w['total_latency_ms'] / 1000:.1f} s")

    all_rows, runs = [], []
    for n in levels:
        for rep in range(1, args.repetitions + 1):
            rows, run = run_level(key, questions, n, args.requests_per_user, rep, args.timeout_s, max_tokens,
                                  not args.no_stream, DENKPAUSE_S, eng["process_names"])
            all_rows += rows
            runs.append(run)
    df, runs_df = pd.DataFrame(all_rows), pd.DataFrame(runs)
    eu.write_csv(df, out / f"results_details{suffix}.csv")
    eu.write_csv(runs_df.round(3), out / f"results_level_runs{suffix}.csv")

    # ---- Zusammenfassung je Stufe ----
    rows = []
    for n, g in df.groupby("stufe_nutzer"):
        ok, r = g[g["status"] == "ok"], runs_df[runs_df["stufe_nutzer"] == n]
        row = {"engine": args.engine, "label": args.label, "gleichzeitige_anfragen": n, "n_anfragen": len(g),
               "n_ok": len(ok), "timeout_rate_pct": 100 * (g["status"] == "timeout").mean(),
               "fehler_rate_pct": 100 * (g["status"] == "fehler").mean(),
               "ausfallrate_gesamt_pct": 100 * (g["status"] != "ok").mean(),
               "latenz_median_s": pct(ok["latenz_ms"], 50) / 1000, "latenz_p95_s": pct(ok["latenz_ms"], 95) / 1000,
               "ttft_median_s": pct(ok["ttft_ms"], 50) / 1000, "ttft_p95_s": pct(ok["ttft_ms"], 95) / 1000,
               "tps_pro_anfrage_mean": ok["tps"].mean(),
               "durchsatz_anfragen_min": r["durchsatz_anfragen_min"].mean(), "durchsatz_token_s": r["durchsatz_token_s"].mean(),
               "durchsatz_token_s_std": r["durchsatz_token_s"].std(), "ram_peak_gb": r["ram_peak_gb"].max(),
               "ram_system_delta_gb": r["ram_system_delta_gb"].max()}
        row["ziel_erfuellt"] = bool(row["n_ok"] > 0 and row["latenz_p95_s"] <= ZIEL_P95_LATENZ_S
                                    and row["ausfallrate_gesamt_pct"] <= ZIEL_FEHLERRATE_PCT
                                    and row["tps_pro_anfrage_mean"] >= ZIEL_MIN_TPS_PRO_ANFRAGE)
        rows.append(row)
    summary = pd.DataFrame(rows)

    # ---- Entscheidung ----
    ge = summary[summary["gleichzeitige_anfragen"] >= args.expected_users]
    if ge.empty:
        verdict = f"nicht beurteilbar: keine Stufe >= {args.expected_users} getestet"
    else:
        s = ge.iloc[0]
        verdict = (f"Stufe {int(s['gleichzeitige_anfragen'])} (>= erwartete {args.expected_users} Nutzer): "
                   + ("Zielwerte ERFÜLLT" if s["ziel_erfuellt"] else "Zielwerte NICHT erfüllt"))
    max_ok = summary.loc[summary["ziel_erfuellt"], "gleichzeitige_anfragen"].max() if summary["ziel_erfuellt"].any() else 0
    summary["erwartete_nutzer_annahme"] = args.expected_users
    summary["bewertung_erwartete_nutzer"] = verdict
    summary["max_stufe_innerhalb_ziel"] = max_ok
    eu.write_csv(summary.round(3), out / f"results_summary{suffix}.csv")

    # ---- Diagramme ----
    x = summary["gleichzeitige_anfragen"].tolist()
    ex.plot_lines(x, {"Median": summary["latenz_median_s"], "P95": summary["latenz_p95_s"]},
                  f"TS-07: Antwortzeit bei steigender Nutzerzahl ({args.engine})", "Gleichzeitige Anfragen",
                  "Latenz bis Antwortende (s)", out / f"diagramm_latenz{suffix}.png",
                  hlines={f"Zielwert P95 ({ZIEL_P95_LATENZ_S:.0f} s)": ZIEL_P95_LATENZ_S})
    ex.plot_lines(x, {"Gesamtdurchsatz (Token/s)": summary["durchsatz_token_s"],
                      "Token/s je Anfrage (Mittel)": summary["tps_pro_anfrage_mean"]},
                  f"TS-07: Durchsatz bei steigender Nutzerzahl ({args.engine})", "Gleichzeitige Anfragen",
                  "Token pro Sekunde", out / f"diagramm_durchsatz{suffix}.png",
                  hlines={"Mindest-Token/s je Anfrage": ZIEL_MIN_TPS_PRO_ANFRAGE})
    ex.plot_lines(x, {"Median": summary["ttft_median_s"], "P95": summary["ttft_p95_s"]},
                  f"TS-07: Wartezeit bis zum ersten Token ({args.engine})", "Gleichzeitige Anfragen",
                  "TTFT (s)", out / f"diagramm_ttft{suffix}.png")
    eu.plot_single_bars([str(v) for v in x], summary["ausfallrate_gesamt_pct"],
                        "TS-07: Timeout- und Fehlerrate", "Anteil fehlgeschlagener Anfragen (%)",
                        out / f"diagramm_fehlerrate{suffix}.png", ylim=(0, 105), fmt="{:.0f} %")
    eu.plot_single_bars([str(v) for v in x], summary["ram_peak_gb"], "TS-07: RAM-Spitzenwert der Engine",
                        "RAM-Peak (GB)", out / f"diagramm_ram{suffix}.png", fmt="{:.1f}")
    eu.plot_box({f"{n}": g.loc[g["status"] == "ok", "latenz_ms"].div(1000).tolist() for n, g in df.groupby("stufe_nutzer")},
                "Latenz (s)", "TS-07: Streuung der Antwortzeiten je Stufe", out / f"diagramm_latenz_boxplot{suffix}.png")

    pd.set_option("display.width", 220)
    print("\n=== Ergebnis TS-07 ===")
    show = ["gleichzeitige_anfragen", "n_ok", "ausfallrate_gesamt_pct", "latenz_median_s", "latenz_p95_s",
            "durchsatz_token_s", "tps_pro_anfrage_mean", "ram_peak_gb", "ziel_erfuellt"]
    print(summary[show].round(2).to_string(index=False))
    print(f"\nBEWERTUNG: {verdict}")
    print(f"Höchste Stufe innerhalb der Zielwerte: {max_ok if max_ok else 'keine'}")
    print(f"Dateien in: {out}")


if __name__ == "__main__":
    main()
