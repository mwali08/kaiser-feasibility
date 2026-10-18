#!/usr/bin/env python3
"""
TS-09: Strukturierte Ausgabe - JSON-Constraint (Ollama-Constrained-Decoding) vs. Freitext

Beide Varianten erhalten DENSELBEN Prompt (Aufforderung, als JSON zu antworten). Einziger Unterschied:
die constrained Variante sendet response_format={"type": "json_object"} (Ollama setzt das intern auf format="json").
Hinweis: llm_client.py kennt keinen format-Parameter; die Variante wird deshalb über register_variant() mit
extra_body registriert (OpenAI-kompatibler Ollama-Endpunkt /v1).

Metriken:
  * strikt gültig (%)   - gesamte Ausgabe ist parsebares JSON-Objekt
  * schema-gültig (%)   - zusätzlich: Schlüssel/Typen korrekt (antwort, konfidenz 0-1, quelle kontext|wissen)
  * tolerant gültig (%) - JSON nach Entfernen von Markdown-Zäunen/Begleittext extrahierbar (Aufwand für Parser)
  * zusätzliche Latenz  - gepaarte Differenz (constrained minus Freitext) je Frage/Wiederholung, mit Bootstrap-95-%-KI
  * vermutlich abgeschnitten - Ausgabe erreicht max_tokens (typischer Fehlerfall bei Constraints)

Beispiel:  python ts09_strukturierte_ausgabe.py --repetitions 3
Trockenlauf: python ts09_strukturierte_ausgabe.py --dry-run --out .\\dryrun
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
MAX_TOKENS = 250
VARIANTEN = {"Freitext (ohne Constraint)": "ts09_frei", "JSON-Constraint (format=json)": "ts09_json"}
PROMPT = ("Beantworte die Frage und gib das Ergebnis AUSSCHLIESSLICH als JSON-Objekt mit genau diesen Schlüsseln aus:\n"
          '{"antwort": "<kurze Antwort auf Deutsch>", "konfidenz": <Zahl zwischen 0 und 1>, '
          '"quelle": "kontext" oder "wissen"}\n\n{kontext}Frage: {frage}')
# ============================================================================ #


def schema_ok(obj) -> bool:
    return (isinstance(obj, dict) and isinstance(obj.get("antwort"), str) and obj["antwort"].strip() != ""
            and isinstance(obj.get("konfidenz"), (int, float)) and not isinstance(obj.get("konfidenz"), bool)
            and 0 <= obj["konfidenz"] <= 1 and obj.get("quelle") in ("kontext", "wissen"))


def parse_strict(text: str):
    try:
        obj = json.loads(text.strip())
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_lenient(text: str):
    t = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def main():
    ex.fix_console()
    args = ex.common_parser(__doc__).parse_args()
    out = Path(args.out) if args.out else HERE
    eu.setup_logging(out)
    max_tokens = args.max_tokens or MAX_TOKENS
    questions = eu.load_questions(Path(args.testset) if args.testset else HERE / "testset.json", args.limit)
    ok_json = '{"antwort": "Wien", "konfidenz": 0.9, "quelle": "wissen"}'
    if args.dry_run:
        ex.setup_dry_run(args.judge_model)
        ex.register_model("ts09_frei", {"provider": "mock", "family": "qwen", "mock_delay_s": 0.02,
                                        "mock_response": "```json\n" + ok_json + "\n```"})
        ex.register_model("ts09_json", {"provider": "mock", "family": "qwen", "mock_delay_s": 0.03, "mock_response": ok_json})
    else:
        pd_.register_variant("ts09_frei", args.model)
        pd_.register_variant("ts09_json", args.model, extra_body={"response_format": {"type": "json_object"}})
    print(f"TS-09 | {len(questions)} Fragen | Wiederholungen: {args.repetitions} | max_tokens: {max_tokens}")
    for key in VARIANTEN.values():                       # Warm-up (Modell laden, nicht messen)
        query_model(key, "Antworte mit einem Wort: Hallo.", stream=False, max_tokens=8, retries=0)

    rows, names = [], list(VARIANTEN)
    todo = [(q, rep) for rep in range(1, args.repetitions + 1) for q in questions]
    for idx, (q, rep) in enumerate(tqdm(todo, desc="TS-09", unit="Frage")):
        prompt = PROMPT.replace("{kontext}", f"Kontext: {q['context']}\n" if q["context"] else "").replace("{frage}", q["question"])
        for name in (names if idx % 2 == 0 else names[::-1]):      # Reihenfolge abwechseln (kein Reihenfolge-Bias)
            rec = {"variante": name, "question_id": q["id"], "repetition": rep}
            try:
                res = query_model(VARIANTEN[name], prompt, stream=not args.no_stream, temperature=0.0,
                                  max_tokens=max_tokens, retries=0)
            except Exception as exc:  # noqa: BLE001
                eu.log.exception("Fehler | %s | %s", name, q["id"])
                res = {"response_text": "", "error": f"{type(exc).__name__}: {exc}", "total_latency_ms": np.nan,
                       "ttft_ms": np.nan, "token_count": 0, "tps": np.nan}
            text = res["response_text"] or ""
            strict, lenient = parse_strict(text), parse_lenient(text)
            rec.update({"rohausgabe": text, "latenz_ms": res["total_latency_ms"], "ttft_ms": res["ttft_ms"],
                        "token_count": res["token_count"], "tps": res["tps"], "error": res["error"],
                        "json_strikt": strict is not None, "schema_gueltig": schema_ok(strict),
                        "json_tolerant": lenient is not None,
                        "vermutlich_abgeschnitten": res["token_count"] >= max_tokens * 0.98})
            if strict is None:
                eu.log.warning("Kein striktes JSON | %s | %s | Rep %d | %s", name, q["id"], rep, text[:80].replace("\n", " "))
            rows.append(rec)
    df = pd.DataFrame(rows)
    eu.write_csv(df, out / "results_details.csv")

    # ---- Zusammenfassung ----
    summ = []
    for name, g in df.groupby("variante", sort=False):
        row = {"variante": name, "n_anfragen": len(g)}
        for col in ("json_strikt", "schema_gueltig", "json_tolerant"):
            row[f"{col}_pct"] = 100 * g[col].mean()
        row["json_strikt_ci95_unten"], row["json_strikt_ci95_oben"] = pd_.wilson_ci(int(g["json_strikt"].sum()), len(g))
        row.update({"abgeschnitten_pct": 100 * g["vermutlich_abgeschnitten"].mean(),
                    "latenz_median_ms": g["latenz_ms"].median(), "latenz_mean_ms": g["latenz_ms"].mean(),
                    "latenz_p95_ms": pd_.pct(g["latenz_ms"], 95), "token_count_mean": g["token_count"].mean(),
                    "tps_mean": g["tps"].mean()})
        summ.append(row)
    summary = pd.DataFrame(summ).set_index("variante").loc[names].reset_index()
    wide = df.pivot_table(index=["question_id", "repetition"], columns="variante", values="latenz_ms")
    diff = (wide[names[1]] - wide[names[0]]).dropna()
    mean, lo, hi = ex.bootstrap_ci(diff.values)
    summary["zusaetzliche_latenz_ms_mean"] = [np.nan, mean]
    summary["zusaetzliche_latenz_ms_median"] = [np.nan, diff.median() if len(diff) else np.nan]
    summary["zusaetzliche_latenz_ci95_unten"], summary["zusaetzliche_latenz_ci95_oben"] = [np.nan, lo], [np.nan, hi]
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # ---- Diagramme ----
    eu.plot_grouped_bars(summary, "variante", {"Striktes JSON": "json_strikt_pct", "Schema korrekt": "schema_gueltig_pct",
                                               "JSON nach Aufbereitung": "json_tolerant_pct"},
                         "TS-09: Wie oft liefert das Modell gültiges JSON?", "Anteil der Anfragen (%)",
                         out / "diagramm_gueltigkeit.png", ylim=(0, 118))
    eu.plot_box({v: g["latenz_ms"].tolist() for v, g in df.groupby("variante", sort=False)},
                "Gesamtlatenz (ms)", "TS-09: Antwortzeit je Variante", out / "diagramm_latenz_boxplot.png")
    eu.plot_single_bars(["JSON-Constraint\ngegenüber Freitext"], [mean], "TS-09: Zusätzliche Latenz durch den JSON-Constraint",
                        "Differenz der Gesamtlatenz (ms; > 0 = langsamer)", out / "diagramm_zusatzlatenz.png", fmt="{:+.0f} ms")

    pd.set_option("display.width", 200)
    print("\n=== Ergebnis TS-09 ===")
    print(summary[["variante", "json_strikt_pct", "schema_gueltig_pct", "json_tolerant_pct", "latenz_median_ms",
                   "abgeschnitten_pct"]].round(1).to_string(index=False))
    print(f"\nZusätzliche Latenz (constrained - frei): Ø {mean:+.0f} ms, 95-%-KI [{lo:+.0f}; {hi:+.0f}] ms")
    print(f"Dateien in: {out}")


if __name__ == "__main__":
    main()