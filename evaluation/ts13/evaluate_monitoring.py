#!/usr/bin/env python3
"""
evaluate_monitoring.py - TS-13: qualitativer Vergleich Prometheus/Grafana vs. Streamlit (Monitoring von KAISER)

Erzeugt die Kriterienmatrix als
  results_monitoring.csv   (Excel-DE: Semikolon, UTF-8 mit BOM)
  results_monitoring.tex   (LaTeX, booktabs; Präambel: \\usepackage{booktabs} und \\usepackage{array})
  results_monitoring.md    (zum Gegenlesen)
Kriterien laut Aufgabe: Echtzeit, Offline-Betrieb, Setup-Aufwand, Reload-Verhalten; ergänzend (markiert): Historie,
Alarmierung, Lizenz. Der Konfigurationsumfang (Dateien/Zeilen) wird aus dem Ordner automatisch gezählt.

WICHTIG: Die Bewertungstexte sind eine begründete Einschätzung zu den in docker-compose.yml/requirements gepinnten
Versionen, KEINE Messung. Spalte "Praxistest" nennt, wie jede Aussage im Versuch zu belegen ist - bitte vor Abgabe
durchführen und Ergebnis/Datum ergänzen (Lizenzen und Defaults ändern sich mit Versionen).

Beispiel:  python evaluate_monitoring.py
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import pandas as pd  # noqa: E402

import eval_utils as eu  # noqa: E402

STACK_FILES = ["docker-compose.yml", "prometheus.yml", "grafana/provisioning/datasources/prometheus.yml",
               "grafana/provisioning/dashboards/dashboards.yml", "grafana/dashboards/kaiser.json"]
STREAMLIT_FILES = ["streamlit_app.py", ".streamlit/config.toml"]


def count_lines(base: Path, files):
    """(Anzahl vorhandener Dateien, Zeilen ohne Leerzeilen/Kommentarzeilen)."""
    n_files = n_lines = 0
    for f in files:
        p = base / f
        if p.exists():
            n_files += 1
            n_lines += sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                           if line.strip() and not line.strip().startswith("#"))
    return n_files, n_lines


def build_matrix(base: Path) -> pd.DataFrame:
    sf, sl = count_lines(base, STACK_FILES)
    tf, tl = count_lines(base, STREAMLIT_FILES)
    rows = [
        ("Echtzeit-Fähigkeit", "Kern",
         "Pull-Modell: Prometheus liest den Exporter im Scrape-Intervall (hier 5 s), Grafana aktualisiert Panels im eingestellten "
         "Refresh (hier 5 s). Aktualität = Scrape-Intervall + Refresh, also Sekundenbereich; kein Push, keine offene Verbindung.",
         "Aktualisierung als Fragment (st.fragment mit run_every, hier 5 s) oder bei Interaktion. Daten kommen direkt aus Python "
         "(kein Scraping), Sekundenbereich möglich; jede Browser-Sitzung rechnet und speichert selbst.",
         "Zeit zwischen Änderung im Exporter und sichtbarer Änderung im Dashboard stoppen (mehrfach, Median)."),
        ("Offline-Betrieb", "Kern",
         "Ja. Images (Prometheus, Grafana, Python) einmal online laden und mit docker save/load offline übertragen; Telemetrie, "
         "Update-Prüfung und News-Feed sind in der docker-compose.yml abgeschaltet. Zusatz-Plugins nur lokal installieren.",
         "Ja. Pakete über lokalen Wheelhouse/Mirror installierbar; Telemetrie in .streamlit/config.toml abgeschaltet "
         "(gatherUsageStats = false). Komponenten aus Drittquellen (CDN) separat prüfen.",
         "Test bei getrennter Netzwerkverbindung: Start, Dashboard-Aufruf, Neustart; ausgehende Verbindungen mitschneiden."),
        ("Setup-Aufwand", "Kern",
         f"Docker plus 3 Dienste (Exporter, Prometheus, Grafana), YAML-Konfiguration, Provisioning von Datenquelle und Dashboard, "
         f"PromQL-Kenntnisse. Konfigurationsumfang: {sf} Dateien, {sl} Zeilen (ohne Exporter).",
         f"pip install streamlit, eine Python-Datei, kein Container. Umfang: {tf} Dateien, {tl} Zeilen. "
         "Speicherung der Zeitreihen und Alarmierung müssten selbst gebaut werden.",
         "Zeit vom leeren Rechner bis zum laufenden Dashboard stoppen (Protokoll der Schritte)."),
        ("Reload-Verhalten", "Kern",
         "Prometheus-Konfiguration ohne Neustart über POST /-/reload (--web.enable-lifecycle) oder SIGHUP; Dashboards aus dem "
         "Provisioning werden alle 10 s neu eingelesen. Browser-Reload verliert nichts: Zeitreihen liegen in der Prometheus-TSDB (Volume).",
         "Browser-Reload startet eine neue Sitzung: Skript läuft komplett neu, st.session_state und Historie sind leer (außer "
         "extern gespeichert). Code-Änderungen: runOnSave bzw. Rerun; neue Abhängigkeiten erfordern Neustart.",
         "Browser-Reload, Container-/Prozess-Neustart und Konfigurationsänderung testen; prüfen, ob Verlauf erhalten bleibt."),
        ("Historie / Speicherung (ergänzend)", "Ergänzend",
         "Zeitreihen-Datenbank mit Aufbewahrung (hier 15 Tage), Abfrage beliebiger Zeiträume per PromQL.",
         "Keine eigene Persistenz; Historie nur im Arbeitsspeicher der Sitzung, sofern nicht selbst Datenbank/Datei angebunden wird.",
         "Dashboard nach Neustart öffnen und 15 Minuten zurückblättern."),
        ("Alarmierung (ergänzend)", "Ergänzend",
         "Alertmanager bzw. Grafana Alerting vorhanden (im Test-Compose nicht aktiviert).",
         "Nicht enthalten; müsste selbst programmiert werden.",
         "Regel für hohe Latenz anlegen und Auslösung prüfen (Prometheus/Grafana)."),
        ("Lizenz (ergänzend)", "Ergänzend",
         "Prometheus: Apache-2.0; Grafana (OSS): AGPL-3.0 - bei Weitergabe/Netzwerkbereitstellung geänderter Versionen beachten.",
         "Streamlit: Apache-2.0.",
         "Lizenztexte der gepinnten Versionen prüfen und Fundstelle zitieren."),
    ]
    return pd.DataFrame(rows, columns=["Kriterium", "Gewichtung", "Prometheus/Grafana", "Streamlit", "Praxistest"])


def tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def to_latex(df: pd.DataFrame) -> str:
    col = lambda w: f">{{\\raggedright\\arraybackslash}}p{{{w}cm}}"        # noqa: E731
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\caption{Qualitativer Vergleich der Monitoring-Lösungen (TS-13)}", "\\label{tab:ts13-monitoring}",
             f"\\begin{{tabular}}{{{col(2.4)}{col(5.4)}{col(5.4)}}}", "\\toprule",
             "\\textbf{Kriterium} & \\textbf{Prometheus/Grafana} & \\textbf{Streamlit} \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{tex_escape(r['Kriterium'])} & {tex_escape(r['Prometheus/Grafana'])} & {tex_escape(r['Streamlit'])} \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def to_markdown(df: pd.DataFrame) -> str:
    out = ["# TS-13: Monitoring-Vergleich - Kriterienmatrix", "",
           "| Kriterium | Prometheus/Grafana | Streamlit | Praxistest |", "|---|---|---|---|"]
    out += [f"| {r['Kriterium']} | {r['Prometheus/Grafana']} | {r['Streamlit']} | {r['Praxistest']} |" for _, r in df.iterrows()]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="Ordner mit den TS-13-Dateien (Standard: Skriptordner)")
    ap.add_argument("--out", default=None, help="Ausgabeordner (Standard: Skriptordner)")
    args = ap.parse_args()
    base = Path(args.dir) if args.dir else HERE
    out = Path(args.out) if args.out else HERE
    out.mkdir(parents=True, exist_ok=True)
    df = build_matrix(base)
    eu.write_csv(df, out / "results_monitoring.csv")
    (out / "results_monitoring.tex").write_text(to_latex(df), encoding="utf-8")
    (out / "results_monitoring.md").write_text(to_markdown(df), encoding="utf-8")
    print(df[["Kriterium", "Gewichtung"]].to_string(index=False))
    print(f"\nGespeichert in {out}: results_monitoring.csv | .tex | .md")


if __name__ == "__main__":
    main()
