#!/usr/bin/env python3
"""
evaluate_frameworks.py - Auswertung des TS-12-Framework-Vergleichs (LangGraph, Microsoft Agent Framework, CrewAI, Haystack)

Liest results_<framework>.csv der vier Test-Skripte, zählt die Codezeilen (LOC) und erzeugt:
  results_summary.csv / results_summary.tex (booktabs)   Vergleichstabelle
  diagramm_task_success.png, diagramm_latenz.png, diagramm_radar.png   (300 dpi, deutsch)

LOC = Zeilen mit Code im Framework-Skript, ohne Leerzeilen, Kommentare und Docstrings. Der gemeinsame Messrahmen
fw_common.py (Prompts, Token-Zähler, CSV) wird NICHT mitgezählt - er ist für alle Frameworks identisch.
Radar-Achsen (0-1, außen = besser): Machbarkeit = Task-Success-Rate; Latenz, Token, Code-Komplexität = bester Wert
dividiert durch den eigenen Wert (der beste Kandidat erhält 1,0). Läufe ohne Messwert (z. B. Fehler) erhalten 0.

Beispiel:  python evaluate_frameworks.py
"""
import argparse
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import eval_utils as eu  # noqa: E402

# Anzeigename -> (CSV-Kennung, Skriptdatei)
FRAMEWORKS = {"LangGraph": ("langgraph", "test_langgraph.py"),
              "Microsoft Agent Framework": ("microsoft_agent", "test_microsoft_agent.py"),
              "CrewAI": ("crewai", "test_crewai.py"),
              "Haystack": ("haystack", "test_haystack.py")}


def count_loc(path: Path) -> float:
    """Codezeilen ohne Leerzeilen, Kommentarzeilen und Docstrings."""
    if not path.exists():
        return np.nan
    src = path.read_text(encoding="utf-8")
    doc_lines = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return float(sum(1 for i, line in enumerate(src.splitlines(), 1)
                     if line.strip() and not line.strip().startswith("#") and i not in doc_lines))


def tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def de(x, nd=1) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="Ordner mit den results_*.csv (Standard: Skriptordner)")
    ap.add_argument("--scripts-dir", default=None, help="Ordner mit den test_*.py für die LOC-Zählung")
    ap.add_argument("--out", default=None, help="Ausgabeordner (Standard: Skriptordner)")
    args = ap.parse_args()
    src_dir = Path(args.dir) if args.dir else HERE
    scripts_dir = Path(args.scripts_dir) if args.scripts_dir else HERE
    out = Path(args.out) if args.out else HERE
    out.mkdir(parents=True, exist_ok=True)

    runs, rows = [], []
    for name, (slug, script) in FRAMEWORKS.items():
        path = src_dir / f"results_{slug}.csv"
        loc = count_loc(scripts_dir / script)
        if not path.exists():
            print(f"WARNUNG: {path.name} fehlt - {name} wird mit Erfolg 0 gewertet.")
            rows.append({"framework": name, "n_laeufe": 0, "task_success_pct": 0.0, "latenz_s_mean": np.nan, "latenz_s_std": np.nan,
                         "latenz_s_median": np.nan, "setup_s_mean": np.nan, "tokens_mean": np.nan, "llm_anfragen_mean": np.nan,
                         "korrekturrunden_mean": np.nan, "fehlgeschlagene_laeufe": np.nan, "loc": loc})
            continue
        df = eu.read_csv(path)
        df["framework"] = name
        runs.append(df)
        ok = df[df["fehler"].isna() | (df["fehler"].astype(str).str.strip() == "")]
        rows.append({"framework": name, "n_laeufe": len(df), "task_success_pct": 100 * df["task_success"].mean(),
                     "latenz_s_mean": ok["latenz_s"].mean(), "latenz_s_std": ok["latenz_s"].std(), "latenz_s_median": ok["latenz_s"].median(),
                     "setup_s_mean": ok["setup_s"].mean(), "tokens_mean": ok["tokens_total"].mean(),
                     "llm_anfragen_mean": ok["llm_anfragen"].mean(), "korrekturrunden_mean": ok["korrekturrunden"].mean(),
                     "fehlgeschlagene_laeufe": len(df) - len(ok), "loc": loc})
    summary = pd.DataFrame(rows)
    allruns = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()

    # ---- Radar-Kennzahlen (0-1, außen = besser) ----
    def rel_best(col):
        v = summary[col].where(summary[col] > 0)
        return (v.min() / v).fillna(0.0)
    summary["score_machbarkeit"] = summary["task_success_pct"] / 100
    summary["score_latenz"] = rel_best("latenz_s_mean") * (summary["task_success_pct"] > 0)
    summary["score_token"] = rel_best("tokens_mean") * (summary["task_success_pct"] > 0)
    summary["score_code"] = rel_best("loc")
    eu.write_csv(summary.round(3), out / "results_summary.csv")

    # ---- LaTeX-Tabelle (booktabs; Präambel: \usepackage{booktabs}) ----
    head = ["Framework", "Task Success (\\%)", "Latenz Ø $\\pm$ SD (s)", "Token Ø", "LLM-Anfragen Ø", "Setup (s)", "LOC"]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Vergleich der Code-First-Frameworks im Mini-Workflow (Primary, Critic, Self-Correction)}",
             "\\label{tab:ts12-frameworks}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & ".join(f"\\textbf{{{h}}}" for h in head) + " \\\\", "\\midrule"]
    for _, r in summary.iterrows():
        lat = "n/a" if pd.isna(r["latenz_s_mean"]) else f"{de(r['latenz_s_mean'])} $\\pm$ {de(r['latenz_s_std'])}"
        lines.append(" & ".join([tex_escape(r["framework"]), de(r["task_success_pct"], 0), lat, de(r["tokens_mean"], 0),
                                 de(r["llm_anfragen_mean"], 1), de(r["setup_s_mean"], 1), de(r["loc"], 0)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "results_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Diagramme (seaborn, 300 dpi, deutsch) ----
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    pal = dict(zip(summary["framework"], sns.color_palette("colorblind", len(summary))))
    order = summary["framework"].tolist()

    def bar(data, y, title, ylabel, path, errorbar=None, fmt="{:.1f}", ylim=None):
        fig, ax = plt.subplots(figsize=(9, 5.8))
        sns.barplot(data=data, x="framework", y=y, hue="framework", order=order, hue_order=order, palette=pal,
                    errorbar=errorbar, capsize=0.12, dodge=False, legend=True, ax=ax)
        for c in ax.containers:
            if hasattr(c, "datavalues"):
                texts = ["n/a" if np.isnan(v) else fmt.format(v).replace(".", ",") for v in c.datavalues]
                if errorbar:       # Beschriftung in den Balken, damit sie nicht mit den Fehlerbalken kollidiert
                    ax.bar_label(c, labels=texts, label_type="center", color="white", fontweight="bold", fontsize=12)
                else:
                    ax.bar_label(c, labels=texts, padding=3, fontsize=12)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace(" ", "\n", 1) if len(o) > 12 else o for o in order])
        ax.set_ylim(*ylim) if ylim else ax.set_ylim(bottom=0)
        if ax.get_legend():
            ax.legend_.set_title("Framework")
            sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    bar(summary, "task_success_pct", "Machbarkeit: Wird der Fehler (Südtirol gehört zu Italien) korrigiert?",
        "Task Success (% der Läufe)", out / "diagramm_task_success.png", fmt="{:.0f} %", ylim=(0, 115))
    if not allruns.empty:
        okruns = allruns[allruns["fehler"].isna() | (allruns["fehler"].astype(str).str.strip() == "")]
        lat = okruns[["framework", "latenz_s"]]
        missing = pd.DataFrame({"framework": [f for f in order if f not in set(lat["framework"])], "latenz_s": np.nan})
        bar(pd.concat([lat, missing], ignore_index=True), "latenz_s",
            "Latenz des Workflows (Mittelwert, Fehlerbalken = Standardabweichung)", "Latenz (s, niedriger = besser)",
            out / "diagramm_latenz.png", errorbar="sd", fmt="{:.1f} s")
    labels = ["Machbarkeit", "Latenz", "Token-\nVerbrauch", "Code-Komplexität\n(LOC)"]
    radar = {r["framework"]: [r["score_machbarkeit"], r["score_latenz"], r["score_token"], r["score_code"]] for _, r in summary.iterrows()}
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for name, vals in radar.items():
        v = [0.0 if pd.isna(x) else float(x) for x in vals]
        ax.plot(ang + ang[:1], v + v[:1], linewidth=2.2, label=name, color=pal[name])
        ax.fill(ang + ang[:1], v + v[:1], alpha=0.10, color=pal[name])
    ax.set_xticks(ang)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=22)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0,25", "0,5", "0,75", "1,0"], fontsize=9)
    ax.set_title("Framework-Vergleich (außen = besser, 1,0 = bester Wert)", fontweight="bold", pad=34)
    ax.legend(title="Framework", loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.savefig(out / "diagramm_radar.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.set_option("display.width", 220)
    print("\n=== Ergebnis TS-12 Frameworks ===")
    print(summary[["framework", "n_laeufe", "task_success_pct", "latenz_s_mean", "tokens_mean", "llm_anfragen_mean", "loc"]]
          .round(1).to_string(index=False))
    print(f"\nDateien in: {out}")


if __name__ == "__main__":
    main()