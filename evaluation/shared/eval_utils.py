"""
eval_utils.py – Geteilte Helfer für TS-02, TS-03, TS-04:
CSV (UTF-8-BOM, Excel-tauglich), Logging, Testset-Loader, VRAM-Messung,
LLM-as-a-Judge, Evaluationsschleife, Nutzwertanalyse/KO-Prüfung, Diagramme (300 dpi, Deutsch).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from llm_client import get_model_spec, query_model  # noqa: E402

log = logging.getLogger("eval")
NAN = float("nan")

# Excel (deutsch): Semikolon + Dezimalkomma + UTF-8 mit BOM
CSV_KW = dict(sep=";", decimal=",", encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# Dateien, Logging, CLI
# --------------------------------------------------------------------------- #
def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, **CSV_KW)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_KW)


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(fh)
    root.addHandler(sh)


def base_parser(description: str, default_models: list[str]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--models", nargs="+", default=default_models, help="Modell-Kurznamen aus models.json")
    p.add_argument("--repetitions", type=int, default=3, help="Wiederholungen je Frage/Modell")
    p.add_argument("--testset", default=None, help="Pfad zum Testset-JSON (Standard: testset.json im Skriptordner)")
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "judge"), help="Modell für LLM-as-a-Judge")
    p.add_argument("--no-judge", action="store_true", help="Judge überspringen (Bewertung später manuell)")
    p.add_argument("--no-stream", action="store_true", help="Ohne Streaming (TTFT = Gesamtlatenz)")
    p.add_argument("--limit", type=int, default=None, help="Nur die ersten N Fragen (zum Testen)")
    p.add_argument("--resume", action="store_true", help="Bereits gespeicherte Antworten nicht neu erzeugen")
    p.add_argument("--out", default=None, help="Ausgabeordner (Standard: Skriptordner)")
    return p


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    """Testset-Struktur: Liste oder {"questions": [...]} mit Feldern
    id, question, [category], [is_fact], [reference_answer], [context]."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["questions"] if isinstance(data, dict) else data
    out = []
    for q in items:
        if "id" not in q or "question" not in q:
            raise ValueError(f"Testset-Eintrag ohne 'id'/'question': {q}")
        out.append({"id": str(q["id"]), "question": q["question"], "category": q.get("category", "allgemein"),
                    "is_fact": bool(q.get("is_fact", False)), "reference_answer": q.get("reference_answer"),
                    "context": q.get("context")})
    return out[:limit] if limit else out


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["valid"] = df["valid"].fillna(False).astype(bool)
    df["is_fact"] = df["is_fact"].fillna(False).astype(bool)
    df["korrekt"] = pd.to_numeric(df["korrekt"].map({True: 1.0, False: 0.0}), errors="coerce")
    return df


def model_label(model: str) -> str:
    try:
        return get_model_spec(model).get("display_name", model)
    except KeyError:
        return model


def model_family(model: str) -> str:
    try:
        return get_model_spec(model).get("family", "unbekannt")
    except KeyError:
        return "unbekannt"


# --------------------------------------------------------------------------- #
# VRAM-Messung (NVIDIA, pynvml)
# --------------------------------------------------------------------------- #
class VRAMMonitor:
    """Misst den VRAM-Spitzenwert (GB) während eines Blocks relativ zum Ausgangswert.
    Hinweis: Vor jedem Modellblock sollten andere Modelle entladen sein (z. B. `ollama stop <modell>`),
    sonst wird der Wert verfälscht. Alternative: 'vram_gb' manuell in models.json eintragen."""

    _warned = False

    def __init__(self, interval: float = 0.2):
        self.interval, self.peak, self.baseline, self.available = interval, 0.0, 0.0, False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(int(os.getenv("GPU_INDEX", "0")))
            self.available = True
        except Exception:  # noqa: BLE001
            if not VRAMMonitor._warned:
                VRAMMonitor._warned = True
                log.warning("VRAM-Messung nicht verfügbar (pynvml/NVIDIA-GPU fehlt) – vram_gb bleibt leer "
                            "(Alternative: 'vram_gb' manuell in models.json).")

    def _used_gb(self) -> float:
        return self._nv.nvmlDeviceGetMemoryInfo(self._h).used / 1024 ** 3

    def __enter__(self):
        if self.available:
            self.baseline = self.peak = self._used_gb()
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self._used_gb())
            except Exception:  # noqa: BLE001
                break

    def __exit__(self, *exc):
        if self.available:
            self._stop.set()
            self._t.join(timeout=2)
            self.peak = max(self.peak, self._used_gb())
        return False

    @property
    def delta_gb(self) -> float:
        return round(max(self.peak - self.baseline, 0.0), 2) if self.available else NAN


# --------------------------------------------------------------------------- #
# LLM-as-a-Judge
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = ("Du bist ein strenger, objektiver Prüfer für Sprachmodell-Antworten. "
                "Antworte AUSSCHLIESSLICH mit einem einzelnen JSON-Objekt, ohne Text davor oder danach.")


def parse_json_from_text(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _clip_int(v, lo=1, hi=5):
    try:
        i = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return i if lo <= i <= hi else None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "ja", "wahr"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "nein", "falsch"):
        return False
    return None


def judge_json(judge_model: str, prompt: str) -> tuple[dict | None, str | None]:
    res = query_model(judge_model, prompt, stream=False, temperature=0.0, system_prompt=JUDGE_SYSTEM, max_tokens=800)
    if res["error"]:
        return None, res["error"]
    data = parse_json_from_text(res["response_text"])
    if data is None:
        return None, "Judge-Antwort nicht als JSON lesbar: " + res["response_text"][:200]
    return data, None


def judge_answer(judge_model: str, question: str, answer: str, reference: str | None, context: str | None) -> dict:
    prompt = f"""Bewerte die folgende Antwort eines Sprachmodells.

FRAGE:
{question}
"""
    if context:
        prompt += f"\nKONTEXT (einzige zulässige Wissensquelle):\n{context}\n"
    if reference:
        prompt += f"\nREFERENZANTWORT (manuell geprüft):\n{reference}\n"
    prompt += f"""
ZU BEWERTENDE ANTWORT:
{answer}

Skalen (ganze Zahlen 1-5):
- antwortqualitaet: 1 = falsch/unbrauchbar, 3 = teilweise richtig oder unvollständig, 5 = vollständig, korrekt, klar, hilfreich.
- deutschqualitaet: 1 = kein oder stark fehlerhaftes Deutsch, 3 = verständlich mit spürbaren Fehlern, 5 = fehlerfreies, natürliches Deutsch.
- faithfulness: {"1 = widerspricht dem Kontext oder erfindet Inhalte, 3 = teils nicht durch den Kontext gedeckt, 5 = jede Aussage ist durch den Kontext gedeckt." if context else "kein Kontext vorhanden -> null"}
- korrekt: {"true, wenn die Kernaussage mit der Referenzantwort übereinstimmt, sonst false." if reference else "keine Referenz vorhanden -> null"}

Antworte nur mit JSON:
{{"antwortqualitaet": <1-5>, "deutschqualitaet": <1-5>, "faithfulness": <1-5 oder null>, "korrekt": <true/false/null>, "begruendung": "<max. 2 Sätze>"}}"""
    data, err = judge_json(judge_model, prompt)
    empty = {"antwortqualitaet": None, "deutschqualitaet": None, "faithfulness": None, "korrekt": None,
             "judge_begruendung": None, "judge_error": err}
    if data is None:
        log.warning("Judge-Fehler: %s", err)
        return empty
    out = {"antwortqualitaet": _clip_int(data.get("antwortqualitaet")),
           "deutschqualitaet": _clip_int(data.get("deutschqualitaet")),
           "faithfulness": _clip_int(data.get("faithfulness")) if context else None,
           "korrekt": _to_bool(data.get("korrekt")) if reference else None,
           "judge_begruendung": str(data.get("begruendung", ""))[:500], "judge_error": None}
    if out["antwortqualitaet"] is None or out["deutschqualitaet"] is None:
        out["judge_error"] = "Ungültige Skalenwerte vom Judge"
        log.warning("Judge lieferte ungültige Werte: %s", data)
    return out


def judge_same_error(judge_model: str, question: str, reference: str, ans_a: str, ans_b: str) -> bool | None:
    prompt = f"""Zwei Sprachmodelle haben dieselbe Frage beantwortet; beide Antworten sind falsch.

FRAGE:
{question}

KORREKTE REFERENZANTWORT:
{reference}

ANTWORT A:
{ans_a}

ANTWORT B:
{ans_b}

Behaupten A und B inhaltlich dieselbe falsche Kernaussage (z. B. dieselbe falsche Zahl, derselbe falsche Name,
dieselbe falsche Behauptung)? Unterschiedliche falsche Antworten oder bloß gleiche Themen zählen als false.
Antworte nur mit JSON: {{"gleicher_fehler": <true/false>, "begruendung": "<1 Satz>"}}"""
    data, err = judge_json(judge_model, prompt)
    if data is None:
        log.warning("Judge-Fehler (gleicher Fehler): %s", err)
        return None
    return _to_bool(data.get("gleicher_fehler"))


# --------------------------------------------------------------------------- #
# Generische Evaluationsschleife (TS-02 / TS-03)
# --------------------------------------------------------------------------- #
def build_prompt(q: dict) -> str:
    if q.get("context"):
        return (f"Kontext:\n{q['context']}\n\nFrage: {q['question']}\n\n"
                "Beantworte die Frage ausschließlich auf Basis des obigen Kontexts.")
    return q["question"]


def evaluate_models(models, questions, repetitions, system_prompt, judge_model, use_judge, stream,
                    jsonl_path: Path, vram_path: Path, resume: bool) -> list[dict]:
    """Fragt jedes Modell (gleicher Prompt/Parameter), bewertet via Judge, speichert jede Antwort sofort."""
    if jsonl_path.exists() and not resume:
        jsonl_path.unlink()
    records = load_jsonl(jsonl_path)
    done = {(r["model"], r["question_id"], r["repetition"]) for r in records}
    vram_store = json.loads(vram_path.read_text(encoding="utf-8")) if vram_path.exists() else {}

    for model in models:
        spec = get_model_spec(model)
        todo = [(q, rep) for rep in range(1, repetitions + 1) for q in questions
                if (model, q["id"], rep) not in done]
        if not todo:
            continue
        with VRAMMonitor() as mon:
            for q, rep in tqdm(todo, desc=f"{model_label(model):<18}", unit="Anfrage"):
                prompt = build_prompt(q)
                rec = {"model": model, "model_name": model_label(model), "family": model_family(model),
                       "question_id": q["id"], "category": q["category"], "is_fact": q["is_fact"],
                       "repetition": rep, "prompt": prompt, "question": q["question"],
                       "reference_answer": q["reference_answer"], "context": q["context"]}
                try:
                    res = query_model(model, prompt, stream=stream, temperature=0.0, system_prompt=system_prompt)
                    rec.update({"response_text": res["response_text"], "ttft_ms": res["ttft_ms"],
                                "total_latency_ms": res["total_latency_ms"], "token_count": res["token_count"],
                                "tps": res["tps"], "error": res["error"]})
                    rec["valid"] = res["error"] is None and len(res["response_text"].strip()) >= 2
                    if not rec["valid"]:
                        log.warning("Ungültige Antwort | %s | %s | Rep %d | %s", model, q["id"], rep, res["error"])
                    elif use_judge:
                        rec.update(judge_answer(judge_model, q["question"], res["response_text"],
                                                q["reference_answer"], q["context"]))
                except Exception as exc:  # noqa: BLE001 – Lauf darf nie abbrechen
                    log.exception("Unerwarteter Fehler | %s | %s", model, q["id"])
                    rec.update({"valid": False, "error": f"{type(exc).__name__}: {exc}", "response_text": ""})
                for k in ("antwortqualitaet", "deutschqualitaet", "faithfulness", "korrekt", "judge_begruendung",
                          "judge_error"):
                    rec.setdefault(k, None)
                records.append(rec)
                append_jsonl(jsonl_path, rec)
        vram = spec.get("vram_gb") if spec.get("vram_gb") is not None else mon.delta_gb
        vram_store[model] = None if (isinstance(vram, float) and math.isnan(vram)) else vram
        vram_path.write_text(json.dumps(vram_store, indent=2), encoding="utf-8")
    return records


def load_vram(vram_path: Path) -> dict:
    d = json.loads(vram_path.read_text(encoding="utf-8")) if vram_path.exists() else {}
    return {k: (NAN if v is None else v) for k, v in d.items()}


def summarize_generation(df: pd.DataFrame, vram_map: dict) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model", sort=False):
        v = g[g["valid"]]
        facts = v[v["is_fact"] & v["korrekt"].notna()]
        row = {"model": model, "modell": model_label(model), "familie": model_family(model),
               "n_anfragen": len(g), "n_gueltig": len(v),
               "ungueltig_quote_pct": round(100 * (1 - len(v) / len(g)), 2) if len(g) else NAN}
        for col in ("antwortqualitaet", "deutschqualitaet", "faithfulness"):
            s = pd.to_numeric(v[col], errors="coerce")
            row[f"{col}_mean"], row[f"{col}_std"] = s.mean(), s.std()
        row["faktentreffer_pct"] = 100 * facts["korrekt"].mean() if len(facts) else NAN
        row["ttft_ms_mean"], row["ttft_ms_std"], row["ttft_ms_median"] = (
            v["ttft_ms"].mean(), v["ttft_ms"].std(), v["ttft_ms"].median())
        row["tps_mean"], row["tps_std"] = v["tps"].mean(), v["tps"].std()
        row["total_latency_ms_mean"], row["token_count_mean"] = v["total_latency_ms"].mean(), v["token_count"].mean()
        row["vram_gb"] = vram_map.get(model, NAN)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# KO-Kriterien und Nutzwertanalyse
# --------------------------------------------------------------------------- #
def check_ko(row: pd.Series, ko: dict) -> tuple[bool, str]:
    """ko = {spalte: {"min": x | "max": y, "label": "..."}}. Fehlende Werte -> 'nicht prüfbar' (kein Ausschluss)."""
    failed, unchecked = [], []
    for col, c in ko.items():
        val = row.get(col, NAN)
        if val is None or pd.isna(val):
            unchecked.append(c["label"])
            continue
        if ("min" in c and val < c["min"]) or ("max" in c and val > c["max"]):
            failed.append(f"{c['label']} (Ist: {val:.2f})")
    msg = "; ".join(failed)
    if unchecked:
        msg += (" | " if msg else "") + "nicht prüfbar: " + ", ".join(unchecked)
    return len(failed) == 0, msg


def nutzwert(row: pd.Series, criteria: dict) -> tuple[float, float, dict]:
    """Lineare Normierung je Kriterium auf 0-1 (worst -> 0, best -> 1, geklemmt), gewichtete Summe * 100.
    Feste Ankerwerte (statt Min-Max über die Kandidaten) -> reproduzierbar und auch bei 2 Modellen aussagekräftig.
    Rückgabe: (Nutzwert 0-100, Gewichtsabdeckung 0-1, {Kriterium: normierter Wert})."""
    total = wsum = 0.0
    parts = {}
    for col, c in criteria.items():
        val = row.get(col, NAN)
        if val is None or pd.isna(val):
            parts[col] = NAN
            continue
        s = min(1.0, max(0.0, (val - c["worst"]) / (c["best"] - c["worst"])))
        parts[col] = s
        total += c["weight"] * s
        wsum += c["weight"]
    full = sum(c["weight"] for c in criteria.values())
    return (100 * total / wsum if wsum else NAN), (wsum / full if full else NAN), parts


# --------------------------------------------------------------------------- #
# Diagramme (300 dpi, deutsche Beschriftung)
# --------------------------------------------------------------------------- #
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold"})


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_grouped_bars(df, label_col, value_cols: dict, title, ylabel, path, err_cols: dict | None = None, ylim=None):
    n, x = len(value_cols), np.arange(len(df))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(df) + 3), 5.5))
    for i, (lab, col) in enumerate(value_cols.items()):
        vals = df[col].astype(float).values
        err = df[err_cols[lab]].astype(float).fillna(0).values if err_cols and lab in err_cols else None
        bars = ax.bar(x + (i - (n - 1) / 2) * w, np.nan_to_num(vals), w, yerr=err, capsize=3, label=lab,
                      color=PALETTE[i % len(PALETTE)])
        for b, v in zip(bars, vals):
            ax.annotate("n/a" if np.isnan(v) else f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(df[label_col])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=n)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_box(data: dict, ylabel, title, path):
    data = {k: [x for x in v if not pd.isna(x)] for k, v in data.items()}
    data = {k: v for k, v in data.items() if v}
    if not data:
        log.warning("Boxplot '%s' übersprungen (keine Daten).", title)
        return
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(data) + 2), 5.5))
    bp = ax.boxplot(list(data.values()), patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticklabels(list(data.keys()))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_radar(series: dict, labels, title, path, vmax=1.0):
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
    for i, (name, vals) in enumerate(series.items()):
        v = [0 if pd.isna(x) else x for x in vals]
        v += v[:1]
        ax.plot(ang, v, label=name, color=PALETTE[i % len(PALETTE)], linewidth=2)
        ax.fill(ang, v, color=PALETTE[i % len(PALETTE)], alpha=0.15)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, vmax)
    ax.set_title(title, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), frameon=False)
    _save(fig, path)


def plot_heatmap(df: pd.DataFrame, title, path, cbar_label, fmt="{:.1f}", cmap="YlGnBu", vmin=None, vmax=None,
                 xlabel="", ylabel=""):
    vals = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * df.shape[1] + 3), max(3.8, 0.8 * df.shape[0] + 2.5)))
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=20, ha="right")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    lo, hi = np.nanmin(vals) if np.isfinite(vals).any() else 0, np.nanmax(vals) if np.isfinite(vals).any() else 1
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            dark = (not np.isnan(v)) and hi > lo and (v - lo) / (hi - lo) > 0.6
            ax.text(j, i, "n/a" if np.isnan(v) else fmt.format(v), ha="center", va="center",
                    color="white" if dark else "black", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
    _save(fig, path)


def plot_single_bars(labels, values, title, ylabel, path, ylim=None, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels) + 2), 5.2))
    bars = ax.bar(labels, np.nan_to_num(np.array(values, dtype=float)), color=PALETTE[: len(labels)])
    for b, v in zip(bars, values):
        ax.annotate("n/a" if pd.isna(v) else fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)
