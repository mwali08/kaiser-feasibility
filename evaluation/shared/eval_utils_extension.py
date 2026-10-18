"""
eval_utils_extension.py – Ergänzungen für TS-05, TS-06, TS-07 (überschreibt NICHTS in eval_utils.py / llm_client.py).

Enthält:
  * register_model()      – Modelle/Server zur Laufzeit in die Registry von llm_client eintragen
  * RAMMonitor            – RAM-Peak der Engine-Prozesse via psutil (Ersatz für VRAM, da keine GPU)
  * Ollama-/Server-Helfer – Erreichbarkeit, Modell-Liste, Entladen, GPU-/Internet-Check
  * run_generation_block  – Einheitlicher Messlauf (Warm-up, Generierung unter RAM-Messung, danach Judge)
  * bootstrap_ci, text_similarity, plot_lines, common_parser, setup_dry_run, fix_console

ABWEICHUNGEN VOM URSPRÜNGLICHEN PLAN (gelten für TS-05/06/07; bitte in der Arbeit dokumentieren):
  1. Mistral Small 4 (24B) wurde durch Mistral 7B ersetzt (Hardware: Laptop, 32 GB RAM, keine GPU).
  2. Qwen3 standard -> qwen3:4b-instruct-2507 (Thinking-Mode ist für den Router ungeeignet).
  3. VRAM ist ohne GPU nicht messbar -> RAM-Peak der Engine-Prozesse (psutil).
  4. Perplexity ist über Ollama nicht verfügbar -> Judge-basierter Ersatz (+ Textähnlichkeit, Antwortlänge).
  5. vLLM (AWQ/GPTQ) benötigt eine GPU -> auf diesem System nicht testbar, wird mit Warnung übersprungen.
"""
from __future__ import annotations

import argparse
import difflib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

import eval_utils as eu  # noqa: E402
import llm_client  # noqa: E402
from llm_client import query_model  # noqa: E402

log = logging.getLogger("eval")
NAN = float("nan")

MOCK_JUDGE_JSON = ('{"antwortqualitaet": 4, "deutschqualitaet": 5, "faithfulness": 4, "korrekt": true, '
                   '"begruendung": "Dry-Run (Mock-Judge)"}')


# --------------------------------------------------------------------------- #
# Allgemeines
# --------------------------------------------------------------------------- #
def fix_console() -> None:
    """Windows-PowerShell: Umlaute in der Konsole korrekt ausgeben."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def common_parser(description: str, default_model: str = "qwen3") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model, help="Modell-Kurzname aus models.json (Standard: %(default)s)")
    p.add_argument("--repetitions", type=int, default=3, help="Wiederholungen")
    p.add_argument("--limit", type=int, default=None, help="Nur die ersten N Fragen (Schnelltest)")
    p.add_argument("--out", default=None, help="Ausgabeordner (Standard: Skriptordner)")
    p.add_argument("--testset", default=None, help="Pfad zum Testset (Standard: testset.json im Skriptordner)")
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "judge"), help="Judge-Modell (Kurzname)")
    p.add_argument("--no-judge", action="store_true", help="Judge überspringen (spart auf CPU viel Zeit)")
    p.add_argument("--no-stream", action="store_true", help="Ohne Streaming (TTFT = Gesamtlatenz)")
    p.add_argument("--max-tokens", type=int, default=None, help="Max. Ausgabe-Tokens je Anfrage")
    p.add_argument("--dry-run", action="store_true", help="Trockenlauf mit Mock-Modellen (kein Server nötig)")
    return p


def register_model(key: str, spec: dict) -> None:
    """Trägt ein Modell zur Laufzeit in die Registry von llm_client ein (models.json bleibt unverändert)."""
    if getattr(llm_client, "_REGISTRY", None) is None:
        try:
            llm_client._REGISTRY = llm_client.load_registry()
        except Exception:  # noqa: BLE001
            llm_client._REGISTRY = {}
    llm_client._REGISTRY[key] = dict(spec)


def set_request_timeout(seconds: float) -> None:
    llm_client.REQUEST_TIMEOUT = float(seconds)


def setup_dry_run(judge_model: str) -> None:
    register_model(judge_model, {"provider": "mock", "family": "judge", "display_name": "Judge (Mock)",
                                 "mock_response": MOCK_JUDGE_JSON})
    log.warning("DRY-RUN: Mock-Modelle, keine echten Messwerte!")
    print("*** DRY-RUN: Mock-Modelle – die Ergebnisse sind KEINE Messwerte ***")


# --------------------------------------------------------------------------- #
# Server-/System-Helfer
# --------------------------------------------------------------------------- #
def server_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """Alle OpenAI-kompatiblen Server (Ollama, llama-server, vLLM) bieten GET <base_url>/models."""
    try:
        return requests.get(base_url.rstrip("/") + "/models", timeout=timeout).status_code < 500
    except requests.RequestException:
        return False


def ollama_host(base_url: str) -> str:
    return base_url.rstrip("/").rsplit("/v1", 1)[0]


def ollama_list_tags(host: str) -> list[str]:
    try:
        r = requests.get(host + "/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        log.warning("Ollama-Modellliste nicht abrufbar: %s", exc)
        return []


def ollama_has_tag(tags: list[str], wanted: str) -> bool:
    return wanted in tags or f"{wanted}:latest" in tags


def ollama_unload(host: str, tag: str) -> None:
    """Entlädt ein Modell aus dem RAM (keep_alive=0), damit Messungen nicht verfälscht werden."""
    try:
        requests.post(host + "/api/generate", json={"model": tag, "keep_alive": 0}, timeout=30)
    except Exception:  # noqa: BLE001
        pass


def ollama_unload_all(host: str) -> None:
    try:
        for m in requests.get(host + "/api/ps", timeout=5).json().get("models", []):
            ollama_unload(host, m["name"])
    except Exception:  # noqa: BLE001
        pass


def has_gpu() -> bool:
    """True, wenn eine NVIDIA-GPU per nvidia-smi gefunden wird (Voraussetzung für vLLM)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:  # noqa: BLE001
        return False


def internet_reachable(timeout: float = 3.0) -> bool:
    """Optionaler Nachweis für den Offline-Betrieb: Läuft die Messung ohne Internet? (nur auf Wunsch aufrufen)"""
    try:
        requests.head("https://pypi.org", timeout=timeout)
        return True
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# RAM-Messung (Ersatz für VRAM)
# --------------------------------------------------------------------------- #
class RAMMonitor:
    """Misst den RAM-Spitzenwert (GB) aller Prozesse, deren Name einen der Suchbegriffe enthält
    (z. B. 'ollama' -> ollama.exe, ollama-Runner; 'llama-server'). Zusätzlich RAM-Anstieg des Gesamtsystems.
    Metrik: Working Set (RSS) der Engine-Prozesse. Bei memory-gemappten Modellen kann der Wert unter der
    Modellgröße liegen, solange nicht alle Seiten benutzt wurden – daher Messung inkl. Warm-up."""

    def __init__(self, process_names: list[str], interval: float = 0.5):
        self.names = [n.lower() for n in process_names]
        self.interval = interval
        self.peak_gb = self.baseline_gb = self.peak_sys_gb = self.baseline_sys_gb = 0.0
        self._procs: list[psutil.Process] = []
        self._last_refresh = 0.0

    def _refresh(self) -> None:
        procs = []
        for p in psutil.process_iter(["name"]):
            try:
                nm = (p.info["name"] or "").lower()
                if any(n in nm for n in self.names):
                    procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self._procs, self._last_refresh = procs, time.time()

    def _engine_gb(self) -> float:
        if time.time() - self._last_refresh > 4:
            self._refresh()
        total = 0
        for p in self._procs:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total / 1024 ** 3

    def _sample(self) -> None:
        self.peak_gb = max(self.peak_gb, self._engine_gb())
        self.peak_sys_gb = max(self.peak_sys_gb, psutil.virtual_memory().used / 1024 ** 3)

    def __enter__(self):
        self._refresh()
        self.baseline_gb = self.peak_gb = self._engine_gb()
        self.baseline_sys_gb = self.peak_sys_gb = psutil.virtual_memory().used / 1024 ** 3
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:  # noqa: BLE001
                pass

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)
        try:
            self._sample()
        except Exception:  # noqa: BLE001
            pass
        return False

    @property
    def meta(self) -> dict:
        return {"ram_peak_gb": round(self.peak_gb, 3),
                "ram_delta_gb": round(max(self.peak_gb - self.baseline_gb, 0), 3),
                "ram_system_delta_gb": round(max(self.peak_sys_gb - self.baseline_sys_gb, 0), 3)}


# --------------------------------------------------------------------------- #
# Einheitlicher Messlauf (TS-05 / TS-06)
# --------------------------------------------------------------------------- #
def text_similarity(a: str, b: str) -> float:
    """Zeichenbasierte Ähnlichkeit 0-100 % (difflib). Bei Temperatur 0 und gleichem Modell ~100 %."""
    if not a or not b:
        return NAN
    return 100 * difflib.SequenceMatcher(None, a, b).ratio()


def run_generation_block(model_key, questions, repetitions, system_prompt, max_tokens, judge_model, use_judge,
                         stream, process_names, desc, extra_fields=None):
    """Phase 0: Warm-up (misst Ladezeit). Phase 1: Generierung UNTER RAM-Messung. Phase 2: Judge (außerhalb der
    Messung, damit das Judge-Modell den RAM-Peak der Engine nicht verfälscht).
    Rückgabe: (records, meta). Bei Warm-up-Fehler: ([], meta mit 'warmup_error')."""
    extra_fields = extra_fields or {}
    meta = {"ladezeit_ms": NAN, "warmup_error": None}
    records = []
    spec = llm_client.get_model_spec(model_key)
    with RAMMonitor(process_names) as mon:
        w = query_model(model_key, "Antworte mit einem Wort: Hallo.", stream=False, temperature=0.0,
                        max_tokens=16, retries=0)
        meta["ladezeit_ms"] = w["total_latency_ms"]
        if w["error"]:
            meta["warmup_error"] = w["error"]
        else:
            todo = [(q, rep) for rep in range(1, repetitions + 1) for q in questions]
            for q, rep in tqdm(todo, desc=desc, unit="Anfrage"):
                prompt = eu.build_prompt(q)
                rec = {"model": model_key, "model_name": spec.get("display_name", model_key),
                       "family": spec.get("family", "unbekannt"), "question_id": q["id"], "category": q["category"],
                       "is_fact": q["is_fact"], "repetition": rep, "prompt": prompt, "question": q["question"],
                       "reference_answer": q["reference_answer"], "context": q["context"], **extra_fields}
                try:
                    res = query_model(model_key, prompt, stream=stream, temperature=0.0,
                                      system_prompt=system_prompt, max_tokens=max_tokens, retries=1)
                    rec.update({"response_text": res["response_text"], "ttft_ms": res["ttft_ms"],
                                "total_latency_ms": res["total_latency_ms"], "token_count": res["token_count"],
                                "tps": res["tps"], "error": res["error"]})
                    rec["valid"] = res["error"] is None and len(res["response_text"].strip()) >= 2
                    rec["antwortlaenge_zeichen"] = len(res["response_text"])
                    if not rec["valid"]:
                        log.warning("Ungültige Antwort | %s | %s | Rep %d | %s", model_key, q["id"], rep, res["error"])
                except Exception as exc:  # noqa: BLE001 – Lauf darf nie abbrechen
                    log.exception("Unerwarteter Fehler | %s | %s", model_key, q["id"])
                    rec.update({"valid": False, "error": f"{type(exc).__name__}: {exc}", "response_text": "",
                                "antwortlaenge_zeichen": 0})
                records.append(rec)
    meta.update(mon.meta)

    # Phase 2: Judge (Cache: identische Antworten – bei Temperatur 0 häufig – nur einmal bewerten)
    cache = {}
    todo_j = [r for r in records if r["valid"] and use_judge]
    for rec in tqdm(todo_j, desc=f"Judge {desc.strip()}", unit="Bewertung", disable=not todo_j):
        key = (rec["question_id"], rec["response_text"])
        if key not in cache:
            cache[key] = eu.judge_answer(judge_model, rec["question"], rec["response_text"],
                                         rec["reference_answer"], rec["context"])
        rec.update(cache[key])
    for rec in records:
        for k in ("antwortqualitaet", "deutschqualitaet", "faithfulness", "korrekt", "judge_begruendung", "judge_error"):
            rec.setdefault(k, None)
    return records, meta


# --------------------------------------------------------------------------- #
# Statistik und Diagramme
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05):
    """Mittelwert und (1-alpha)-Bootstrap-Konfidenzintervall. Rückgabe: (mean, low, high)."""
    v = np.asarray([x for x in values if not pd.isna(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else NAN), NAN, NAN
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def plot_lines(x, series: dict, title, xlabel, ylabel, path, hlines: dict | None = None, yerr: dict | None = None,
               logx=False):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (lab, ys) in enumerate(series.items()):
        e = yerr.get(lab) if yerr else None
        ax.errorbar(x, ys, yerr=e, marker="o", linewidth=2, capsize=3, label=lab, color=eu.PALETTE[i % len(eu.PALETTE)])
    for j, (lab, y) in enumerate((hlines or {}).items()):
        ax.axhline(y, linestyle="--", color="#d62728", alpha=0.8, label=lab)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in x])
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    eu._save(fig, path)


def plot_scatter_labeled(xs, ys, labels, title, xlabel, ylabel, path, vline=None, vline_label=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        if pd.isna(x) or pd.isna(y):
            continue
        ax.scatter(x, y, s=140, color=eu.PALETTE[i % len(eu.PALETTE)], zorder=3)
        ax.annotate(lab, (x, y), xytext=(8, 8), textcoords="offset points")
    if vline is not None:
        ax.axvline(vline, linestyle="--", color="#d62728", label=vline_label)
        ax.legend(frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    eu._save(fig, path)
