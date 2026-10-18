"""
eval_utils_phase_d.py - Ergänzungen für Phase D (TS-08 bis TS-12). Überschreibt nichts Bestehendes.
  * embed_texts()      Embeddings über Ollama (/api/embed, nomic-embed-text) inkl. Latenz
  * cosine_matrix()    Cosine-Similarity-Matrix
  * register_variant() Modell-Variante zur Laufzeit registrieren (anderer Seed, extra_body, ...)
  * wilson_ci(), pct() kleine Statistik-Helfer
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import requests

import eval_utils_extension as ex
import llm_client

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
_DRY = False


def set_dry_run(flag: bool) -> None:
    global _DRY
    _DRY = flag


def _mock_vec(text: str) -> np.ndarray:
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=64)


def embed_texts(texts, model: str = EMBED_MODEL, host: str | None = None, timeout: float = 120):
    """Rückgabe: (Array [n, d], Latenz in ms). Hinweis nomic-embed-text: Präfixe 'search_query: ' /
    'search_document: ' verbessern die Qualität (vom Aufrufer zu setzen)."""
    t0 = time.perf_counter()
    if _DRY:
        vecs = [_mock_vec(t) for t in texts]
    else:
        r = requests.post(f"{host or OLLAMA_URL}/api/embed", json={"model": model, "input": list(texts)}, timeout=timeout)
        r.raise_for_status()
        vecs = r.json()["embeddings"]
    return np.asarray(vecs, dtype=float), (time.perf_counter() - t0) * 1000


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def register_variant(new_key: str, base_key: str, **overrides) -> str:
    """Kopiert die Registry-Einstellungen von base_key unter new_key und überschreibt einzelne Felder
    (z. B. seed=11 oder extra_body={...}). extra_body wird mit vorhandenem extra_body zusammengeführt."""
    spec = dict(llm_client.get_model_spec(base_key))
    if "extra_body" in overrides:
        overrides["extra_body"] = {**spec.get("extra_body", {}), **overrides["extra_body"]}
    spec.update(overrides)
    spec["display_name"] = overrides.get("display_name", spec.get("display_name", new_key))
    ex.register_model(new_key, spec)
    return new_key


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95-%-Wilson-Intervall (in %) für einen Anteil k/n - sinnvoll bei kleinen Stichproben."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def pct(values, p):
    v = [x for x in values if x == x and x is not None]
    return float(np.percentile(v, p)) if v else float("nan")