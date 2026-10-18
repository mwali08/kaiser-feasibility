"""
llm_client.py – Einheitliche API-Schicht für OpenAI, Anthropic, Google, Mistral
sowie lokale OpenAI-kompatible Server (Ollama, vLLM, LM Studio).

Verwendung:
    from llm_client import query_model
    r = query_model("qwen3", "Was ist die Hauptstadt von Österreich?")
    r["response_text"], r["ttft_ms"], r["total_latency_ms"], r["token_count"], r["tps"]

Modelle werden über shared/models.json (Kurzname -> Provider/Modell-ID) oder
direkt als "provider:modell-id" (z. B. "openai:gpt-4o") angesprochen.
API-Keys kommen aus der .env-Datei (siehe .env.example).

Metrik-Definitionen (für den Methodikteil der Arbeit):
  * TTFT   = Zeit von Anfragebeginn bis zum ersten empfangenen Token/Textfragment
             (bei Reasoning-Modellen zählt auch das erste Reasoning-Fragment).
  * Gesamtlatenz = Zeit von Anfragebeginn bis zum Ende der Antwort.
  * TPS    = Token / Generierungszeit (Gesamtlatenz – TTFT). Ohne Streaming oder
             bei sehr kurzen Antworten: Token / Gesamtlatenz.
  * Token  = vom Server gemeldete Ausgabe-Tokens (inkl. Reasoning-Tokens);
             falls nicht gemeldet: Schätzung Zeichen/4 (Flag token_count_estimated).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    _HERE = Path(__file__).resolve().parent
    load_dotenv(_HERE.parent / ".env")
    load_dotenv()
except ImportError:  # python-dotenv optional
    pass

SEED = int(os.getenv("LLM_SEED", "42"))
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT_S", "300"))
DEFAULT_MAX_TOKENS = 2048
KNOWN_PROVIDERS = {"openai", "anthropic", "google", "mistral", "openai_compatible", "ollama_native", "mock"}

_REGISTRY: dict | None = None


# --------------------------------------------------------------------------- #
# Modell-Registry
# --------------------------------------------------------------------------- #
def load_registry() -> dict:
    path = Path(os.getenv("MODELS_JSON", Path(__file__).resolve().parent / "models.json"))
    with open(path, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def get_model_spec(model: str) -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    if model in _REGISTRY:
        spec = dict(_REGISTRY[model])
        spec["name"] = model
        return spec
    if ":" in model:
        provider, model_id = model.split(":", 1)
        if provider in KNOWN_PROVIDERS:
            return {"name": model, "provider": provider, "model_id": model_id, "family": provider}
    raise KeyError(f"Modell '{model}' nicht in models.json und kein gültiges 'provider:modell-id'.")


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def strip_thinking(text: str) -> str:
    """Entfernt Thinking-Inhalte, auch wenn nur der schließende Tag vorhanden ist."""
    # 1. Vollständige <think>...</think>-Blöcke entfernen
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. Falls nur </think> vorhanden ist: alles davor abschneiden
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # 3. Falls nur <think> ohne Ende vorhanden: alles danach verwerfen
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def _fail(model: str, msg: str) -> dict:
    nan = float("nan")
    return {"response_text": "", "ttft_ms": nan, "total_latency_ms": nan, "token_count": 0,
            "tps": nan, "model": model, "error": msg, "token_count_estimated": False}


# --------------------------------------------------------------------------- #
# Provider-Implementierungen – liefern dict(text, ttft[s]|None, total[s], tokens|None)
# --------------------------------------------------------------------------- #
def _call_openai_like(spec, prompt, system, stream, temperature, max_tokens):
    """OpenAI, Mistral (OpenAI-kompatible API) und lokale Server (Ollama/vLLM)."""
    from openai import OpenAI

    provider = spec["provider"]
    base_url = spec.get("base_url") or {"mistral": "https://api.mistral.ai/v1"}.get(provider)
    key_env = spec.get("api_key_env") or {"openai": "OPENAI_API_KEY", "mistral": "MISTRAL_API_KEY"}.get(
        provider, "LOCAL_API_KEY")
    client = OpenAI(api_key=os.getenv(key_env) or "not-needed", base_url=base_url,
                    timeout=REQUEST_TIMEOUT, max_retries=0)

    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kwargs = {"model": spec["model_id"], "messages": messages}
    if spec.get("send_temperature", True):
        kwargs["temperature"] = temperature
    kwargs[spec.get("max_tokens_param", "max_tokens")] = max_tokens
    if provider in ("openai", "openai_compatible"):
        kwargs["seed"] = spec.get("seed", SEED)
    if spec.get("extra_body"):
        kwargs["extra_body"] = spec["extra_body"]

    t0 = time.perf_counter()
    if not stream:
        r = client.chat.completions.create(**kwargs)
        total = time.perf_counter() - t0
        return {"text": r.choices[0].message.content or "", "ttft": None, "total": total,
                "tokens": r.usage.completion_tokens if r.usage else None}

    kwargs["stream"] = True
    if provider in ("openai", "openai_compatible"):
        kwargs["stream_options"] = {"include_usage": True}
    parts, ttft, tokens, n_chunks = [], None, None, 0
    for chunk in client.chat.completions.create(**kwargs):
        if getattr(chunk, "usage", None):
            tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None) or ""
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
        if (piece or reasoning) and ttft is None:
            ttft = time.perf_counter() - t0
        if piece:
            parts.append(piece)
            n_chunks += 1
    total = time.perf_counter() - t0
    return {"text": "".join(parts), "ttft": ttft, "total": total, "tokens": tokens}


def _call_anthropic(spec, prompt, system, stream, temperature, max_tokens):
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=REQUEST_TIMEOUT, max_retries=0)
    kwargs = {"model": spec["model_id"], "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    if spec.get("send_temperature", True):
        kwargs["temperature"] = temperature

    t0 = time.perf_counter()
    if not stream:
        r = client.messages.create(**kwargs)
        total = time.perf_counter() - t0
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return {"text": text, "ttft": None, "total": total, "tokens": r.usage.output_tokens}

    parts, ttft = [], None
    with client.messages.stream(**kwargs) as s:
        for piece in s.text_stream:
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(piece)
        final = s.get_final_message()
    total = time.perf_counter() - t0
    return {"text": "".join(parts), "ttft": ttft, "total": total, "tokens": final.usage.output_tokens}


def _call_google(spec, prompt, system, stream, temperature, max_tokens):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    cfg_args = {"max_output_tokens": max_tokens}
    if spec.get("send_temperature", True):
        cfg_args["temperature"] = temperature
    if system:
        cfg_args["system_instruction"] = system
    cfg = types.GenerateContentConfig(**cfg_args)

    t0 = time.perf_counter()
    if not stream:
        r = client.models.generate_content(model=spec["model_id"], contents=prompt, config=cfg)
        total = time.perf_counter() - t0
        um = getattr(r, "usage_metadata", None)
        return {"text": r.text or "", "ttft": None, "total": total,
                "tokens": getattr(um, "candidates_token_count", None) if um else None}

    parts, ttft, tokens = [], None, None
    for chunk in client.models.generate_content_stream(model=spec["model_id"], contents=prompt, config=cfg):
        if chunk.text:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(chunk.text)
        um = getattr(chunk, "usage_metadata", None)
        if um and getattr(um, "candidates_token_count", None):
            tokens = um.candidates_token_count
    total = time.perf_counter() - t0
    return {"text": "".join(parts), "ttft": ttft, "total": total, "tokens": tokens}

def _call_ollama_native(spec, prompt, system, stream, temperature, max_tokens):
    """Nativer Ollama /api/chat Endpunkt – unterstützt think:false korrekt."""
    import requests as _requests

    base = spec.get("base_url", "http://localhost:11434/v1").replace("/v1", "")
    url = f"{base}/api/chat"
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload = {
        "model": spec["model_id"],
        "messages": messages,
        "stream": stream,
        "think": spec.get("think", False),   # <-- HIER wird Thinking gesteuert
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": spec.get("seed", SEED)},
    }

    t0 = time.perf_counter()
    if not stream:
        r = _requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        total = time.perf_counter() - t0
        msg = data.get("message", {})
        text = msg.get("content", "") or ""
        return {"text": text, "ttft": None, "total": total, "tokens": data.get("eval_count")}

    parts, ttft = [], None
    with _requests.post(url, json=payload, timeout=REQUEST_TIMEOUT, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            piece = msg.get("content", "") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            if piece:
                parts.append(piece)
            if chunk.get("done"):
                total = time.perf_counter() - t0
                return {"text": "".join(parts), "ttft": ttft, "total": total, "tokens": chunk.get("eval_count")}
    total = time.perf_counter() - t0
    return {"text": "".join(parts), "ttft": ttft, "total": total, "tokens": None}

def _call_mock(spec, prompt, system, stream, temperature, max_tokens):
    """Offline-Testprovider (kein Netzwerk) – für Trockenläufe der Skripte."""
    t0 = time.perf_counter()
    time.sleep(spec.get("mock_delay_s", 0.01))
    ttft = time.perf_counter() - t0
    time.sleep(spec.get("mock_delay_s", 0.01))
    text = spec.get("mock_response", "Mock-Antwort.")
    return {"text": text, "ttft": ttft if stream else None,
            "total": time.perf_counter() - t0, "tokens": len(text.split())}


_PROVIDERS = {"openai": _call_openai_like, "mistral": _call_openai_like, "openai_compatible": _call_openai_like, "ollama_native": _call_ollama_native,
              "anthropic": _call_anthropic, "google": _call_google, "mock": _call_mock}


# --------------------------------------------------------------------------- #
# Öffentliche Schnittstelle
# --------------------------------------------------------------------------- #
def query_model(model, prompt, stream=True, temperature=0.0, system_prompt=None,
                max_tokens=None, retries=2):
    """
    Returns: dict mit keys:
      response_text (str), ttft_ms (float), total_latency_ms (float), token_count (int),
      tps (float), model (str), error (str | None), token_count_estimated (bool)
    Wirft nie eine Exception: Fehler stehen in 'error' (response_text dann leer).
    """
    try:
        spec = get_model_spec(model)
    except Exception as exc:  # noqa: BLE001
        return _fail(model, f"Konfiguration: {exc}")
    fn = _PROVIDERS.get(spec.get("provider"))
    if fn is None:
        return _fail(model, f"Unbekannter Provider: {spec.get('provider')}")
    max_tokens = max_tokens or spec.get("max_tokens") or DEFAULT_MAX_TOKENS

    last_error = None
    for attempt in range(retries + 1):
        try:
            raw = fn(spec, prompt, system_prompt, stream, temperature, max_tokens)
            text = strip_thinking(raw["text"]) if spec.get("strip_thinking", True) else raw["text"].strip()
            total_ms = raw["total"] * 1000
            ttft_ms = raw["ttft"] * 1000 if raw.get("ttft") is not None else total_ms
            tokens, estimated = raw.get("tokens"), False
            if not tokens:
                tokens, estimated = max(1, round(len(raw["text"]) / 4)), True
            gen_s = (total_ms - ttft_ms) / 1000
            tps = tokens / gen_s if gen_s > 0.001 else tokens / (total_ms / 1000)
            return {"response_text": text, "ttft_ms": ttft_ms, "total_latency_ms": total_ms,
                    "token_count": int(tokens), "tps": tps, "model": model,
                    "error": None if text else "Leere Antwort", "token_count_estimated": estimated}
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2 ** attempt)
    return _fail(model, last_error or "Unbekannter Fehler")
