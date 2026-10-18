"""
fw_common.py - gemeinsamer Messrahmen für die vier TS-12-Framework-Skripte (nur Python-Standardbibliothek,
damit er in jeder Framework-Umgebung ohne Zusatzpakete läuft).

Identisch für alle Frameworks (damit nur die Orchestrierung verglichen wird):
  * Fragestellung, Prompts, Parser für das Critic-Urteil, Erfolgsprüfung (Task Success)
  * Token-Zählung: ein lokaler Zähl-Proxy (TokenProxy) sitzt zwischen Framework und Ollama und summiert die von
    Ollama gemeldeten usage-Werte aller Chat-Anfragen -> gleiche Messmethode für alle Frameworks
  * Zeitmessung: setup_s = Import + Aufbau des Frameworks, latenz_s = reine Ausführung des Workflows
    (Modell-Warm-up vorab, damit die Ladezeit des Modells keinem Framework zugeschlagen wird)
  * Robustheit: Jede Ausnahme wird abgefangen, die CSV wird immer geschrieben (task_success = 0, fehler = Text)

Workflow (identisch): Primary antwortet -> Critic prüft ("URTEIL: OK|FEHLER") -> bei FEHLER überarbeitet der Primary
mit der Kritik (max. MAX_KORREKTUREN = 2 Korrekturrunden; nach der letzten Runde prüft der Critic noch einmal).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

FRAGE = "Wie viele Bundesländer hat Österreich? Nenne auch die Hauptstadt von Südtirol."
PRIMARY_MODEL = "qwen3:4b-instruct-2507-q8_0"
CRITIC_MODEL = "gemma3:4b"
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MAX_KORREKTUREN = 2
MAX_TOKENS = 400
TEMPERATURE = 0.0

PRIMARY_SYSTEM = "Du bist ein sachlicher Assistent. Antworte auf Deutsch, knapp und korrekt."
CRITIC_SYSTEM = "Du bist ein strenger Faktenprüfer. Antworte auf Deutsch."

# Hinweis: Heuristik per String-Vergleich - die finale Antwort steht zur manuellen Kontrolle in der CSV.
ERFOLGS_MUSTER = ["italien", "italienisch", "italy", "nicht zu österreich", "nicht teil von österreich",
                  "kein teil österreichs", "nicht zu oesterreich", "Bozen", "Südtirol"]


def primary_prompt() -> str:
    return FRAGE


def critic_prompt(antwort: str) -> str:
    return ("Prüfe die folgende Antwort auf sachliche Fehler, falsche Annahmen und irreführende Auslassungen.\n\n"
            f"Frage: {FRAGE}\n\nAntwort: {antwort}\n\n"
            "Antworte exakt in diesem Format:\nURTEIL: OK oder FEHLER\nBEGRÜNDUNG: <ein bis zwei Sätze>")


def revision_prompt(antwort: str, kritik: str) -> str:
    return (f"Frage: {FRAGE}\n\nDeine bisherige Antwort: {antwort}\n\nKritik eines Prüfers: {kritik}\n\n"
            "Überarbeite die Antwort unter Berücksichtigung der Kritik. Gib NUR die verbesserte Antwort aus.")


def parse_verdict(text: str):
    """Rückgabe: (urteil, kritik) mit urteil in {'OK', 'FEHLER', 'UNKLAR'}. UNKLAR wird wie OK behandelt (Abbruch)."""
    m = re.search(r"URTEIL:\s*\**\s*(OK|FEHLER)", text or "", re.IGNORECASE)
    if not m:
        return "UNKLAR", (text or "").strip()
    kritik = (text.split("BEGRÜNDUNG:", 1)[1] if "BEGRÜNDUNG:" in text else text).strip()
    return m.group(1).upper(), kritik


def task_success(antwort: str) -> int:
    a = (antwort or "").lower()
    return int(any(p in a for p in ERFOLGS_MUSTER))


# --------------------------------------------------------------------------- #
# Token-Zähl-Proxy
# --------------------------------------------------------------------------- #
class TokenProxy:
    """Leitet alle Anfragen unverändert an Ollama weiter und summiert die usage-Felder der Antworten
    (JSON und SSE-Streams mit usage-Chunk). Das Framework wird auf proxy.base_url gerichtet."""

    def __init__(self, upstream_base_url: str):
        u = urlparse(upstream_base_url)
        self.origin = f"{u.scheme}://{u.netloc}"
        self.prompt_tokens = self.completion_tokens = self.requests = 0
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _forward(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization", "accept")}
                req = urllib.request.Request(outer.origin + self.path, data=body, headers=headers, method=self.command)
                try:
                    with urllib.request.urlopen(req, timeout=900) as r:
                        code, data, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as e:
                    code, data, ctype = e.code, e.read(), e.headers.get("Content-Type", "application/json")
                except Exception as e:  # noqa: BLE001
                    code, data, ctype = 502, json.dumps({"error": str(e)}).encode(), "application/json"
                if code == 200 and self.command == "POST":
                    outer._count(data, ctype)
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _forward

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}{u.path.rstrip('/')}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _count(self, data: bytes, ctype: str):
        usages = []
        text = data.decode("utf-8", "replace")
        if "event-stream" in ctype:
            for line in text.splitlines():
                if line.startswith("data:") and "[DONE]" not in line:
                    try:
                        usages.append(json.loads(line[5:].strip()).get("usage"))
                    except json.JSONDecodeError:
                        pass
        else:
            try:
                usages.append(json.loads(text).get("usage"))
            except (json.JSONDecodeError, AttributeError):
                pass
        usages = [u for u in usages if u]
        if usages:
            u = usages[-1]
            with self._lock:
                self.prompt_tokens += int(u.get("prompt_tokens") or 0)
                self.completion_tokens += int(u.get("completion_tokens") or 0)
                self.requests += 1

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def _warmup(base_url: str, models):
    """Modelle direkt (am Proxy vorbei) laden, damit die Ladezeit nicht gemessen wird."""
    for m in models:
        body = json.dumps({"model": m, "messages": [{"role": "user", "content": "Hallo"}], "max_tokens": 1}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"})
        urllib.request.urlopen(req, timeout=900).read()


# --------------------------------------------------------------------------- #
# Messlauf + CSV
# --------------------------------------------------------------------------- #
FIELDS = ["framework", "lauf", "latenz_s", "setup_s", "tokens_total", "tokens_prompt", "tokens_completion",
          "llm_anfragen", "task_success", "korrekturrunden", "critic_finales_urteil", "fehler", "finale_antwort"]


def _fmt(v):
    return str(v).replace(".", ",") if isinstance(v, float) else v


def run_benchmark(framework: str, setup_fn, script_file: str):
    """setup_fn(cfg) importiert/baut das Framework und liefert run() -> dict(final_answer, korrekturrunden, finales_urteil).
    cfg = dict(base_url, primary, critic, frage, max_tokens, temperature)."""
    ap = argparse.ArgumentParser(description=f"TS-12 Framework-Test: {framework}")
    ap.add_argument("--repetitions", type=int, default=1, help="Anzahl Durchläufe (Standard 1; für die Arbeit >= 3)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama (OpenAI-kompatibel)")
    ap.add_argument("--primary", default=PRIMARY_MODEL)
    ap.add_argument("--critic", default=CRITIC_MODEL)
    ap.add_argument("--out", default=None, help="Ausgabeordner (Standard: Skriptordner)")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(script_file).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    slug = Path(script_file).stem.replace("test_", "")
    rows = []
    proxy = TokenProxy(args.base_url)
    try:
        if not args.no_warmup:
            try:
                print(f"[{framework}] Modelle laden (Warm-up) ...")
                _warmup(args.base_url, [args.primary, args.critic])
            except Exception as exc:  # noqa: BLE001
                print(f"[{framework}] WARNUNG: Warm-up fehlgeschlagen ({exc}) - ist Ollama gestartet?")
        for lauf in range(1, args.repetitions + 1):
            row = {"framework": framework, "lauf": lauf, "latenz_s": float("nan"), "setup_s": float("nan"), "tokens_total": 0,
                   "tokens_prompt": 0, "tokens_completion": 0, "llm_anfragen": 0, "task_success": 0, "korrekturrunden": 0,
                   "critic_finales_urteil": "", "fehler": "", "finale_antwort": ""}
            proxy.prompt_tokens = proxy.completion_tokens = proxy.requests = 0
            cfg = {"base_url": proxy.base_url, "primary": args.primary, "critic": args.critic, "frage": FRAGE,
                   "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}
            t_setup = time.perf_counter()
            t_run = None
            try:
                run = setup_fn(cfg)
                row["setup_s"] = time.perf_counter() - t_setup
                t_run = time.perf_counter()
                res = run()
                row["latenz_s"] = time.perf_counter() - t_run
                row["finale_antwort"] = (res.get("final_answer") or "").strip()
                row["korrekturrunden"] = int(res.get("korrekturrunden", 0))
                row["critic_finales_urteil"] = res.get("finales_urteil", "")
                row["task_success"] = task_success(row["finale_antwort"])
            except BaseException as exc:  # noqa: BLE001 - jede Framework-Störung wird protokolliert, Lauf endet mit Erfolg 0
                if isinstance(exc, KeyboardInterrupt):
                    raise
                row["fehler"] = f"{type(exc).__name__}: {exc}"[:500]
                row["latenz_s"] = (time.perf_counter() - t_run) if t_run else float("nan")
                if row["setup_s"] != row["setup_s"]:
                    row["setup_s"] = time.perf_counter() - t_setup
                print(f"[{framework}] FEHLER im Lauf {lauf}: {row['fehler']}", file=sys.stderr)
            row.update(tokens_prompt=proxy.prompt_tokens, tokens_completion=proxy.completion_tokens,
                       tokens_total=proxy.prompt_tokens + proxy.completion_tokens, llm_anfragen=proxy.requests)
            rows.append(row)
            print(f"[{framework}] Lauf {lauf}: Erfolg={row['task_success']} | {row['latenz_s']:.1f} s | "
                  f"{row['tokens_total']} Token | {row['llm_anfragen']} LLM-Anfragen | Korrekturrunden={row['korrekturrunden']}")
    finally:
        proxy.close()
        path = out / f"results_{slug}.csv"          # Excel-DE: Semikolon, Dezimalkomma, UTF-8 mit BOM
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
            w.writeheader()
            for r in rows:
                w.writerow({k: _fmt(r.get(k, "")) if k != "finale_antwort" else str(r.get(k, "")).replace("\n", " ") for k in FIELDS})
        print(f"[{framework}] gespeichert: {path}")