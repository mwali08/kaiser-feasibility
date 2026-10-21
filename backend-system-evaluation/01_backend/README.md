# Initialisierung

1. `01_backend` in IntelliJ/PyCharm öffnen.
2. Umgebungen installieren:
   ```powershell
   uv sync --locked --project .\fastapi-evaluation
   uv sync --locked --project .\litestar-evaluation
   uv sync --locked --project .\sanic-evaluation
   ```
3. Zentrales Profil **FastAPI**, **Litestar** oder **Sanic** starten.
4. In der jeweiligen `main.py` selbst weiterentwickeln.

CLI aus `01_backend` (je ein Terminal):

```powershell
uv --directory .\fastapi-evaluation run --locked python -m uvicorn main:app --reload --port 8000
uv --directory .\litestar-evaluation run --locked python -m uvicorn main:app --reload --port 8001
uv --directory .\sanic-evaluation run --locked python -m uvicorn main:app --reload --port 8002
```

Profile: `.run/`; jedes verwendet seine eigene `.venv`. Beenden: `Strg+C`.
Bei einer noch aktivierten fremden Umgebung vorher `deactivate` ausführen.
Sanic läuft unter Windows über Uvicorn im ASGI-Modus.
