kaiser/
├── README.md                         # Projektübersicht und gemeinsame Startanleitung
├── compose.yaml                      # Gemeinsame Services und Netzwerk
├── .env.example                      # Konfigurationsvorlage ohne Geheimnisse
├── .gitignore
│
├── backend/                          # Zentrale FastAPI / Proxy
│   ├── README.md
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI erstellen
│   │   ├── config.py                 # Umgebungsvariablen einlesen
│   │   ├── schemas.py                # Anfrage- und Antwortmodelle
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   └── ask.py                # POST /ask
│   │   └── clients/
│   │       ├── __init__.py
│   │       └── orchestration.py      # n8n-Webhook über httpx aufrufen
│   └── tests/
│       ├── test_ask.py
│       └── test_orchestration.py
│
├── vector_retrieval/                 # Dokumentverarbeitung und Vector-Suche
│   ├── README.md
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── schemas.py                # Tool-Antwort: results
│   │   ├── database.py               # Qdrant-Verbindung
│   │   ├── mcp_server.py             # Retrieval-Tool für n8n
│   │   ├── search.py                 # Relevante Textstellen suchen
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       ├── loader.py
│   │       ├── chunker.py
│   │       ├── embeddings.py
│   │       └── pipeline.py           # Dokumentverarbeitung starten
│   └── tests/
│       └── test_search.py
│
├── graph_qa/                         # Knowledge Graph und Qualitätssicherung
│   ├── README.md
│   ├── requirements.txt              # Graph-Service
│   ├── requirements-tools.txt        # Extraktion, Evaluation und Tracking
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── schemas.py                # Graph-Modelle und Tool-Antwort: triples
│   │   ├── database.py               # Neo4j-Verbindung
│   │   ├── mcp_server.py             # Graph-Tool für n8n
│   │   ├── search.py                 # Passende Beziehungen abfragen
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       ├── extractor.py          # Informationen aus Text extrahieren
│   │       └── pipeline.py           # Graph aufbauen
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py              # RAGAS-Auswertung
│   │   ├── test_dataset.json
│   │   └── custom_metrics.py         # Optional
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── mlflow_tracking.py
│   │   └── prompt_registry.py        # Optional
│   └── tests/
│       └── test_search.py
│
├── llm_orchestration/                # n8n und lokales LLM
│   ├── README.md                     # Webhook, Workflow-Import, Zugangsdaten-Setup
│   ├── n8n/
│   │   └── workflows/
│   │       └── ask_workflow.json     # Exportierter Workflow ohne Geheimnisse
│   ├── ollama/
│   │   └── Modelfile                 # Optional: Modellparameter/System-Prompt
│   ├── mocks/
│   │   ├── mcp_server.py             # Testersatz für Vector- und Graph-Tools
│   │   └── requirements.txt
│   └── tests/
│       ├── test_ollama.sh
│       ├── test_webhook.sh
│       └── fixtures/
│           ├── webhook_request.json
│           └── webhook_response.json
│
├── tests/                            # Gemeinsame Integrationstests
│   └── test_end_to_end.py
│
└── docs/
    ├── architecture.md              # Gesamter Datenfluss
    ├── responsibilities.md          # Zuständigkeiten im Team
    ├── setup.md                     # Lokales Setup
    ├── air_gap.md                   # Offline-Vorbereitung und Betrieb
    └── contracts/
        ├── backend_api.md           # Client ↔ FastAPI
        ├── orchestration_webhook.md # FastAPI ↔ n8n
        ├── vector_tool.md           # results: text, source, date
        └── graph_tool.md            # triples: subject, predicate, object
