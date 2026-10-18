

# KAISER Feasibility Studies (TS-01 bis TS-14)

##### Verfasser: Ramis Ekici

##### Datum: 18.10.2026

Dieses Repository enthält die Python-Skripte zur empirischen Evaluierung der KAISER-Architektur (lokales Multi-Agenten-System für das BMLV). 

## Einrichtung

1. **Ollama starten:** Stelle sicher, dass Ollama läuft. Für die Tests sollten alle Modelle gleichzeitig im RAM bleiben, um Latenz-Verzerrungen durch Nachladen zu vermeiden:
   
   ```powershell
   $env:OLLAMA_MAX_LOADED_MODELS="4"
   $env:OLLAMA_KEEP_ALIVE="2h"
   ollama serve
   ```

2. **Abhängigkeiten installieren:**
   
   `pip install -r requirements.txt`

3. **Modelle:** Die Auswahl der lokalen Modelle (z.B. `qwen3:4b-instruct-2507-q4_K_M`, `gemma3:4b`, `prometheus2`) ist in `shared/models.json` definiert.

## Struktur & Ausführung

Die Testszenarien sind in Phasen unterteilt:

- `ts01` bis `ts04`: Modellauswahl (Phase B)

- `ts05` bis `ts07`: Technologie (Phase C)

- `ts08` bis `ts12`: Orchestrierung (Phase D)

- `ts13` bis `ts14`: Betrieb (Phase E)

**Beispielaufruf für ein Szenario:**

```powershell
cd ts02
python ts02_primary_actor.py --limit 3 --repetitions 1
```





*Nützliche Parameter:* `--limit 3` (Schnelltest mit den ersten 3 Fragen), `--repetitions N` (Anzahl der Durchläufe), `--resume` (setzt unterbrochene Läufe fort).

## Wichtige Hinweise zur Methodik (für die Thesis)

- **Hardware-Randbedingung:** Alle Tests wurden auf einem Laptop mit 32 GB RAM und **ohne dedizierte GPU** (reine CPU-Inferenz) durchgeführt. vLLM (GPU-only) wurde daher ausgeschlossen.

- **Speichermessung:** Da kein VRAM vorhanden ist, wurde der Arbeitsspeicher (RAM) via `psutil` als Proxy gemessen. Messfehler bei Ollama-Prozessen wurden dokumentiert und korrigiert.

- **Bewertung:** Die Qualitätsbewertung (Antwortqualität, Deutsch, Faithfulness) erfolgte per **LLM-as-a-Judge** (`llama3.1:8b`, Temperatur 0.0). Eine manuelle Stichprobenprüfung ist als Limitation in der Arbeit dokumentiert.

- **Quantisierung:** Sofern nicht anders angegeben, wurde das Q4_K_M-Modell verwendet (bester Kompromiss aus Speed und Qualität, siehe TS-06).
