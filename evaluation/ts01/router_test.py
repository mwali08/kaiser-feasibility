import requests
import json
import time
import numpy as np
import pandas as pd

# --- Konfiguration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["qwen3:4b", "ministral-3:3b"]
TESTSET_FILE = "testset.json"
TEMPERATURE = 0.0

# === KATEGORIEN UND ACTOREN HIER DEFNIEREN ===
# Passe das an deine echte Taxonomie an!
CATEGORIES = ["Wetter", "Smalltalk", "Finanzen", "Technik", "Sonstiges"]
ACTORS = ["Primary", "Primary+Secondary"]

PROMPT_TEMPLATE = """Du bist ein Klassifikations-Router. Analysiere die Anfrage und ordne sie zu.

ERLAUBTE KATEGORIEN:
- "Wetter": Fragen zu Wetter, Klima, Vorhersagen.
- "Smalltalk": Begrüßungen, Witze, Gedichte, persönliche Fragen, Meinungen.
- "Finanzen": Berechnungen, Geld, Umsatz, Buchhaltung, Investitionen.
- "Technik": Programmierung, IT, Naturwissenschaften, Erklärungen technischer Konzepte.
- "None": Die Anfrage gehört in KEINE der obigen Kategorien (z.B. Medizin, Recht, Handwerk, Reisen, Kochen).

ERLAUBTE ACTORS:
- "Primary": EINE einzige Domäne reicht aus, um die Anfrage zu beantworten.
- "Primary+Secondary": ZWEI Domänen sind beteiligt (z.B. Finanzen UND Recherche, oder Technik UND Pädagogik).

BEISPIELE FÜR ACTOR-ZUORDNUNG:
- "Berechne 15% von 200" -> Primary (nur Finanzen)
- "Erzähl mir einen Witz" -> Primary (nur Smalltalk)
- "Wie wird das Wetter morgen?" -> Primary (nur Wetter)
- "Umsatz von Apple 2024 recherchieren" -> Primary+Secondary (Finanzen + Recherche)
- "Erkläre die Relativitätstheorie didaktisch aufbereitet" -> Primary+Secondary (Technik + Pädagogik)

WENN DIE ANFRAGE IN KEINE KATEGORIE PASST:
Nutze "None" für BEIDE Felder.
Beispiele: "Wie repariere ich mein Auto?", "Bestelle eine Pizza", "Was ist die Hauptstadt von Australien?"

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt:
{{"category": "<Kategorie>", "actor": "<Primary oder Primary+Secondary oder None>"}}

Anfrage: {text}
"""

def query_ollama(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,  # WICHTIG für Qwen3: Thinking-Mode aus!
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 150
        }
    }
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        latency_ms = (time.time() - start_time) * 1000
        return response.json().get("response", ""), latency_ms
    except Exception as e:
        return f"ERROR: {e}", 0

def normalize(s):
    """Normalisiert Strings für robusten Vergleich."""
    return str(s).strip().lower().replace(" ", "").replace("_", "")

def evaluate_model(model_name, testset):
    results = []
    print(f"\nTeste Modell: {model_name}...")
    
    for item in testset:
        prompt = PROMPT_TEMPLATE.format(
            categories=", ".join(CATEGORIES),
            text=item["text"]
        )
        raw_response, latency = query_ollama(model_name, prompt)
        
        parsed_category = "PARSE_ERROR"
        parsed_actor = "PARSE_ERROR"
        is_valid_format = False
        
        try:
            parsed = json.loads(raw_response)
            parsed_category = parsed.get("category", "MISSING")
            parsed_actor = parsed.get("actor", "MISSING")
            if parsed_category != "MISSING" and parsed_actor != "MISSING":
                is_valid_format = True
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Robuster Vergleich (case-insensitive)
        is_correct = (
            normalize(parsed_category) == normalize(item["expected_category"]) and
            normalize(parsed_actor) == normalize(item["expected_actor"])
        )
        
        # OOS-False-Positive: OOS-Anfrage, aber Modell liefert keine "None"
        is_false_positive = item["is_oos"] and normalize(parsed_category) not in ("none", "parseerror", "missing", "")
        
        results.append({
            "model": model_name, "id": item["id"], "text": item["text"],
            "expected_cat": item["expected_category"], "parsed_cat": parsed_category,
            "expected_actor": item["expected_actor"], "parsed_actor": parsed_actor,
            "raw_response": raw_response[:200],  # Zum Debuggen
            "correct": is_correct, "valid_format": is_valid_format,
            "is_oos": item["is_oos"], "false_positive": is_false_positive,
            "latency_ms": latency
        })
    return results

def calculate_metrics(results, model_name):
    df = pd.DataFrame(results)
    oos_df = df[df["is_oos"] == True]
    fpr = (oos_df["false_positive"].sum() / len(oos_df)) * 100 if len(oos_df) > 0 else 0.0
    acc = df["correct"].mean() * 100
    
    return {
        "Modell": model_name,
        "Genauigkeit (%)": round(acc, 2),
        "Latenz Median (ms)": round(np.median(df["latency_ms"]), 2),
        "Latenz P95 (ms)": round(np.percentile(df["latency_ms"], 95), 2),
        "Gültiges Format (%)": round(df["valid_format"].mean() * 100, 2),
        "Falsch-Positiv-Rate OOS (%)": round(fpr, 2),
        "KO-Kriterium (>=90%)": "Bestanden" if acc >= 90 else "Nicht bestanden"
    }

if __name__ == "__main__":
    # UTF-8 explizit erzwingen (behebt "ErzÃ¤hl"-Problem)
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        testset = json.load(f)
        
    all_results, summary_metrics = [], []
    
    for model in MODELS:
        res = evaluate_model(model, testset)
        all_results.extend(res)
        summary_metrics.append(calculate_metrics(res, model))
    
    # UTF-8 mit BOM für Excel, damit Umlaute korrekt dargestellt werden
    pd.DataFrame(all_results).to_csv("router_test_details.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summary_metrics)
    
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG DER MESSERGEBNISSE")
    print("="*60)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("router_test_summary.csv", index=False, encoding="utf-8-sig")