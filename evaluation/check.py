import sys
sys.path.insert(0, 'shared')
from llm_client import query_model

fragen = [
    "Wie viele Bundesländer hat Österreich?",
    "Wer schrieb 'Der Process'?",
    "Was ist die Hauptstadt von Frankreich?",
    "Wie viele Kontinente gibt es?",
    "Welches Element hat die Ordnungszahl 26?",
]

for m in ['qwen3', 'mistral_small_4']:
    print(f"\n=== {m} ===")
    for f in fragen:
        r = query_model(m, f)
        print(f"  {r['response_text'][:100]}")