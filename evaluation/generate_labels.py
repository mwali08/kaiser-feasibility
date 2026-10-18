import json

# 1. Lade TS-02 Antworten
responses = []
with open('ts02/responses_ts02.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        responses.append(json.loads(line))

# 2. Lade fehlerhafte Antworten
with open('ts04/flawed_answers.json', 'r', encoding='utf-8') as f:
    flawed = json.load(f)['answers']

labels = {"_hinweis": "Format: {id: {score, is_hallucination}}"}

# 3. Labels für TS-02 Antworten (Dictionary!)
for r in responses:
    key = f"{r['question_id']}_{r['model']}"
    labels[key] = {
        "score": 5,
        "is_hallucination": False
    }

# 4. Labels für fehlerhafte Antworten
for f_ans in flawed:
    labels[f_ans['id']] = {
        "score": f_ans['ref_score'],
        "is_hallucination": (f_ans['error_type'] == 'hallucination')
    }

# 5. Speichern
with open('ts04/reference_labels.json', 'w', encoding='utf-8') as f:
    json.dump(labels, f, indent=2, ensure_ascii=False)

print(f"✅ {len(labels)-1} Reference Labels als Dictionary generiert")