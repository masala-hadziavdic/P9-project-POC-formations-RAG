import json
import time
import requests
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CONFIG
# ==========================================================

API_URL = "http://127.0.0.1:8000/ask"
ANNOTATED_DATASET = "data/annotated_dataset.json"

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ==========================================================
# Charger le dataset
# ==========================================================

with open(ANNOTATED_DATASET, "r", encoding="utf-8") as f:
    annotated_data = json.load(f)


# ==========================================================
# Appel API
# ==========================================================

def call_api(question):

    start = time.perf_counter()

    response = requests.post(
        API_URL,
        json={"question": question}
    )

    elapsed = time.perf_counter() - start

    response.raise_for_status()

    return response.json(), elapsed


# ==========================================================
# Evaluation
# ==========================================================

semantic_scores = []
exact_matches = 0
partial_matches = 0
retrieval_recalls = []
response_times = []

print("=" * 60)
print("RAG EVALUATION")
print("=" * 60)

for sample in annotated_data:

    question = sample["question"]
    expected = sample["expected_answer"]

    print(f"\nQuestion : {question}")

    result, elapsed = call_api(question)

    response_times.append(elapsed)

    titles = [r["title"] for r in result["results"]]
    answer = " ".join(titles)

    # ======================================================
    # Exact Match
    # ======================================================

    if expected.lower() == answer.lower():
        exact_matches += 1

    # ======================================================
    # Partial Match
    # ======================================================

    if expected.lower() in answer.lower():
        partial_matches += 1

    # ======================================================
    # Recall@5
    # ======================================================

    recall = any(
        expected.lower() in title.lower()
        for title in titles
    )

    retrieval_recalls.append(int(recall))

    # ======================================================
    # Similarité sémantique
    # ======================================================

    emb_expected = model.encode([expected])

    emb_answer = model.encode([answer])

    similarity = cosine_similarity(
        emb_expected,
        emb_answer
    )[0][0]

    semantic_scores.append(similarity)

    print(f"Recall@5 : {recall}")
    print(f"Semantic similarity : {similarity:.3f}")
    print(f"Response time : {elapsed:.3f} sec")

# ==========================================================
# Résultats
# ==========================================================

n = len(annotated_data)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Nombre de questions                : {n}")

print(f"Exact Match                        : {exact_matches}/{n} ({100*exact_matches/n:.1f} %)")

print(f"Partial Match                      : {partial_matches}/{n} ({100*partial_matches/n:.1f} %)")

print(f"Recall@5 moyen                     : {100*np.mean(retrieval_recalls):.1f} %")

print(f"Similarité sémantique moyenne      : {np.mean(semantic_scores):.3f}")

print(f"Temps de réponse moyen             : {np.mean(response_times):.3f} sec")

print("=" * 60)