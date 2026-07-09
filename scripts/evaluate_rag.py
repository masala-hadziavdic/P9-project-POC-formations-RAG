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
# DATASET
# ==========================================================

with open(ANNOTATED_DATASET, "r", encoding="utf-8") as f:
    annotated_data = json.load(f)


# ==========================================================
# API
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
# EVALUATION
# ==========================================================

theme_scores = []

location_scores = []

semantic_scores = []

response_times = []

returned_docs = []

print("=" * 70)
print("RAG EVALUATION")
print("=" * 70)

for sample in annotated_data:

    question = sample["question"]

    expected_answer = sample["expected_answer"]

    expected_themes = [
        t.lower()
        for t in sample["expected_themes"]
    ]

    expected_locations = [
        l.lower()
        for l in sample["expected_locations"]
    ]

    print(f"\nQuestion : {question}")

    result, elapsed = call_api(question)

    response_times.append(elapsed)

    returned_docs.append(result["n_results"])

    titles = " ".join(
        r["title"]
        for r in result["results"]
    )

    chunks = " ".join(
        r["chunk"]
        for r in result["results"]
    )

    cities = [
        r["city"].lower()
        for r in result["results"]
    ]

    full_text = (titles + " " + chunks).lower()

    # ======================================================
    # THEMES
    # ======================================================

    if expected_themes:

        found = sum(
            theme in full_text
            for theme in expected_themes
        )

        theme_score = found / len(expected_themes)

    else:

        theme_score = 1.0

    theme_scores.append(theme_score)

    # ======================================================
    # LOCATIONS
    # ======================================================

    if expected_locations:

        found = sum(
            city in cities
            for city in expected_locations
        )

        location_score = found / len(expected_locations)

    else:

        location_score = 1.0

    location_scores.append(location_score)

    # ======================================================
    # SEMANTIC SIMILARITY
    # ======================================================

    emb_expected = model.encode(
        [expected_answer],
        normalize_embeddings=True
    )

    emb_chunk = model.encode(
        [chunks],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        emb_expected,
        emb_chunk
    )[0][0]

    semantic_scores.append(similarity)

    print(f"Themes      : {theme_score:.2f}")

    print(f"Locations   : {location_score:.2f}")

    print(f"Semantic    : {similarity:.3f}")

    print(f"Results     : {result['n_results']}")

    print(f"Time        : {elapsed:.2f} sec")


# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 70)

print("FINAL RESULTS")

print("=" * 70)

print(f"Questions évaluées                : {len(annotated_data)}")

print(f"Couverture moyenne des thèmes     : {100*np.mean(theme_scores):.1f} %")

print(f"Exactitude des localisations      : {100*np.mean(location_scores):.1f} %")

print(f"Similarité sémantique moyenne     : {np.mean(semantic_scores):.3f}")

print(f"Nombre moyen de résultats         : {np.mean(returned_docs):.2f}")

print(f"Temps moyen de réponse            : {np.mean(response_times):.2f} sec")

print("=" * 70)