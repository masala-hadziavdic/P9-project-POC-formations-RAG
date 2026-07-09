from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss

# -------------------------
# MODELS
# -------------------------
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# -------------------------
# MOTS-CLÉS MÉTIERS
# -------------------------
BTP_KEYWORDS = [
    "btp", "bâtiment", "batiment", "construction",
    "électricien", "electricien", "maçon", "macon",
    "plombier", "travaux publics",
    "technicien d'études", "technicien detudes",
    "gros œuvre", "gros oeuvre"
]

INFO_KEYWORDS = [
    "informatique", "numérique", "numerique",
    "développeur", "developpeur",
    "cybersécurité", "cybersecurite",
    "logiciel", "help desk",
    "développeur web", "developpeur web",
    "data", "python", "réseau", "reseau"
]

# -------------------------
# RETRIEVE
# -------------------------
def retrieve(query, index, chunks, metadata, k=5):

    print("🔥 RETRIEVE FINAL CHARGÉ")

    query_lower = query.lower()

    # -------------------------
    # Embedding question
    # -------------------------
    q = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    # -------------------------
    # FAISS SEARCH
    # -------------------------
    n_candidates = min(
        index.ntotal,
        max(1000, k * 100)
    )

    D, I = index.search(q, n_candidates)

    candidates = []

    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue

        meta = metadata[idx]

        candidates.append({
            "score": float(D[0][rank]),
            "title": meta.get("title", ""),
            "city": meta.get("city", ""),
            "chunk": chunks[idx]
        })

    if not candidates:
        return []

    # -------------------------
    # Détection ville
    # -------------------------
    query_city = None

    if "paris" in query_lower:
        query_city = "paris"
    elif "marseille" in query_lower:
        query_city = "marseille"

    # -------------------------
    # Filtre métier
    # -------------------------
    is_btp_query = any(kw in query_lower for kw in BTP_KEYWORDS)
    is_info_query = any(kw in query_lower for kw in INFO_KEYWORDS)

    filtered_candidates = []

    for c in candidates:

        text = (c["title"] + " " + c["chunk"]).lower()

        if is_btp_query and not any(kw in text for kw in BTP_KEYWORDS):
            continue

        if is_info_query and not any(kw in text for kw in INFO_KEYWORDS):
            continue

        filtered_candidates.append(c)

    # SAFE fallback
    if filtered_candidates:
        candidates = filtered_candidates

    # -------------------------
    # 🔥 FILTRE VILLE STRICT (AJOUT FINAL)
    # -------------------------
    if query_city:
        city_filtered = [
            c for c in candidates
            if c["city"].strip().lower() == query_city
        ]

        # fallback sécurité (évite liste vide)
        if city_filtered:
            candidates = city_filtered

    # -------------------------
    # BOOSTING
    # -------------------------
    for c in candidates:

        c["boost"] = 0

        city = c["city"].strip().lower()
        text = (c["title"] + " " + c["chunk"]).lower()

        # Ville boost léger (déjà filtré mais garde ranking)
        if query_city:
            if city == query_city:
                c["boost"] += 2
            else:
                c["boost"] -= 1

        # BTP boost
        if is_btp_query:
            if any(kw in text for kw in BTP_KEYWORDS):
                c["boost"] += 8
            else:
                c["boost"] -= 4

        # INFO boost
        if is_info_query:
            if any(kw in text for kw in INFO_KEYWORDS):
                c["boost"] += 8
            else:
                c["boost"] -= 4

    # -------------------------
    # CROSS ENCODER
    # -------------------------
    pairs = [
        (
            query,
            f"Ville : {c['city']}\nTitre : {c['title']}\nDescription : {c['chunk']}"
        )
        for c in candidates
    ]

    scores = reranker.predict(pairs, batch_size=64)

    # -------------------------
    # SCORE FINAL
    # -------------------------
    for i, s in enumerate(scores):

        final_score = float(s) + candidates[i]["boost"]

        candidates[i]["rerank"] = final_score
        candidates[i]["confidence"] = round(
            1 / (1 + np.exp(-final_score)),
            3
        )

    # -------------------------
    # TRI FINAL
    # -------------------------
    candidates.sort(key=lambda x: x["rerank"], reverse=True)

    # -------------------------
    # DÉDUPLICATION
    # -------------------------
    seen = set()
    final_results = []

    for c in candidates:

        key = (c["title"].strip().lower(), c["city"].strip().lower())

        if key in seen:
            continue

        seen.add(key)
        c.pop("boost", None)

        final_results.append(c)

        if len(final_results) >= k:
            break

    return final_results