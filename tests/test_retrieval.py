import faiss
import pickle

from rag.retrieval import retrieve


def test_retrieve():

    index = faiss.read_index("data/index.faiss")

    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("data/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    docs = retrieve(
        "informatique",
        index,
        chunks,
        metadata
    )

    assert len(docs) > 0