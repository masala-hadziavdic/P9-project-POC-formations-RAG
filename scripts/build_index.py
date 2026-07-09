import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


def build_index():

    try:
        print("Loading model...")
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        data_dir = Path("data")

        chunks_path = data_dir / "chunks.pkl"
        index_path = data_dir / "index.faiss"

        # -------------------------
        # Vérification fichiers
        # -------------------------
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"{chunks_path} not found"
            )

        # -------------------------
        # Chargement des chunks
        # -------------------------
        print("Loading chunks...")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        if len(chunks) == 0:
            raise ValueError(
                "chunks.pkl is empty"
            )

        print(f"Documents loaded: {len(chunks)}")

        # -------------------------
        # Création embeddings
        # -------------------------
        print("Creating embeddings...")

        embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        print(
            f"Embeddings shape: {embeddings.shape}"
        )

        # -------------------------
        # Construction index FAISS
        # -------------------------
        print("Building FAISS index...")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(embeddings)

        # -------------------------
        # Sauvegarde
        # -------------------------
        faiss.write_index(
            index,
            str(index_path)
        )

        print("✅ Index rebuilt successfully")
        print(f"Documents indexed: {index.ntotal}")

        return True

    except Exception as e:

        print("❌ Build failed")
        print(f"Error: {e}")

        raise e


if __name__ == "__main__":
    build_index()