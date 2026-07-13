from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import pickle
import subprocess

from rag.retrieval import retrieve
from rag.generation import generate_answer

import rag.retrieval

print("Retrieval file:", rag.retrieval.__file__)
print("Retrieve function:", retrieve)

app = FastAPI(
    title="Formation RAG API",
    description="API REST pour rechercher des formations",
    version="1.0"
)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------

print("Loading FAISS index...")
index = faiss.read_index("data/index.faiss")

print("Loading chunks...")
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Loading metadata...")
with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("API ready")


# ---------------------------------------------------
# Request model
# ---------------------------------------------------

class QueryRequest(BaseModel):
    question: str


# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "RAG API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "index_size": index.ntotal
    }


# ---------------------------------------------------
# Retrieval only
# ---------------------------------------------------

@app.post("/ask")
def ask(req: QueryRequest):

    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question vide"
        )

    results = retrieve(
        question,
        index,
        chunks,
        metadata,
        k=5
    )

    formatted_results = []

    for c in results:

        formatted_results.append({

            "title": c["title"],

            "city": c["city"],

            "score": round(c["score"], 3),

            "rerank": round(c["rerank"], 3),

            "confidence": c["confidence"],

            "chunk": (
                c["chunk"][:400] + "..."
                if len(c["chunk"]) > 400
                else c["chunk"]
            )

        })

    return {

        "question": question,

        "n_results": len(formatted_results),

        "results": formatted_results

    }


# ---------------------------------------------------
# Retrieval + TinyLlama
# ---------------------------------------------------

@app.post("/ask_llm")
def ask_llm(req: QueryRequest):

    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question vide"
        )

    results = retrieve(
        question,
        index,
        chunks,
        metadata,
        k=5
    )

    context = "\n\n".join(
        c["chunk"]
        for c in results
    )

    generated_answer = generate_answer(
        question,
        context
    )

    formatted_results = []

    for c in results:

        formatted_results.append({

            "title": c["title"],

            "city": c["city"],

            "score": round(c["score"], 3),

            "rerank": round(c["rerank"], 3),

            "confidence": c["confidence"],

            "chunk": (
                c["chunk"][:400] + "..."
                if len(c["chunk"]) > 400
                else c["chunk"]
            )

        })

    return {

        "question": question,

        "generated_answer": generated_answer,

        "retrieved_context": context,

        "n_results": len(formatted_results),

        "results": formatted_results

    }


# ---------------------------------------------------
# Rebuild index
# ---------------------------------------------------

@app.post("/rebuild")
def rebuild():

    result = subprocess.run(
        ["python", "scripts/build_index.py"]
    )

    if result.returncode != 0:

        raise HTTPException(
            status_code=500,
            detail="Index rebuild failed"
        )

    global index, chunks, metadata

    index = faiss.read_index("data/index.faiss")

    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("data/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return {

        "status": "success",

        "documents": index.ntotal

    }