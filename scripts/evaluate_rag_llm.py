import json
import os
import traceback
from dotenv import load_dotenv

import pandas as pd
import requests
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_mistralai import ChatMistralAI


# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = "http://127.0.0.1:8000/ask"

ANNOTATED_DATASET = "data/annotated_dataset.json"

JSON_OUTPUT = "evaluation_results.json"

CSV_OUTPUT = "evaluation_results.csv"


# ==========================================================
# API KEY
# ==========================================================
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found")


# ==========================================================
# LLM
# ==========================================================

llm = ChatMistralAI(
    api_key=api_key,
    model="mistral-small-latest",
    temperature=0,
)


# ==========================================================
# DATASET
# ==========================================================

with open(ANNOTATED_DATASET, "r", encoding="utf-8") as f:
    annotated_dataset = json.load(f)


# ==========================================================
# API
# ==========================================================

def call_api(question):

    response = requests.post(
        API_URL,
        json={
            "question": question,
            "k": 5,
        },
    )

    response.raise_for_status()

    return response.json()


# ==========================================================
# BUILD DATASET
# ==========================================================

questions = []
answers = []
contexts = []
ground_truths = []

print("=" * 70)
print("Building evaluation dataset...")
print("=" * 70)

for sample in annotated_dataset:

    print(sample["question"])

    result = call_api(sample["question"])

    answer = "\n".join(
        r["title"]
        for r in result["results"]
    )

    context = [
        r["chunk"]
        for r in result["results"]
    ]

    questions.append(sample["question"])
    answers.append(answer)
    contexts.append(context)
    ground_truths.append(sample["expected_answer"])


dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)


# ==========================================================
# METRICS
# ==========================================================

metrics = [

    faithfulness,

    answer_relevancy,

    context_precision,

    context_recall,

]


# ==========================================================
# EVALUATION
# ==========================================================

print()
print("=" * 70)
print("Running RAGAS evaluation...")
print("=" * 70)

try:

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
    )

    df = results.to_pandas()

except Exception:

    traceback.print_exc()

    raise


# ==========================================================
# SAVE CSV
# ==========================================================

df.to_csv(
    CSV_OUTPUT,
    index=False,
    encoding="utf-8"
)


# ==========================================================
# SAVE JSON
# ==========================================================

records = df.to_dict(orient="records")

with open(JSON_OUTPUT, "w", encoding="utf-8") as f:

    json.dump(
        records,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 70)
print("RAGAS SUMMARY")
print("=" * 70)

print(df)

print()

numeric = df.select_dtypes(include="number")

print("Average scores")
print("----------------")

print(numeric.mean())

print()

print("Minimum scores")
print("----------------")

print(numeric.min())

print()

print("Maximum scores")
print("----------------")

print(numeric.max())

print()

print("=" * 70)
print("Files generated")
print("=" * 70)

print(JSON_OUTPUT)
print(CSV_OUTPUT)