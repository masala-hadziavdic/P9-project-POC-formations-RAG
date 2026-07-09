import json
import requests

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_utilization

# ==========================================================
# CONFIG
# ==========================================================

API_URL = "http://127.0.0.1:8000/ask"

ANNOTATED_DATASET = "data/annotated_dataset.json"

# ==========================================================
# Charger le dataset annoté
# ==========================================================

with open(ANNOTATED_DATASET, "r", encoding="utf-8") as f:
    annotated_data = json.load(f)


# ==========================================================
# Appel API
# ==========================================================

def call_api(question):

    response = requests.post(

        API_URL,

        json={
            "question": question,
            "k": 5
        }

    )

    response.raise_for_status()

    return response.json()


# ==========================================================
# Construire Dataset RAGAS
# ==========================================================

def build_dataset():

    questions = []

    answers = []

    contexts = []

    ground_truths = []

    for sample in annotated_data:

        print(f"Question : {sample['question']}")

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

    return Dataset.from_dict(

        {

            "question": questions,

            "answer": answers,

            "contexts": contexts,

            "ground_truth": ground_truths

        }

    )


# ==========================================================
# MAIN
# ==========================================================

dataset = build_dataset()

result = evaluate(

    dataset,

    metrics=[

        context_utilization

    ]

)

print("\n==========================")

print("RAGAS RESULTS")

print("==========================")

print(result)