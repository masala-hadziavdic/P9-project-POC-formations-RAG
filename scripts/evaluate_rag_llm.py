import json
import os
import re

import pandas as pd
import requests

from dotenv import load_dotenv
from mistralai.client import Mistral


# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = "http://127.0.0.1:8000/ask_llm"

ANNOTATED_DATASET = "data/annotated_dataset.json"

CSV_OUTPUT = "evaluation_results.csv"

JSON_OUTPUT = "evaluation_results.json"


# ==========================================================
# API KEY
# ==========================================================

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if api_key is None:
    raise ValueError("MISTRAL_API_KEY not found")


client = Mistral(api_key=api_key)


# ==========================================================
# LOAD DATASET
# ==========================================================

with open(ANNOTATED_DATASET, "r", encoding="utf-8") as f:
    annotated_dataset = json.load(f)


# ==========================================================
# CALL RAG API
# ==========================================================

def call_rag(question):

    response = requests.post(
        API_URL,
        json={
            "question": question
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()



# ==========================================================
# CLEAN MISTRAL JSON RESPONSE
# ==========================================================

def clean_json_response(text):

    # Remove markdown blocks
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    return text.strip()



# ==========================================================
# ASK MISTRAL EVALUATOR
# ==========================================================

def evaluate_with_mistral(
    question,
    expected,
    context,
    answer,
):

    prompt = f"""
You are an expert RAG evaluator.

Evaluate a Retrieval-Augmented Generation (RAG) system.

Question:
{question}

Retrieved context:
{context}

Expected answer:
{expected}

Generated answer:
{answer}

Evaluate the generated answer.

Give four scores from 0 to 10.

Faithfulness:
Does the generated answer contain only information present in the retrieved context?

Relevance:
Does the generated answer answer the user's question?

Completeness:
Does it include all important information from the retrieved context?

Correctness:
Is the generated answer factually correct according to the retrieved context?

Return ONLY valid JSON.

{{
    "faithfulness": 0,
    "relevance": 0,
    "completeness": 0,
    "correctness": 0,
    "comment": "short explanation"
}}
"""


    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
    )


    return response.choices[0].message.content



# ==========================================================
# MAIN LOOP
# ==========================================================

results = []


print("=" * 70)
print("Evaluating...")
print("=" * 70)


for sample in annotated_dataset[:17]:


    question = sample["question"]

    expected = sample["expected_answer"]


    rag = call_rag(question)


    answer = rag.get("generated_answer", "")

    context = rag.get("retrieved_context", "")


    print(question)


    judgement = evaluate_with_mistral(
    question,
    expected,
    context,
    answer,
)


    judgement = clean_json_response(judgement)


    try:

        scores = json.loads(judgement)


    except Exception as e:

        print("JSON parsing error:")
        print(judgement)

        scores = {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "correctness": 0,
            "comment": f"Invalid JSON from Mistral: {e}"
        }



    # ensure all keys exist

    scores.setdefault("faithfulness", 0)

    scores.setdefault("relevance", 0)

    scores.setdefault("completeness", 0)

    scores.setdefault("correctness", 0)

    scores.setdefault("comment", "")


    scores["question"] = question

    scores["expected"] = expected

    scores["answer"] = answer

    scores["context"] = context


    results.append(scores)



# ==========================================================
# SAVE RESULTS
# ==========================================================


df = pd.DataFrame(results)


df.to_csv(
    CSV_OUTPUT,
    index=False,
    encoding="utf-8"
)



with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )



# ==========================================================
# SUMMARY
# ==========================================================

print()

print(df)


print()

print("Average scores:")

print(
    df[
        [
            "faithfulness",
            "relevance",
            "completeness",
            "correctness"
        ]
    ].mean()
)


print()

print("Saved")

print(CSV_OUTPUT)

print(JSON_OUTPUT)