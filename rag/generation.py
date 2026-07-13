from transformers import pipeline


print("Loading TinyLlama...")


llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)


print("✅ TinyLlama loaded")


def generate_answer(question, context):

    prompt = f"""
Tu es un assistant spécialisé dans les formations.

Utilise uniquement le contexte suivant :

{context}


Question:
{question}


Réponse:
"""


    result = llm(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        return_full_text=False
    )


    return result[0]["generated_text"].strip()