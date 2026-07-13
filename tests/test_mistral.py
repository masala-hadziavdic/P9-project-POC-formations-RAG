from dotenv import load_dotenv
import os

from mistralai.client import Mistral

# Charger le fichier .env
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found")

# Création du client
client = Mistral(api_key=api_key)

print("Connexion à Mistral...")

# Appel au modèle
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": "Réponds uniquement par : Bonjour Amela !"
        }
    ]
)

print("\nRéponse de Mistral :")
print(response.choices[0].message.content)