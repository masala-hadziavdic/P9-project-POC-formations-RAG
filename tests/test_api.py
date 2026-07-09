import requests

url = "http://127.0.0.1:8000/ask"

questions = [
    "Quelles formations sont proposées à Paris ?",
    "Quelles formations financées existent ?",
    "Je cherche une formation en informatique"
]

for q in questions:

    res = requests.post(url, json={"question": q})

    print("\nQUESTION:", q)
    print("RESULTS:")

    for r in res.json()["results"]:
        print("-", r["title"])