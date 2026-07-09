import json

def test_dataset():

    with open("data/annotated_dataset.json", encoding="utf-8") as f:

        dataset = json.load(f)

    assert len(dataset) >= 18