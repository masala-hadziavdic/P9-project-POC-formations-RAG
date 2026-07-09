from rag.retrieval import retrieve

def test_retrieve():

    docs = retrieve("informatique")

    assert len(docs) > 0