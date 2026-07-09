from scripts.build_index import build_index

def test_build():

    index = build_index()

    assert index is not None