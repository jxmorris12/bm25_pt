import torch

from bm25_pt import BM25


def test_tiny_score_slow():
    bm25 = BM25()
    corpus = [
        "Hello there good man!",
        "It is quite windy in London London",
        "How is the weather today?"
    ]
    bm25.index(corpus)

    query = "windy London"
    doc_scores = bm25.score_slow(query)

    print("test_tiny_slow – all_scores:", doc_scores)

    assert doc_scores.shape == (3,)
    assert doc_scores[0] == 0.0
    assert doc_scores[1] > 0.0
    assert doc_scores[2] == 0.0

    assert isinstance(doc_scores, torch.Tensor)


def test_tiny_score():
    bm25 = BM25()
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]
    bm25.index(corpus)

    query = "windy London"
    doc_scores = bm25.score(query)

    print("test_tiny_batch – all_scores:", doc_scores)

    assert doc_scores.shape == (3,)
    assert doc_scores[0] == 0.0
    assert doc_scores[1] > 0.0
    assert doc_scores[2] == 0.0

    assert isinstance(doc_scores, torch.Tensor)