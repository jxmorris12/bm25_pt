import torch

from bm25_pt import BM25


def test_tiny_slow():
    bm25 = BM25()
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]
    bm25.index(corpus)

    query = "windy London"
    tokenized_query = query.split(" ")
    doc_scores = bm25.score_slow(tokenized_query)

    print("test_tiny_slow – all_scores:", doc_scores)

    assert doc_scores.shape == (3,)
    assert doc_scores[0] == 0.0
    assert doc_scores[1] > 0.0
    assert doc_scores[2] == 0.0

    assert isinstance(doc_scores, torch.Tensor)


def test_tiny_batch():
    bm25 = BM25()
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]
    bm25.index(corpus)

    query = "windy London"
    tokenized_query = query.split(" ")
    doc_scores = bm25.score(tokenized_query)

    print("test_tiny_batch – all_scores:", doc_scores)

    assert doc_scores.shape == (3,)
    assert doc_scores[0] == 0.0
    assert doc_scores[1] > 0.0
    assert doc_scores[2] == 0.0

    assert isinstance(doc_scores, torch.Tensor)