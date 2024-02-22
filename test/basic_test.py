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


def test_scores_equal():
    bm25 = BM25()
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
        "wind is not the worst weather we could imagine",
    ]
    bm25.index(corpus)

    query = "windy London"

    doc_scores_slow = bm25.score(query)
    doc_scores = bm25.score(query)
    
    torch.testing.assert_close(doc_scores, doc_scores_slow)


def test_scores_equal_2():
    bm25 = BM25()
    corpus = [
        "A high weight in tf–idf is reached by a high term frequency",
        "(in the given document) and a low document frequency of the term",
        "in the whole collection of documents; the weights hence tend to filter",
        "out common terms. Since the ratio inside the idf's log function is always",
        "greater than or equal to 1, the value of idf (and tf–idf) is greater than or equal",
        "to 0. As a term appears in more documents, the ratio inside the logarithm approaches",
        "1, bringing the idf and tf–idf closer to 0.",
    ]
    bm25.index(corpus)

    queries = ["weights", "ratio logarithm"]

    doc_scores_slow = bm25.score_batch(queries)
    doc_scores = bm25.score_batch(queries)

    assert doc_scores[0].argmax() == 2
    assert doc_scores[1].argmax() == 5
    
    torch.testing.assert_close(doc_scores, doc_scores_slow)

    # check term counts
    count_of_the =  " ".join(corpus + [""]).count("the ")
    token_of_the = bm25.tokenizer.encode("the", add_special_tokens=False, return_tensors='pt').item()
    assert count_of_the == bm25._corpus.sum(0)[1996]
