import pytest
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


def test_scores_equal_batch():
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

    queries = ["weights", "ratio logarithm", "term filter common"]

    doc_scores_slow = bm25.score_batch(queries)
    doc_scores = bm25.score_batch(queries)
    doc_scores_unbatched = bm25.score_batch(queries, batch_size=1)

    assert doc_scores[0].argmax() == 2
    assert doc_scores[1].argmax() == 5
    
    torch.testing.assert_close(doc_scores, doc_scores_slow)
    torch.testing.assert_close(doc_scores, doc_scores_unbatched)

    # check term counts
    count_of_the =  " ".join(corpus + [""]).count("the ")
    token_of_the = bm25.tokenizer.encode("the", add_special_tokens=False, return_tensors='pt').item()
    assert count_of_the == bm25._corpus.sum(0)[1996]


def test_scores_equal_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    bm25_cpu = BM25(device='cpu') # This is the default
    bm25_gpu = BM25(device='cuda')
    corpus = [
        "Now imagine you’re a software engineer building a RAG system for your company.",
        "You decide to store your vectors in a vector database. You notice that in a",
        "vector database, what's stored are embedding vectors, not the text data itself.",
        "The database fills up with rows and rows of random-seeming numbers",
        "that represent text data but never ‘sees’ any text data at all.",

    ]
    bm25_cpu.index(corpus)
    bm25_gpu.index(corpus)

    queries = ["vectors text database", "see text data", "software rows all at"]

    doc_scores_cpu = bm25_cpu.score_batch(queries)
    doc_scores_gpu = bm25_gpu.score_batch(queries)
    
    torch.testing.assert_close(doc_scores_cpu, doc_scores_gpu.cpu())
