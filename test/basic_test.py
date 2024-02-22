import bm25


def test_tiny():
    bm25 = bm25.BM25()
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]
    bm25.index(corpus)

    query = "windy London"
    tokenized_query = query.split(" ")

    doc_scores = bm25.score_slow(tokenized_query)