from typing import List

import pickle
import pytest
import os
import torch

from bm25_pt import BM25


current_folder = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def scrolls_corpus() -> List[str]:
    file_path = os.path.join(current_folder, "scrolls_test_corpus.p")
    return pickle.load(open(file_path, "rb"))

@pytest.fixture
def scrolls_queries() -> List[str]:
    file_path = os.path.join(current_folder, "scrolls_test_queries.p")
    return pickle.load(open(file_path, "rb"))

def test_scrolls(scrolls_corpus, scrolls_queries):
    scrolls_corpus = scrolls_corpus[:3]
    bm25 = BM25()
    print("indexing")
    bm25.index(scrolls_corpus)
    nc = 4
    print("scoring (slow)")
    slow_scores_list = [bm25.score_slow(scrolls_queries[i]) for i in range(nc)]
    doc_scores_slow = torch.stack(slow_scores_list)
    print("scoring (fast)")
    doc_scores = bm25.score_batch(scrolls_queries[:nc])

    torch.testing.assert_close(doc_scores, doc_scores_slow)