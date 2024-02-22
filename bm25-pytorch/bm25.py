from typing import List, Optional

import math
import torch
import transformers


class TokenizedBM25:
    k1: float
    b: float
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._vocab_size = 100_000 # TODO set
        self.corpus_size = 0
        self.avgdl = 0

        self._num_documents_containing_word = torch.zeros((self._vocab_size,), dtype=torch.long)
        self._corpus = None

    @property
    def _corpus_size(self) -> int:
        return len(self._corpus) if (self._corpus is not None) else -1
    
    def index(self, corpus: torch.Tensor) -> None:
        self._corpus = corpus.to_sparse()

    def num_documents_containing(self, word: int) -> int:
        return self._num_documents_containing_word[word]

    def IDF(self, word: int) -> float:
        num = (self.corpus_size - self.num_documents_containing(word) + 0.5)
        den = self.num_documents_containing(word) - 0.5
        return math.log(num / den + 1)

    def _score_pair_slow(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        this_document_length = (document != 0).int().sum()
        score = 0
        for word in query:
            occurrences = (document == word).int().sum()
            num = occurrences * (self.k1 + 1)
            den = occurrences + (self.k1 * 
                                    (1 - self.b + self.b * (this_document_length / self.average_document_length)))
            score += self.IDF(word) * num / den
        return score

    def score_slow(self, query: torch.Tensor) -> torch.Tensor:
        scores = [self._score_pair_slow(query, document) for document in self._corpus]
        return torch.Tensor(scores)
    
    def score(self, query: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score_batch(self, queries: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BM25(TokenizedBM25):
    k1: float
    b: float
    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer: Optional[transformers.PreTrainedTokenizer], k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(self, k1=k1, b=b)
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
    
    def index(self, corpus: List[str]) -> None:
        corpus_ids = self.tokenizer(corpus, return_tensors='pt')
        return super().index(corpus=corpus_ids)

    def score_slow(self, query: str) -> torch.Tensor:
        query_ids = self.tokenizer(query, return_tensors='pt')
        return super().score_slow(query=query_ids)
    
    def score(self, query: torch.Tensor) -> torch.Tensor:
        query_ids = self.tokenizer(query, return_tensors='pt')
        return super().score(query=query_ids)

    def score_batch(self, queries: torch.Tensor) -> torch.Tensor:
        queries_ids = self.tokenizer(queries, return_tensors='pt')
        return super().score(query=queries_ids)