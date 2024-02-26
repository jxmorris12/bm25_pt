from typing import List, Optional

import functools
import math
import torch
import transformers


def documents_to_bags(docs: torch.Tensor, vocab_size: int) -> torch.sparse.Tensor:
    num_docs, seq_length = docs.shape
    batch_idxs = torch.arange(num_docs)[:, None].expand(-1, seq_length)
    idxs = torch.stack((
        batch_idxs, docs
    ))
    idxs = idxs.reshape((2, -1))
    vals = (docs > 0).int().flatten()
    return torch.sparse_coo_tensor(idxs, vals, size=(num_docs, vocab_size)).coalesce()


class TokenizedBM25:
    k1: float
    b: float
    vocab_size: int
    ########################################################
    _documents: Optional[List[str]]
    _corpus: Optional[torch.sparse.Tensor]
    _corpus_lengths: Optional[torch.Tensor]
    _average_document_length: Optional[float]
    def __init__(self, k1: float = 1.5, b: float = 0.75, vocab_size: int = 100_000):
        self.k1 = k1
        self.b = b
        self.vocab_size = vocab_size

        self._documents_containing_word = None
        self._word_counts = None
        self._documents = None
        self._corpus = None
        self._corpus_lengths = None
        self._average_document_length = None

    @property
    def _corpus_size(self) -> int:
        return len(self._corpus) if (self._corpus is not None) else -1

    def docs_to_bags(self, documents: torch.Tensor) -> torch.sparse.Tensor:
        return documents_to_bags(documents, vocab_size=self.vocab_size)

    def index(self, documents: torch.Tensor) -> None:
        self._documents = documents
        self._corpus_lengths = (documents != 0).sum(1).float()
        self._average_document_length = self._corpus_lengths.mean()
        self._corpus = self.docs_to_bags(documents=documents)
        self._word_counts = self._corpus.sum(dim=0).to_dense()
        self._documents_containing_word = torch.zeros(self.vocab_size, dtype=torch.long)
        token_ids, token_counts = self._corpus.coalesce().indices()[1].unique(return_counts=True)
        self._documents_containing_word = self._documents_containing_word.scatter_add(0, token_ids, token_counts)
        
        idf_num = (self._corpus_size - self._documents_containing_word + 0.5)
        idf_den = (self._documents_containing_word + 0.5)
        self._IDF = (idf_num / idf_den + 1).log()

    @functools.cache
    def compute_IDF(self, word: int) -> float:
        num = (self._corpus_size - self._documents_containing_word[word] + 0.5)
        den = (self._documents_containing_word[word] + 0.5)
        return math.log(num / den + 1)

    def _score_pair_slow(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        this_document_length = (document != 0).int().sum()
        score = 0
        for word in query:
            occurrences = (document == word).int().sum()
            num = occurrences * (self.k1 + 1)
            den = occurrences + (self.k1 * 
                                    (1 - self.b + self.b * (this_document_length / self._average_document_length)))
            word_score = self.compute_IDF(word) * num / den
            score += word_score
        return score

    def score_slow(self, query: torch.Tensor) -> torch.Tensor:
        scores = [self._score_pair_slow(query, document) for document in self._documents]
        return torch.Tensor(scores)
    
    def score(self, query: torch.Tensor) -> torch.Tensor:
        return self.score_batch(query[None]).flatten()

    def score_batch(self, queries: torch.Tensor) -> torch.Tensor:
        # TODO: Batch idf computation, this shouldn't be too slow though since it's cached.
        num_queries, seq_length = queries.shape
        query_idxs = torch.arange(num_queries)[:, None].expand(-1, seq_length)
        idxs = torch.stack((
            query_idxs, queries
        ))
        idxs = idxs.reshape((2, -1))
        idf_vals = [[self.compute_IDF(w) for w in q] for q in queries.tolist()]
        idf_vals = torch.tensor(idf_vals, dtype=torch.float).flatten()
        queries_idf = torch.sparse_coo_tensor(idxs, idf_vals, size=(num_queries, self.vocab_size)).coalesce()

        occurrences = (queries_idf.float() @ self._corpus.float().T).to_dense()
        scores_n = (occurrences * (self.k1 + 1))
        scores_d = (occurrences + (self.k1 * 
                                  (1 - self.b + self.b * (self._corpus_lengths / self._average_document_length))))
        
        return scores_n / scores_d


class BM25(TokenizedBM25):
    k1: float
    b: float
    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer: Optional[transformers.PreTrainedTokenizer] = None, k1: float = 1.5, b: float = 0.75) -> None:
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        # TODO aggregate beyond subword level here...
        self.tokenizer = tokenizer
        tokenizer_fn = functools.partial(
            tokenizer,
            return_tensors='pt',
            truncation=False,
            padding=True,
            add_special_tokens=False,
        )
        self.tokenizer_fn = lambda s: tokenizer_fn(s).input_ids
        super().__init__(k1=k1, b=b, vocab_size=tokenizer.vocab_size)
    
    def index(self, documents: List[str]) -> None:
        documents_ids = self.tokenizer_fn(documents)
        return super().index(documents=documents_ids)

    def score_slow(self, query: str) -> torch.Tensor:
        query_ids = self.tokenizer_fn(query).flatten()
        return super().score_slow(query=query_ids)
    
    def score(self, query: str) -> torch.Tensor:
        return self.score_batch(queries=[query]).flatten()

    def score_batch(self, queries: List[str]) -> torch.Tensor:
        queries_ids = self.tokenizer_fn(queries)
        return super().score_batch(queries=queries_ids)

    def text_to_bags(self, documents: torch.Tensor) -> torch.sparse.Tensor:
        document_ids = self.tokenizer_fn(documents)
        return self.documents_to_bags(document_ids, vocab_size=self.vocab_size)