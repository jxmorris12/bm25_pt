from typing import Iterable, List, Optional

import functools
import math
import scipy
import torch
import transformers
import tqdm

def documents_to_bags(docs: torch.Tensor, vocab_size: int) -> torch.sparse.Tensor:
    num_docs, seq_length = docs.shape
    batch_idxs = torch.arange(num_docs)[:, None].expand(-1, seq_length)
    idxs = torch.stack((
        batch_idxs, docs
    ))
    idxs = idxs.reshape((2, -1))
    vals = (docs > 0).int().flatten()
    return torch.sparse_coo_tensor(idxs, vals, size=(num_docs, vocab_size)).coalesce()


def torch_sparse_to_scipy(t: torch.sparse.Tensor):
    indices = t.coalesce().indices()
    values = t.coalesce().values()
    size = t.size()
    coo_matrix = scipy.sparse.coo_matrix((values.numpy(), (indices[0].numpy(), indices[1].numpy())), shape=size)
    return coo_matrix.tocsr()


def sparse_divide(A: torch.sparse.Tensor, B: torch.sparse.Tensor) -> torch.sparse.Tensor:
    """Have to do sparse division on CPU in scipy."""
    device = A.device
    A = torch_sparse_to_scipy(A)
    B = torch_sparse_to_scipy(B)
    r = (A / B)
    return torch.sparse_coo_tensor(r.nonzero(), r.data, r.shape, device=device)


class TokenizedBM25:
    k1: float
    b: float
    vocab_size: int
    ########################################################
    _corpus: Optional[torch.sparse.Tensor]
    _corpus_lengths: Optional[torch.Tensor]
    _average_document_length: Optional[float]
    def __init__(self, k1: float = 1.5, b: float = 0.75, vocab_size: int = 100_000, device: str = 'cpu'):
        self.k1 = k1
        self.b = b
        self.vocab_size = vocab_size

        self._documents_containing_word = None
        self._word_counts = None
        self._corpus = None
        self._corpus_lengths = None
        self._average_document_length = None
        self.device = device

    @property
    def _corpus_size(self) -> int:
        return len(self._corpus) if (self._corpus is not None) else -1

    def docs_to_bags(self, documents: torch.Tensor) -> torch.sparse.Tensor:
        return documents_to_bags(documents, vocab_size=self.vocab_size).to(self.device)
    
    def index(self, documents: torch.Tensor) -> None:
        bag_size = 4096
        i = 0
        bags = []
        while i < len(documents):
            bags.append(self.docs_to_bags(documents[i:i+bag_size]))
            i += bag_size
        corpus = torch.cat(bags, dim=0)
        self._index_corpus(corpus)

    def _index_corpus(self, corpus: torch.Tensor) -> None:
        self._corpus = corpus.to(self.device)
        self._corpus_lengths = self._corpus.sum(1).float().to_dense()
        self._average_document_length = self._corpus_lengths.mean()
        self._word_counts = self._corpus.sum(dim=0).to_dense()
        self._documents_containing_word = torch.zeros(self.vocab_size, dtype=torch.long, device=self.device)
        token_ids, token_counts = self._corpus.coalesce().indices()[1].unique(return_counts=True)
        self._documents_containing_word = self._documents_containing_word.scatter_add(0, token_ids, token_counts).to(self.device)
        idf_num = (self._corpus_size - self._documents_containing_word + 0.5)
        idf_den = (self._documents_containing_word + 0.5)
        self._IDF = (idf_num / idf_den + 1).log()

    @functools.cache
    def compute_IDF(self, word: int) -> float:
        num = (self._corpus_size - self._documents_containing_word[word] + 0.5)
        den = (self._documents_containing_word[word] + 0.5)
        return math.log(num / den + 1)

    def _score_pair_slow(self, query: torch.Tensor, document_bag: torch.sparse.Tensor) -> torch.Tensor:
        this_document_length = document_bag.sum().item()
        score = 0
        for word in query:
            occurrences = (document_bag[word]).int()
            num = occurrences * (self.k1 + 1)
            den = occurrences + (self.k1 * 
                                    (1 - self.b + self.b * (this_document_length / self._average_document_length)))
            word_score = self.compute_IDF(word) * num / den
            print("word:", word, "num:", num, "den:", den, "idf:", self.compute_IDF(word))
            score += word_score
            print("\t total:", score)
        return score

    def score_slow(self, query: torch.Tensor) -> torch.Tensor:
        scores = [self._score_pair_slow(query, document_bag) for document_bag in self._corpus]
        return torch.Tensor(scores)
    
    def score(self, query: torch.Tensor) -> torch.Tensor:
        return self.score_batch(query[None]).flatten()

    def _score_batch(self, queries: torch.Tensor) -> torch.Tensor:
        # TODO: Batch idf computation, this shouldn't be too slow though since it's cached.
        num_queries, seq_length = queries.shape
        queries_bag = self.docs_to_bags(queries)

        num = (self._corpus * (self.k1 + 1))
        normalized_lengths = (self.k1 * (1 - self.b + self.b * (self._corpus_lengths[:, None] / self._average_document_length)))
        den = normalized_lengths.repeat((1, self._corpus.shape[1])) + self._corpus
        score = (self._IDF[None, :] * sparse_divide(num, den)).sum()

        bm25_scores = queries_bag @ scores.T
        
        return bm25_scores

    def score_batch(self, queries: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        i = 0
        scores = []
        batch_size = batch_size or len(queries)
        while i < len(queries):
            scores.append(self._score_batch(queries[i:i+batch_size]))
            i += batch_size
        return torch.cat(scores, dim=0)

class BM25(TokenizedBM25):
    k1: float
    b: float
    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer: Optional[transformers.PreTrainedTokenizer] = None, k1: float = 1.5, b: float = 0.75, device: str = 'cpu') -> None:
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        # TODO aggregate beyond subword level here...
        self.tokenizer = tokenizer
        tokenizer_fn = functools.partial(
            tokenizer,
            truncation=False,
            padding=True,
            add_special_tokens=False,
        )
        self.tokenizer_fn = lambda s: tokenizer_fn(s, return_tensors='pt').input_ids
        super().__init__(k1=k1, b=b, vocab_size=tokenizer.vocab_size, device=device)
    
    def index(self, documents: List[str]) -> None:
        bag_size = 512
        bags = []
        i = 0
        pbar = tqdm.auto.tqdm("tokenizing & bagging", colour="blue", leave=False, total=len(documents))
        while i < len(documents):
            bags.append(self.text_to_bags(documents[i:i+bag_size]))
            pbar.update(bag_size)
            i += bag_size
        corpus = torch.cat(bags, dim=0)
        self._documents = documents
        return super()._index_corpus(corpus=corpus)

    def score_slow(self, query: str) -> torch.Tensor:
        query_ids = self.tokenizer_fn(query).flatten()
        return super().score_slow(query=query_ids)
    
    def score(self, query: str) -> torch.Tensor:
        return self.score_batch(queries=[query]).flatten()

    def score_batch(self, queries: List[str], batch_size: Optional[int] = None) -> torch.Tensor:
        i = 0
        scores = []
        batch_size = batch_size or len(queries)
        while i < len(queries):
            queries_ids = self.tokenizer_fn(queries[i:i+batch_size]) 
            scores.append(super().score_batch(queries=queries_ids, batch_size=batch_size))
            i += batch_size
        return torch.cat(scores, dim=0)

    def text_to_bags(self, documents: torch.Tensor) -> torch.sparse.Tensor:
        document_ids = self.tokenizer_fn(documents)
        return self.docs_to_bags(document_ids)
