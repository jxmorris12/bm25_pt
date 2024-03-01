# bm25-pt

A minimal implementation of [Okapi BM25](https://en.wikipedia.org/wiki/Okapi) using PyTorch. (Also uses [HuggingFace tokenizers](https://huggingface.co/docs/tokenizers/en/index) behind the scenes to tokenize text.)

```bash
pip install bm25_pt
```

## Usage


```python
from bm25_pt import BM25

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
doc_scores = bm25.score_batch(queries)
print(doc_scores)
>> tensor([[0.0000, 0.0000, 1.4238, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.5317, 0.0000, 2.0203, 0.0000]])
```

can also call `score()` with a

### Use your own tokenizer

You can use your own tokenizer if you want. Simply provide your tokenizer to the `BM25` constructor:

```python
from bm25_pt import BM25
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")
bm25 = BM25(tokenizer=tokenizer)
```

### Use a GPU

To perform operations on a GPU, it's easy, just pass a device to initialization:

```python
from bm25_pt import BM25

bm25 = BM25(device='cuda')
```

then proceed to use the library as normal.ss    


### Benchmarking on LoCO

I benchmarked this library on the [Stanford LoCo benchmark](https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval). I'm not using any special tokenizer, so it's just using `bert-base-uncased` and matching based on subword tokens. We could probably get better performance by using a word-based model, perhaps by using a cased model, and certainly by tuning hyperparameters, especially `b` and `k1`.

| Retrieval Encoders  | Seq Len  | Tau Scrolls – Summ. | Tau Scrolls – Gov. | Tau Scrolls QMSUM | QASPER - Title | QASPER - Abstract | Average |
|---------------------|----------|---------------------|--------------------|-------------------|----------------|-------------------|---------|
|    bm25_pt   | infinity |         89.5        |        96.1        |        56.2       |      94.3      |        99.4       |   87.1  |
|  M2-BERT-2048 (80M) |   2048   |         81.8        |        94.7        |        58.5       |      87.3      |        95.5       |   83.6  |
|  M2-BERT-8192 (80M) |   8192   |         94.7        |        96.5        |        64.1       |      86.8      |        97.5       |   85.9  |
| M2-BERT-32768 (80M) |   32768  |         98.6        |        98.5        |        69.5       |      97.4      |        98.7       |   92.5  |


As you can see below, this library is very fast, even on CPU:

| Dataset | Size | Index time (s) | Scoring time (s) |
|---------------------|----------|---------------------|--------------------|
| Tau Scrolls – Summ. | 3673 | 43.2 | 14.5 |
| Tau Scrolls – Gov. | 17,457 | 622.7 | 328.2 |
| Tau Scrolls QMSUM | 272 | 5.5 | 0.2 |
| QASPER - Title | 416 | 6.1 | 0.2 |
| QASPER - Abstract | 416 | 5.9 | 0.2

I think that most of the time is spent tokenizing long documents. It could likely be sped up with a faster tokenizer and/or multiprocessing.
