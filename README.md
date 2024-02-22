# bm25-pt

A minimal BM25 implementation using PyTorch. (Also uses [HuggingFace tokenizers](https://huggingface.co/docs/tokenizers/en/index) behind the scenes to tokenize text.)

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

then proceed to use the library as normal.ss    