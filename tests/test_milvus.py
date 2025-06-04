import unittest
import pprint
import numpy as np
from deepsearcher.vector_db import Milvus, RetrievalResult
from deepsearcher.utils import log

class TestMilvus(unittest.TestCase):
    def test_milvus(self):
        d = 8
        collection = "hello_deepsearcher"
        milvus = Milvus()
        milvus.init_collection(
            dim=d,
            collection=collection,
        )
        rng = np.random.default_rng(seed=19530)
        milvus.insert_data(
            collection=collection,
            chunks=[
                RetrievalResult(
                    embedding=rng.random((1, d))[0],
                    text="hello world",
                    reference="local file: hi.txt",
                    metadata={"a": 1},
                ),
                RetrievalResult(
                    embedding=rng.random((1, d))[0],
                    text="hello milvus",
                    reference="local file: hi.txt",
                    metadata={"a": 1},
                ),
                RetrievalResult(
                    embedding=rng.random((1, d))[0],
                    text="hello deep learning",
                    reference="local file: hi.txt",
                    metadata={"a": 1},
                ),
                RetrievalResult(
                    embedding=rng.random((1, d))[0],
                    text="hello llm",
                    reference="local file: hi.txt",
                    metadata={"a": 1},
                ),
            ],
        )
        top_2 = milvus.search_data(
            collection=collection, vector=rng.random((1, d))[0], top_k=2
        )
        log.info(pprint.pformat(top_2))

    def test_clear_collection(self):
        d = 8
        collection = "hello_deepsearcher"
        milvus = Milvus()
        milvus.init_collection(
            dim=d,
            only_init_client=True,
            collection=collection,
        )
        milvus.clear_db(collection=collection)
        self.assertFalse(milvus.client.has_collection(collection, timeout=5))


if __name__ == "__main__":
    unittest.main()
