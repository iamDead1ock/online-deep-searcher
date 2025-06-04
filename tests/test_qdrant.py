import unittest
import numpy as np

from deepsearcher.vector_db import RetrievalResult

from deepsearcher.vector_db import Qdrant

try:
    import qdrant_client

    client_installed = True
except ImportError:
    client_installed = False


@unittest.skipIf(not client_installed, "qdrant-client is not installed")
class TestQdrant(unittest.TestCase):
    def test_qdrant(self):
        d = 8
        collection = "hello_deepsearcher"
        qdrant = Qdrant(location=":memory:")
        qdrant.init_collection(
            dim=d,
            collection=collection,
        )

        rng = np.random.default_rng(seed=19530)

        qdrant.insert_data(
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
                    text="hello qdrant",
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

        top_2 = qdrant.search_data(collection=collection, vector=rng.random((1, d))[0], top_k=2)

        self.assertEqual(len(top_2), 2, "Search should return 2 results")
        for result in top_2:
            self.assertIsInstance(result, RetrievalResult, "Result should be a RetrievalResult")

    def test_clear_collection(self):
        d = 8
        collection = "hello_deepsearcher"
        qdrant = Qdrant(location=":memory:")
        qdrant.init_collection(
            dim=d,
            collection=collection,
        )

        qdrant.clear_db(collection=collection)

        collections = qdrant.list_collections()
        collection_names = [c.collection_name for c in collections]
        self.assertNotIn(collection, collection_names, "Collection should be removed")

    def test_list_collections(self):
        d = 8
        collection = "hello_deepsearcher"
        qdrant = Qdrant(location=":memory:")
        qdrant.init_collection(
            dim=d,
            collection=collection,
        )

        collections = qdrant.list_collections(dim=d)

        self.assertTrue(len(collections) > 0, "Should return at least one collection")
        collection_names = [c.collection_name for c in collections]
        self.assertIn(collection, collection_names, "Created collection should be in the list")


if __name__ == "__main__":
    unittest.main()
