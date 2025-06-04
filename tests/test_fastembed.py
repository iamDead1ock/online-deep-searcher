import unittest
import importlib

from deepsearcher.embedding import FastEmbedEmbedding

try:
    import fastembed

    fastembed_installed = True
except ImportError:
    fastembed_installed = False


@unittest.skipIf(not fastembed_installed, "fastembed is not installed")
class TestFastEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding = FastEmbedEmbedding()

    def test_embed_query(self):
        """
        Test embedding of a single query text
        """
        query = "Hello, world!"
        embedding = self.embedding.embed_query(query)

        self.assertIsInstance(embedding, list, "Embedding should be a list")
        self.assertTrue(
            all(isinstance(x, float) for x in embedding), "All elements should be floats"
        )

        self.assertEqual(
            len(embedding),
            self.embedding.dimension,
            "Embedding dimension should match model's dimension",
        )

    def test_embed_documents(self):
        """
        Test embedding of multiple document texts
        """
        documents = ["First document text", "Second document text", "Third document text"]
        embeddings = self.embedding.embed_documents(documents)

        self.assertIsInstance(embeddings, list, "Result should be a list of embeddings")
        self.assertEqual(
            len(embeddings), len(documents), "Number of embeddings should match number of documents"
        )

        for embedding in embeddings:
            self.assertIsInstance(embedding, list, "Each embedding should be a list")
            self.assertTrue(
                all(isinstance(x, float) for x in embedding), "All elements should be floats"
            )
            self.assertEqual(
                len(embedding),
                self.embedding.dimension,
                "Embedding dimension should match model's dimension",
            )

    def test_dimension(self):
        """
        Test that dimension is correctly calculated
        """
        dimension = self.embedding.dimension

        self.assertIsInstance(dimension, int, "Dimension should be an integer")
        self.assertGreater(dimension, 0, "Dimension should be a positive number")

    def test_different_models(self):
        """
        Test initializing with different models
        """
        test_models = [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
        ]

        for model in test_models:
            try:
                embedding = FastEmbedEmbedding(model=model)

                query_embedding = embedding.embed_query("Test query")

                self.assertIsInstance(query_embedding, list, f"Embedding failed for model {model}")
                self.assertTrue(
                    all(isinstance(x, float) for x in query_embedding),
                    f"Invalid embedding for model {model}",
                )
            except Exception as e:
                self.fail(f"Failed to initialize or embed with model {model}: {e}")

    def test_lazy_loading(self):
        """
        Verify that the model is lazily loaded
        """
        embedding = FastEmbedEmbedding()

        self.assertIsNone(embedding._embedding_model, "Model should not be loaded until first use")

        embedding.embed_query("Test lazy loading")

        self.assertIsNotNone(
            embedding._embedding_model, "Model should be loaded after first embedding"
        )


if __name__ == "__main__":
    unittest.main()
