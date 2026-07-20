import unittest

import conftest  # noqa: F401 — sets up env


class TestRetrieverConfig(unittest.TestCase):
    def test_config_retrieval_params(self):
        from config.manager import get_config
        cfg = get_config().retrieval
        self.assertGreater(cfg.top_k, 0)
        self.assertGreaterEqual(cfg.score_threshold, 0)
        self.assertGreaterEqual(cfg.text_k, cfg.top_k)

    def test_search_text_returns_empty_when_no_index(self):
        from retriever.search import search_text
        result = search_text("test query", k=5)
        self.assertEqual(result, [])

    def test_search_image_returns_empty_when_no_index(self):
        from retriever.search import search_image
        result = search_image("test query", k=5)
        self.assertEqual(result, [])

    def test_search_chunk_image_returns_empty_when_no_index(self):
        from retriever.search import search_chunk_image
        result = search_chunk_image("test query", k=5)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
