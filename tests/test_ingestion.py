import unittest

import conftest  # noqa: F401


class TestIngestionConfig(unittest.TestCase):
    def test_config_chunking_params(self):
        from config.manager import get_config
        cfg = get_config().chunking
        self.assertGreater(cfg.max_tokens, 0)
        self.assertGreaterEqual(cfg.overlap_tokens, 0)

    def test_doc_type_classification(self):
        from core.constants import DocType, DOC_TYPE_SIGNALS
        self.assertIn(DocType.RESUME, DOC_TYPE_SIGNALS)
        self.assertIn("resume", DOC_TYPE_SIGNALS[DocType.RESUME])
        self.assertIn(DocType.EMAIL, DOC_TYPE_SIGNALS)
        self.assertIn("from:", DOC_TYPE_SIGNALS[DocType.EMAIL])

    def test_image_audio_extensions(self):
        from core.constants import IMAGE_EXT, AUDIO_EXT
        self.assertIn(".png", IMAGE_EXT)
        self.assertIn(".jpg", IMAGE_EXT)
        self.assertIn(".mp3", AUDIO_EXT)
        self.assertIn(".wav", AUDIO_EXT)

    def test_classify_uploads(self):
        from frontend.streamlit_app import classify_uploads
        docs, images, audio = classify_uploads(["report.pdf", "photo.jpg", "recording.mp3"])
        self.assertEqual(docs, ["report.pdf"])
        self.assertEqual(images, ["photo.jpg"])
        self.assertEqual(audio, ["recording.mp3"])


if __name__ == "__main__":
    unittest.main()
