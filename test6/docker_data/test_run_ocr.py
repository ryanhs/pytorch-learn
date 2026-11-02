import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_ocr


class OCRMetricsTests(unittest.TestCase):
    def test_exact_transcription_has_zero_error(self):
        self.assertEqual(run_ocr.character_error_rate("Sugar, salt", "Sugar, salt"), 0)
        self.assertEqual(run_ocr.word_error_rate("Sugar, salt", "Sugar, salt"), 0)

    def test_metrics_ignore_case_and_whitespace(self):
        self.assertEqual(
            run_ocr.character_error_rate("Sugar,  salt", "sugar, salt"), 0
        )
        self.assertEqual(run_ocr.word_error_rate("Sugar salt", "sugar\nsalt"), 0)

    def test_word_error_rate_counts_insertions_and_substitutions(self):
        self.assertEqual(run_ocr.word_error_rate("sugar salt", "sugar malt oil"), 1)

    def test_empty_reference_has_no_error_rate(self):
        self.assertEqual(run_ocr.character_error_rate("", "anything"), "")
        self.assertEqual(run_ocr.word_error_rate("", "anything"), "")


class OCRManifestTests(unittest.TestCase):
    def test_rejects_image_paths_outside_images_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_ocr.resolve_image(Path(directory), "../outside.jpg")

    @patch.object(run_ocr, "run_tesseract", return_value="Sugar salt")
    def test_processes_image_and_scores_only_verified_transcription(self, run_tesseract):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "123.jpg").write_bytes(b"image")
            (images_dir / "456.jpg").write_bytes(b"image")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "barcode,product_name,image_filename,review_status,"
                "verified_transcription,verified_transcription_scope\n"
                "123,Example,123.jpg,verified_by_visual_review,Sugar salt,full_image\n"
                "456,Other,456.jpg,needs_manual_review,\n",
                encoding="utf-8",
            )

            results = run_ocr.process_manifest(manifest, images_dir, "eng", 6)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["character_error_rate"], 0)
            self.assertEqual(results[0]["word_error_rate"], 0)
            self.assertEqual(results[1]["character_error_rate"], "")
            self.assertEqual(results[1]["word_error_rate"], "")
            self.assertEqual(run_tesseract.call_count, 2)

    @patch.object(run_ocr, "run_tesseract", return_value="Sugar, salt and nutrition facts")
    def test_ingredient_only_transcription_is_not_scored_as_full_image(
        self, run_tesseract
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "123.jpg").write_bytes(b"image")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "barcode,product_name,image_filename,review_status,"
                "verified_transcription,verified_transcription_scope\n"
                "123,Example,123.jpg,verified_by_visual_review,"
                "Sugar salt,ingredients\n",
                encoding="utf-8",
            )

            results = run_ocr.process_manifest(manifest, images_dir, "eng", 6)

            self.assertEqual(results[0]["character_error_rate"], "")
            self.assertEqual(results[0]["word_error_rate"], "")


if __name__ == "__main__":
    unittest.main()
