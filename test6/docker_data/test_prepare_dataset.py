import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_read_barcodes_skips_comments_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "barcodes.txt"
            path.write_text("# sample\n12345678\n\n12345678\n87654321\n", encoding="utf-8")

            self.assertEqual(prepare_dataset.read_barcodes(path), ["12345678", "87654321"])

    @patch.object(prepare_dataset, "get_json")
    def test_collect_product_accepts_requested_language_image(self, get_json):
        get_json.return_value = {
            "status": "success",
            "product": {
                "product_name": "Example",
                "ingredients_text": "Sugar",
                "image_ingredients_url": (
                    "https://images.openfoodfacts.org/ingredients_en.1.400.jpg"
                ),
            },
        }

        product = prepare_dataset.collect_product("12345678", "en", "test/1.0")

        self.assertEqual(product["review_status"], "needs_manual_review")
        self.assertTrue(product["image_url"].endswith("ingredients_en.1.400.jpg"))

    @patch.object(prepare_dataset, "get_json")
    def test_collect_product_rejects_other_language_image(self, get_json):
        get_json.return_value = {
            "status": "success",
            "product": {
                "product_name": "Example",
                "ingredients_text": "Sucre",
                "image_ingredients_url": (
                    "https://images.openfoodfacts.org/ingredients_fr.1.400.jpg"
                ),
            },
        }

        product = prepare_dataset.collect_product("12345678", "en", "test/1.0")

        self.assertEqual(product["review_status"], "no_image_in_requested_language")
        self.assertEqual(product["image_url"], "")

    @patch.object(prepare_dataset, "get_json")
    def test_collect_product_reports_missing_product(self, get_json):
        get_json.return_value = {"status": "failure"}

        self.assertIsNone(
            prepare_dataset.collect_product("12345678", "en", "test/1.0")
        )

    def test_preserve_manual_review_when_image_is_unchanged(self):
        image_url = "https://images.openfoodfacts.org/ingredients_en.1.400.jpg"
        row = {
            "image_url": image_url,
            "review_status": "needs_manual_review",
            "verified_transcription": "",
        }
        previous_row = {
            "image_url": image_url,
            "review_status": "verified_by_visual_review",
            "verified_transcription": "Ingredients: Sugar.",
            "verified_transcription_scope": "ingredients",
        }

        prepare_dataset.preserve_manual_review(row, previous_row)

        self.assertEqual(row["review_status"], "verified_by_visual_review")
        self.assertEqual(row["verified_transcription"], "Ingredients: Sugar.")
        self.assertEqual(row["verified_transcription_scope"], "ingredients")

    def test_does_not_preserve_transcription_for_changed_image(self):
        row = {
            "image_url": "https://images.openfoodfacts.org/ingredients_en.2.400.jpg",
            "review_status": "needs_manual_review",
            "verified_transcription": "",
        }
        previous_row = {
            "image_url": "https://images.openfoodfacts.org/ingredients_en.1.400.jpg",
            "review_status": "verified_by_visual_review",
            "verified_transcription": "Ingredients: Sugar.",
        }

        prepare_dataset.preserve_manual_review(row, previous_row)

        self.assertEqual(row["review_status"], "needs_manual_review")
        self.assertEqual(row["verified_transcription"], "")


if __name__ == "__main__":
    unittest.main()
