# Test 6: Food Label OCR Dataset Pilot

This pilot collects a small number of Open Food Facts ingredient-panel images for manual OCR review. It does not download the full product database, train a model, or treat Open Food Facts ingredient text as ground truth.

## Start with one example

From the repository root:

```bash
cd test6
./run.sh
```

Like the earlier test runners, this builds a lightweight container and opens an interactive shell. From inside the container, prepare the starter sample with:

```bash
python prepare_dataset.py --barcodes pilot_barcodes.txt --output-dir data
```

The seed list now contains 26 products: the original Nutella example plus 25 cereal and chocolate candidates. The output is written to the mounted `test6/data/` directory on the host, which is ignored by Git. This data-preparation step does not need ROCm or GPU access.

After running, open `test6/data/images/<barcode>.jpg`. The manifest currently has 26 records and all 26 have downloaded English ingredient-panel images; the 25 new records are marked `needs_manual_review`. Compare the printed panel with `off_ingredients_text` in `test6/data/manifest.csv`, then fill `verified_transcription` only after checking the image. Keep each product/image pair only if the panel is readable and the text can be verified.

## Run baseline OCR

The container includes Tesseract OCR. From inside the container, run it across the downloaded images:

```bash
python run_ocr.py
```

This writes `data/ocr_results.csv` with OCR text for each image. Character and word error rates are populated only if `verified_transcription` has been manually checked and `verified_transcription_scope` is set to `full_image`—the transcription must cover all readable text in that photo. Ingredient-only transcriptions are not scored against whole-image OCR. Unverified Open Food Facts text is never used as ground truth. Inspect `ocr_text` alongside the label images before drawing conclusions. To use an alternate language pack or page segmentation mode, pass Tesseract's language code and PSM, for example `python run_ocr.py --language eng --psm 6`.

## Add examples

Add one numeric product barcode per line to `docker_data/pilot_barcodes.txt`. Blank lines and lines beginning with `#` are ignored. Use products for the intended test language; the script accepts only ingredient-panel images whose image URL indicates that language:

```bash
python prepare_dataset.py \
  --barcodes pilot_barcodes.txt \
  --output-dir data \
  --language en
```

Open Food Facts asks API clients to identify themselves with a custom User-Agent. Set one appropriate to your project, and include a contact address if available:

```bash
python prepare_dataset.py \
  --barcodes pilot_barcodes.txt \
  --output-dir data \
  --user-agent 'FoodLabelPilot/0.1 (contact: you@example.org)'
```

Requests are made sequentially and paced to stay below the documented product-read rate limit. A curated 100–300 item pilot should take roughly 7–21 minutes, plus image transfer time. Do not use the search API as a bulk crawler.

## Curation and evaluation

The generated manifest contains the barcode, product name, Open Food Facts ingredient text, image URL, review status, and blank `verified_transcription` and `verified_transcription_scope` fields for new products. The supplied ingredient text is a reference only; it can be incomplete or wrong. Manually verify transcriptions from the actual package photo and exclude unreadable, mismatched, or out-of-scope labels. Set the scope to `full_image` only when your transcription covers all readable text in the photo; use `ingredients` for an ingredient-only transcription.

Use the printed label as the OCR transcription target. Split by product barcode only after curation so images or crops of one package cannot leak between training and evaluation. Keep a separate set of phone-captured photos for the final device test.

## Data use

Downloaded images and the generated manifest stay under the ignored `test6/data/` directory; do not commit them. Open Food Facts database contents are subject to ODbL/DbCL terms and product images to CC BY-SA, with possible additional rights in packaging artwork. Review the [official reuse conditions](https://world.openfoodfacts.org/data) before redistributing data or derived datasets. This pilot is for OCR/model evaluation, not medical or food-safety advice.
