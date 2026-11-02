import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://world.openfoodfacts.org/api/v3.6/product"
FIELDS = "code,product_name,ingredients_text,image_ingredients_url"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
REQUEST_INTERVAL_SECONDS = 4.2
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MANIFEST_FIELDS = (
    "barcode",
    "product_name",
    "off_ingredients_text",
    "image_filename",
    "image_url",
    "product_url",
    "review_status",
    "verified_transcription",
    "verified_transcription_scope",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a small, barcode-curated Open Food Facts OCR pilot."
    )
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language", default="en")
    parser.add_argument("--user-agent", default="pytorch-learn-dataset-pilot/0.1")
    return parser.parse_args()


def read_barcodes(path):
    barcodes = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        barcode = line.strip()
        if not barcode or barcode.startswith("#"):
            continue
        if not barcode.isdigit():
            raise ValueError(f"{path}:{line_number}: barcode must contain only digits")
        if barcode not in seen:
            barcodes.append(barcode)
            seen.add(barcode)
    return barcodes


def get_json(url, user_agent):
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Open Food Facts request failed for {url}: {error}") from error


def get_image(url, destination, user_agent):
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise RuntimeError(
                    f"Expected an image from {url}, received {content_type}"
                )
            image = response.read(MAX_IMAGE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Image download failed for {url}: {error}") from error

    if len(image) > MAX_IMAGE_BYTES:
        raise RuntimeError(f"Image exceeds the 10 MB limit: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(image)


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_manifest(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as manifest_file:
        return {
            row["barcode"]: row
            for row in csv.DictReader(manifest_file)
            if row.get("barcode")
        }


def preserve_manual_review(row, previous_row):
    if (
        previous_row
        and row["image_url"]
        and row["image_url"] == previous_row.get("image_url")
    ):
        row["review_status"] = previous_row.get(
            "review_status", row["review_status"]
        )
        row["verified_transcription"] = previous_row.get(
            "verified_transcription", ""
        )
        row["verified_transcription_scope"] = previous_row.get(
            "verified_transcription_scope", ""
        )


def collect_product(barcode, language, user_agent):
    query = urlencode({"fields": FIELDS})
    url = f"{API_BASE}/{barcode}.json?{query}"
    record = get_json(url, user_agent)
    if record.get("status") != "success":
        return None

    product = record.get("product") or {}
    image_url = product.get("image_ingredients_url") or ""
    language_marker = f"ingredients_{language}."
    if image_url and language_marker in image_url:
        review_status = (
            "needs_manual_review"
            if product.get("ingredients_text")
            else "needs_transcription"
        )
    elif image_url:
        image_url = ""
        review_status = "no_image_in_requested_language"
    else:
        review_status = "missing_ingredient_image"

    return {
        "barcode": barcode,
        "product_name": product.get("product_name", ""),
        "off_ingredients_text": product.get("ingredients_text", ""),
        "image_url": image_url,
        "product_url": f"https://world.openfoodfacts.org/product/{barcode}",
        "review_status": review_status,
        "verified_transcription": "",
        "verified_transcription_scope": "",
    }


def main():
    args = parse_args()
    if not re.fullmatch(r"[a-z]{2}", args.language):
        raise ValueError("--language must be a two-letter lowercase language code")

    barcodes = read_barcodes(args.barcodes)
    if not barcodes:
        raise ValueError(f"No barcodes found in {args.barcodes}")

    image_dir = args.output_dir / "images"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    rows = []
    last_request_time = None
    previous_rows = read_existing_manifest(manifest_path)

    for index, barcode in enumerate(barcodes):
        if last_request_time is not None:
            elapsed = time.monotonic() - last_request_time
            time.sleep(max(0, REQUEST_INTERVAL_SECONDS - elapsed))

        row = collect_product(barcode, args.language, args.user_agent)
        last_request_time = time.monotonic()
        if row is not None:
            preserve_manual_review(row, previous_rows.get(barcode))
        if row is None:
            row = {
                "barcode": barcode,
                "product_name": "",
                "off_ingredients_text": "",
                "image_url": "",
                "product_url": f"https://world.openfoodfacts.org/product/{barcode}",
                "review_status": "product_not_found",
                "verified_transcription": "",
                "verified_transcription_scope": "",
            }
        elif row["image_url"]:
            filename = f"{barcode}.jpg"
            get_image(row["image_url"], image_dir / filename, args.user_agent)
            row["image_filename"] = filename
        else:
            row["image_filename"] = ""

        rows.append(row)
        write_manifest(manifest_path, rows)
        print(f"[{index + 1}/{len(barcodes)}] {barcode}: {row['review_status']}")

    print(f"Wrote {len(rows)} records to {manifest_path}")


if __name__ == "__main__":
    main()
