import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_DATA_DIR = Path("data")
OUTPUT_FIELDS = (
    "barcode",
    "product_name",
    "image_filename",
    "review_status",
    "ocr_text",
    "verified_transcription",
    "verified_transcription_scope",
    "character_error_rate",
    "word_error_rate",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline Tesseract OCR on curated ingredient-label images."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.csv")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_DATA_DIR / "images")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR / "ocr_results.csv")
    parser.add_argument("--language", default="eng", help="Tesseract language code")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    return parser.parse_args()


def normalized_text(text):
    return " ".join(text.lower().split())


def edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference, hypothesis, unit):
    reference_units = unit(normalized_text(reference))
    hypothesis_units = unit(normalized_text(hypothesis))
    if not reference_units:
        return ""
    return edit_distance(reference_units, hypothesis_units) / len(reference_units)


def character_error_rate(reference, hypothesis):
    return error_rate(reference, hypothesis, list)


def word_error_rate(reference, hypothesis):
    return error_rate(reference, hypothesis, str.split)


def run_tesseract(image_path, language, psm):
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError(
            "Tesseract is not installed. Rebuild the Test 6 image with ./run.sh."
        )
    result = subprocess.run(
        [executable, str(image_path), "stdout", "-l", language, "--psm", str(psm)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_manifest(path):
    required_fields = {
        "barcode",
        "product_name",
        "image_filename",
        "review_status",
        "verified_transcription",
    }
    with path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing_fields))}"
            )
        return list(reader)


def resolve_image(images_dir, filename):
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Invalid image filename in manifest: {filename!r}")
    image_path = images_dir / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Image listed in manifest does not exist: {image_path}")
    return image_path


def process_manifest(manifest_path, images_dir, language, psm):
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError(f"No records found in {manifest_path}")

    results = []
    for index, row in enumerate(rows, 1):
        image_filename = row["image_filename"].strip()
        if not image_filename:
            print(f"[{index}/{len(rows)}] {row['barcode']}: skipped (no image)")
            continue

        image_path = resolve_image(images_dir, image_filename)
        ocr_text = run_tesseract(image_path, language, psm)
        verified_text = (row["verified_transcription"] or "").strip()
        transcription_scope = (row.get("verified_transcription_scope") or "").strip()
        full_image_reference = (
            verified_text if transcription_scope == "full_image" else ""
        )
        results.append(
            {
                "barcode": row["barcode"],
                "product_name": row["product_name"],
                "image_filename": image_filename,
                "review_status": row["review_status"],
                "ocr_text": ocr_text,
                "verified_transcription": verified_text,
                "verified_transcription_scope": transcription_scope,
                "character_error_rate": (
                    character_error_rate(full_image_reference, ocr_text)
                    if full_image_reference
                    else ""
                ),
                "word_error_rate": (
                    word_error_rate(full_image_reference, ocr_text)
                    if full_image_reference
                    else ""
                ),
            }
        )
        status = (
            "scored against full-image transcription"
            if full_image_reference
            else "OCR only (no verified full-image transcription)"
        )
        print(f"[{index}/{len(rows)}] {row['barcode']}: {status}")
    return results


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.language):
        raise ValueError("--language must be a Tesseract language code")
    if not 0 <= args.psm <= 13:
        raise ValueError("--psm must be between 0 and 13")

    results = process_manifest(args.manifest, args.images_dir, args.language, args.psm)
    write_results(args.output, results)
    scored_count = sum(bool(row["character_error_rate"]) for row in results)
    print(
        f"Wrote {len(results)} OCR results to {args.output}; "
        f"{scored_count} compared with verified transcriptions."
    )


if __name__ == "__main__":
    main()
