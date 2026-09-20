#!/usr/bin/env python3
"""
CSV Translation Script using DeepL API

Translates the "open_response" column in a CSV file to a target language while
preserving all other columns unchanged.
"""

import argparse
import csv
import sys

from pathlib import Path
from typing import Any, Protocol
from dotenv import load_dotenv

# Try to import deepl, but allow script to run without it for dry-run mode
try:
    import deepl

    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False


class _TextResult(Protocol):
    text: str


class _TextTranslator(Protocol):
    def translate_text(self, text: str, *, target_lang: str, source_lang: str = "EN") -> _TextResult:
        pass


class FakeTranslator:
    """Mock translator for dry-run mode that returns the original text."""

    def translate_text(self, text: str, target_lang: str, source_lang: str = "EN") -> _TextResult:
        """Return a mock translation object that just returns the original text."""

        class MockTranslation:
            def __init__(self, text: str) -> None:
                self.text = text

        return MockTranslation(text)


def get_translator(api_key: str, dry_run: bool) -> _TextTranslator:
    """
    Get the appropriate translator based on dry_run mode.

    Args:
        api_key: DeepL API key (ignored in dry_run mode)
        dry_run: If True, returns a fake translator

    Returns:
        Translator instance (real or fake)
    """
    if dry_run:
        print("Running in DRY RUN mode - no API calls will be made")
        return FakeTranslator()
    else:
        if not DEEPL_AVAILABLE:
            print("Error: deepl package is not installed.")
            print("Install it with: pip install deepl")
            print("Or use --dry-run flag to test without the package.")
            sys.exit(1)
        if not api_key:
            print("Error: DeepL API key is required for real translation mode.")
            print("Please set DEEPL_API_KEY environment variable or use --dry-run flag.")
            sys.exit(1)
        return deepl.Translator(api_key)  # type: ignore[return-value]


def translate_csv_file(input_file: str, target_lang: str, translator: _TextTranslator, dry_run: bool) -> str:
    """
    Translate the CSV file and save to a new file.

    Args:
        input_file: Path to the input CSV file
        target_lang: Target language code
        translator: DeepL translator instance
        dry_run: Whether this is a dry run

    Returns:
        Path to the output file
    """
    # Read the input file
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Read all rows from the CSV
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            print("Error: CSV file appears to be empty or malformed.")
            sys.exit(1)

        if "open_response" not in fieldnames:
            print("Error: CSV file must contain an 'open_response' column.")
            print(f"Found columns: {', '.join(fieldnames)}")
            sys.exit(1)

        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {input_file}")

    # Translate the data
    translated_rows = []
    for idx, row in enumerate(rows, 1):
        print(f"Translating row {idx}/{len(rows)}", end="\r")

        # Create a copy of the row
        translated_row = row.copy()

        # Translate only the open_response field if it's not empty
        if row.get("open_response") and row["open_response"].strip():
            original_text = row["open_response"]
            translation = translator.translate_text(original_text, target_lang=target_lang, source_lang="EN")
            translated_row["open_response"] = translation.text

        translated_rows.append(translated_row)

    print()  # New line after progress

    # Generate output filename
    output_path = input_path.with_name(input_path.stem + f"_{target_lang.lower()}" + input_path.suffix)

    # Save the translated data
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(translated_rows)

    return str(output_path)


def main() -> None:
    """Main function to parse arguments and run the translation."""
    parser = argparse.ArgumentParser(
        description="Translate CSV file's 'open_response' column to target language using DeepL API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no API calls, test format only)
  python translate_csv.py input.csv --target-lang DE --dry-run

  # Real translation to German
  python translate_csv.py input.csv --target-lang DE --api-key YOUR_API_KEY

  # Real translation using environment variable
  export DEEPL_API_KEY=your_api_key_here
  python translate_csv.py input.csv --target-lang FR

Supported languages:
  DE (German), FR (French), ES (Spanish), IT (Italian), JA (Japanese),
  NL (Dutch), PL (Polish), PT (Portuguese), RU (Russian), ZH (Chinese), etc.
  See DeepL documentation for full list.
        """,
    )

    parser.add_argument("input_file", help="Path to the input CSV file")

    parser.add_argument(
        "--target-lang",
        "-t",
        required=True,
        help="Target language code (e.g., DE, FR, ES, IT)",
    )

    parser.add_argument(
        "--api-key",
        "-k",
        default=None,
        help="DeepL API key (or set DEEPL_API_KEY environment variable)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - test format without making API calls (no credits used)",
    )

    args = parser.parse_args()

    # Get API key from arguments or environment
    import os

    load_dotenv()
    api_key = args.api_key or os.environ.get("DEEPL_API_KEY")
    if api_key is None:
        api_key = ""  # handled by get_translator

    # Get translator (real or fake based on dry_run)
    translator = get_translator(api_key, args.dry_run)

    # Normalize target language to uppercase
    target_lang = args.target_lang.upper()

    print(f"Translating to: {target_lang}")

    # Perform translation
    output_file = translate_csv_file(args.input_file, target_lang, translator, args.dry_run)

    print(f"Translation complete!")
    print(f"Output saved to: {output_file}")

    if args.dry_run:
        print("\n This was a dry run. To perform real translation, remove --dry-run flag")


if __name__ == "__main__":
    main()
