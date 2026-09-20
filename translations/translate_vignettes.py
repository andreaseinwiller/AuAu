#!/usr/bin/env python3
"""
JSON Translation Script using DeepL API

Translates specific fields in a JSON file to a target language while preserving
the original structure and keeping certain fields untranslated.
"""

import argparse
import json
import sys

from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Try to import deepl, but allow script to run without it for dry-run mode
try:
    import deepl

    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False


class FakeTranslator:
    """Mock translator for dry-run mode that returns the original text."""

    def translate_text(self, text: str, target_lang: str, source_lang: str = "EN") -> Any:
        """Return a mock translation object that just returns the original text."""

        class MockTranslation:
            def __init__(self, text: str) -> None:
                self.text = text

        return MockTranslation(text)


def get_translator(api_key: str, dry_run: bool) -> Any:
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
        return deepl.Translator(api_key)


def translate_vignette(vignette: dict[str, Any], translator: Any, target_lang: str) -> dict[str, Any]:
    """
    Translate specific fields in a vignette object.

    Args:
        vignette: The vignette object to translate
        translator: DeepL translator instance (real or fake)
        target_lang: Target language code (e.g., 'DE', 'FR', 'ES')

    Returns:
        Translated vignette object
    """
    # Fields to translate
    fields_to_translate = [
        "scenario",
        "question",
        "option_autho_high",
        "option_autho_medium",
        "option_neutral",
        "option_antiautho_medium",
        "option_antiautho_high",
    ]

    # Create a copy of the vignette
    translated_vignette = vignette.copy()

    # Translate each field
    for field in fields_to_translate:
        if field in vignette and vignette[field]:
            original_text = vignette[field]
            translation = translator.translate_text(original_text, target_lang=target_lang, source_lang="EN")
            translated_vignette[field] = translation.text

    return translated_vignette


def translate_json_file(input_file: str, target_lang: str, translator: Any, dry_run: bool) -> str:
    """
    Translate the JSON file and save to a new file.

    Args:
        input_file: Path to the input JSON file
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

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate that data is an array
    if not isinstance(data, list):
        print("Error: JSON file must contain an array at the root level.")
        sys.exit(1)

    print(f"Loaded {len(data)} items from {input_file}")

    # Translate the data
    translated_data = []
    total_vignettes = sum(len(item.get("vignettes", [])) for item in data)
    current_vignette = 0

    for item in data:
        translated_item = {
            "statement": item["statement"],  # Keep statement untranslated
            "vignettes": [],
        }

        for vignette in item.get("vignettes", []):
            current_vignette += 1
            print(
                f"  Translating vignette {current_vignette}/{total_vignettes} (ID: {vignette.get('id', 'unknown')})",
                end="\r",
            )

            translated_vignette = translate_vignette(vignette, translator, target_lang)
            translated_item["vignettes"].append(translated_vignette)

        translated_data.append(translated_item)

    print()  # New line after progress

    # Generate output filename
    output_path = input_path.with_name(input_path.name.removesuffix("en.json") + target_lang.lower() + ".json")

    # Save the translated data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    return str(output_path)


def main() -> None:
    """Main function to parse arguments and run the translation."""
    parser = argparse.ArgumentParser(
        description="Translate JSON file to target language using DeepL API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no API calls, test format only)
  python translate_json.py input.json --target-lang DE --dry-run

  # Real translation to German
  python translate_json.py input.json --target-lang DE --api-key YOUR_API_KEY

  # Real translation using environment variable
  export DEEPL_API_KEY=your_api_key_here
  python translate_json.py input.json --target-lang FR

Supported languages:
  DE (German), FR (French), ES (Spanish), IT (Italian), JA (Japanese),
  NL (Dutch), PL (Polish), PT (Portuguese), RU (Russian), ZH (Chinese), etc.
  See DeepL documentation for full list.
        """,
    )

    parser.add_argument("input_file", help="Path to the input JSON file")

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
    output_file = translate_json_file(args.input_file, target_lang, translator, args.dry_run)

    print(f"Translation complete!")
    print(f"Output saved to: {output_file}")

    if args.dry_run:
        print("\n This was a dry run. To perform real translation, remove --dry-run flag")


if __name__ == "__main__":
    main()
