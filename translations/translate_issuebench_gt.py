import argparse
import json
import os
import deepl

from dotenv import load_dotenv

from llm_audit import BASE_DIR


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate english ground truth from issuebench to another language with deepl."
    )
    parser.add_argument("language", default="de", choices=["ru", "de", "zh"])
    parser.add_argument("--output-file", default="translation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    data_path = BASE_DIR / "resources" / "input" / "datasets" / "issuebench" / "ground_truth_data_rebuttal.json"
    with data_path.open() as f:
        data = json.load(f)

    # Load existing translations if the cache file exists
    cache_path = (
        BASE_DIR / "resources" / "input" / "datasets" / "issuebench" / f"ground_truth_data_{args.language}.json"
    )
    translation_cache: dict[str, str] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            cache_data = json.load(f)
        response_key = f"response_{args.language}"
        translation_cache = {
            item["response"]: item[response_key] for item in cache_data if "response" in item and response_key in item
        }

    load_dotenv()
    deepl_client = deepl.DeepLClient(os.environ["DEEPL_API_KEY"])

    for item in data:
        text_to_translate = item["response"]
        if text_to_translate in translation_cache:
            item[f"response_{args.language}"] = translation_cache[text_to_translate]
        else:
            result = deepl_client.translate_text(text_to_translate, target_lang=args.language, source_lang="EN")
            assert isinstance(result, deepl.TextResult)
            item[f"response_{args.language}"] = result.text

    with open(args.output_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
