import argparse
import json
import os
import deepl

from dotenv import load_dotenv

from issuebench import prepare_generation_data


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate english prompts from the issuebench to another language with an LLM."
    )
    parser.add_argument("language", default="de", choices=["ru", "de", "zh"])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--num-prompts",
        default=1000,
        type=int,
        help="For how many prompts answers should be generated. Random selection. Only needed when generating. (default: 10000)",
    )
    parser.add_argument("--output-file", default="translation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    inputs, metas = prepare_generation_data(
        argparse.Namespace(seed=args.seed, num_prompts=args.num_prompts, language="en")
    )

    load_dotenv()
    deepl_client = deepl.DeepLClient(os.environ["DEEPL_API_KEY"])

    results = deepl_client.translate_text(inputs, target_lang=args.language, source_lang="EN")
    assert isinstance(results, list)
    texts = [{"text": r.text} for r in results]
    with open(args.output_file, "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
