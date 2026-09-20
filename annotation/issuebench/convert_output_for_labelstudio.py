import argparse
import json
import random

from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_to_convert")  # judge dir
    parser.add_argument("output_file_path")
    parser.add_argument(
        "-n",
        "--num-samples",
        default=50,
        type=int,
        help="How many answers are selected for the resulting file",
    )
    parser.add_argument("--seed", default=42)
    parser.add_argument("--stratify-by-model", action="store_true", default=False)
    parser.add_argument("--select-autho-samples", action="store_true", default=False)
    parser.add_argument("--select-non-autho-samples", action="store_true", default=False)
    return parser.parse_args()


def parse_judge_response(response: str) -> None | dict[str, int]:
    val_to_boolean = {"yes": 1, "no": -1, "borderline": 0}
    try:
        parsed: dict[str, str] = json.loads(response)
        retval = {key: val_to_boolean[val] for key, val in parsed.items()}
        return retval
    except json.JSONDecodeError:
        return None
    except KeyError:
        return None


def extract_text_to_annotate(user_prompts: list[str]) -> str:
    """Extract the text following 'Text to annotate:' in the user_prompts field."""
    marker = "Text to annotate:"
    for prompt in user_prompts:
        idx = prompt.find(marker)
        if idx != -1:
            return prompt[idx + len(marker) :].strip()
    return ""


def extract_task(generating_user_prompts: list[str]) -> str:
    """Join generating_user_prompts into a single string."""
    return " ".join(generating_user_prompts)


def convert() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Crawl through dir tree to find all json files
    root = Path(args.dir_to_convert)
    data = []
    for dirpath, _, filenames in root.walk():
        for fpath in map(
            lambda fname: dirpath / fname,
            filter(lambda s: s.endswith(".json"), filenames),
        ):
            with fpath.open("r") as f:
                data.extend(json.load(f))

    if args.select_autho_samples:
        responses = [parse_judge_response(d["response"]) if d["response"] else {} for d in data]
        data = [
            data
            for resp, data in zip(responses, data)
            if resp
            and any(
                [
                    resp["aggression"] >= 0,
                    resp["submission"] >= 0,
                    resp["conventionalism"] >= 0,
                ]
            )
        ]
        with open("debug.json", "w") as f:
            json.dump(data, f, indent=True)

    if args.select_non_autho_samples:
        responses = [parse_judge_response(d["response"]) if d["response"] else {} for d in data]
        data = [
            data
            for resp, data in zip(responses, data)
            if resp
            and all(
                [
                    resp["aggression"] < 0,
                    resp["submission"] < 0,
                    resp["conventionalism"] < 0,
                ]
            )
        ]

    if args.stratify_by_model:
        print(
            f"Stratifying sample by model. This means --num-samples ({args.num_samples}) refers to the number of samples per model."
        )
        subsets = defaultdict(list)
        for result in data:
            subsets[result["generating_model_name"]].append(result)

        sample = []
        for subset in subsets.values():
            sample += random.sample(subset, args.num_samples)
    else:
        sample = random.sample(data, args.num_samples)

    # Convert the data elements into the format needed for labelstudio:
    # {"data": {ACTUAL_DATA}}
    sample = [
        {
            "data": {
                **elem,
                "text": extract_text_to_annotate(elem.get("user_prompts", [])),
                "task": extract_task(elem.get("generating_user_prompts", [])),
            }
        }
        for elem in sample
    ]

    with open(args.output_file_path, "w") as f:
        json.dump(sample, f, indent=True)


if __name__ == "__main__":
    convert()
