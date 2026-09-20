import argparse
import json

from typing import Any
from loguru import logger

from llm_audit.eval.issuebench import convert_to_groundtruth_format


def is_groundtruth_format(path: str) -> bool:
    required_fields = ["prompt", "model", "label_aggression", "label_submission", "label_conventionalism", "response"]
    try:
        with open(path) as f:
            data = json.load(f)
            for elem in data:
                if not all(elem.get(field) for field in required_fields):
                    return False
    except json.JSONDecodeError:
        return False
    return True


def remove_duplicates(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated = []
    seen = set()

    for elem in data:
        key = json.dumps(elem, sort_keys=True)
        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(elem)

    return deduplicated


def add_combined_label(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for elem in data:
        if elem.get("label_combined") is None:
            elem["label_combined"] = (
                1.0
                if any(
                    [
                        elem["label_aggression"],
                        elem["label_submission"],
                        elem["label_conventionalism"],
                    ]
                )
                else -1.0
            )
    return data


def add_refusal_label(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for elem in data:
        if elem.get("label_refusal") is None:
            elem["label_refusal"] = -1.0
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--inputs", "-i", nargs="+", required=True)
parser.add_argument("--output", "-o", required=True)
args = parser.parse_args()

result = []
for path in args.inputs:
    if is_groundtruth_format(path):
        logger.info(f"{path} is already in required format")
        with open(path) as f:
            data = json.load(f)
    else:
        logger.info(f"Converting {path}")
        data = convert_to_groundtruth_format([path])

    result.extend(data)

deduplicated_result = remove_duplicates(result)
duplicates_removed = len(result) - len(deduplicated_result)
if duplicates_removed:
    logger.info(f"Removed {duplicates_removed} duplicate entries")

deduplicated_result = add_combined_label(deduplicated_result)
deduplicated_result = add_refusal_label(deduplicated_result)

with open(args.output, "w") as f:
    json.dump(deduplicated_result, f, indent=4)
