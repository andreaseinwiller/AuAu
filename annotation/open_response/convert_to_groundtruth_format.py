import argparse
import json

import pandas as pd
from loguru import logger

from llm_audit.eval.open_response import convert_to_groundtruth_format


def is_groundtruth_format(path: str) -> bool:
    required_fields: list[str] = [
        "open_response",
        "dataset",
        "question_id",
        "model",
        "options_canonical",
        "disagree_to_agree",
        "neutral",
        "open_response_meta__user_prompts",
        "generating_model",
        "task",
        "refusal",
        "label",
    ]

    if path.endswith(".csv"):
        data = pd.read_csv(path)
        for required_field in required_fields:
            if required_field not in data.columns:
                raise ValueError("Found .csv without necessary columns. Only json exports can be converted.")
    elif path.endswith(".json"):
        try:
            with open(path) as f:
                data = json.load(f)
                for elem in data:
                    if not all(elem.get(field) for field in required_fields):
                        return False
        except json.JSONDecodeError:
            return False
    return True


parser = argparse.ArgumentParser()
parser.add_argument("--inputs", "-i", nargs="+", required=True)
parser.add_argument("--output", "-o", required=True)
args = parser.parse_args()

result = []
for path in args.inputs:
    if is_groundtruth_format(path):
        logger.info(f"{path} is already in required format")
        data = pd.read_csv(path)
    else:
        logger.info(f"Converting {path}")
        data = convert_to_groundtruth_format(path)
    result.append(data)

final_df = pd.concat(result, axis="index")
logger.info(f"Output file has {len(final_df)} annotations")
final_df.to_csv(args.output)
