#!/usr/bin/env python3
"""Remove unparsed <think> traces from one columnar IssueBench judge file.

The source JSON is column-oriented: each top-level key is a column and each
column maps row ids to values. This script writes additive outputs only:
  * results_reprocessed.json beside the source file
  * reprocess_summary.json beside the source file

It intentionally leaves label/refusal and all other non-response columns untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import re

from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path(
    "resources/output/reduction_eval_datav5-en-1_0-1-42/IssueBench/judge/"
    "tiiuae/Falcon-H1R-7B/unknown_model/results.json"
)

THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def split_thinking(response: str) -> tuple[list[str], str]:
    traces = [match.group(1).strip() for match in THINK_RE.finditer(response)]
    without_thinking = THINK_RE.sub("", response).strip()
    return traces, without_thinking


def percent(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(part * 100 / total, 4)


def sorted_row_ids(data: dict[str, object]) -> list[str]:
    response_col = data.get("response")
    if not isinstance(response_col, dict):
        raise ValueError("Expected a columnar JSON object with a dict-valued 'response' column")
    return sorted(response_col, key=lambda key: int(key) if key.isdigit() else key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-name", default="results_reprocessed.json")
    parser.add_argument("--summary-name", default="reprocess_summary.json")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{args.input} does not contain a columnar JSON object")

    row_ids = sorted_row_ids(data)
    response_col = data["response"]
    if not isinstance(response_col, dict):
        raise ValueError("Expected 'response' to be a dict")

    reprocessed = copy.deepcopy(data)
    reprocessed["original_response"] = copy.deepcopy(response_col)
    reprocessed["thinking_traces"] = {}

    stats: Counter[str] = Counter(rows_total=len(row_ids))
    unchanged_examples: list[dict[str, str]] = []

    for row_id in row_ids:
        response = response_col.get(row_id)
        if not isinstance(response, str):
            stats["missing_or_non_string_response"] += 1
            traces: list[str] = []
            final_text = response
        else:
            traces, final_text = split_thinking(response)

        reprocessed["thinking_traces"][row_id] = traces
        reprocessed["response"][row_id] = final_text

        if traces:
            stats["responses_with_thinking"] += 1
            stats["thinking_traces_total"] += len(traces)
            stats["responses_changed"] += 1
        else:
            stats["responses_without_thinking"] += 1
            if len(unchanged_examples) < 10:
                unchanged_examples.append(
                    {
                        "row_id": row_id,
                        "response_prefix": str(final_text)[:300],
                    }
                )

    total = stats["rows_total"]
    summary = {
        "input": str(args.input),
        "output": str(args.input.with_name(args.output_name)),
        "stats": {
            **dict(stats),
            "responses_with_thinking_percent": percent(stats["responses_with_thinking"], total),
            "responses_changed_percent": percent(stats["responses_changed"], total),
        },
        "untouched_columns": ["label", "refusal"],
        "unchanged_examples": unchanged_examples,
    }

    output_path = args.input.with_name(args.output_name)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(reprocessed, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    summary_path = args.input.with_name(args.summary_name)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary["stats"], indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
