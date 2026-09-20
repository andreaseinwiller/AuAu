#!/usr/bin/env python3
"""Reprocess Falcon-H1R judge outputs with unparsed <think> traces.

This writes additive files only:
  * results_reprocessed.json beside every source results.json
  * reprocess_summary.json in the dataset root
"""

from __future__ import annotations

import argparse
import copy
import json
import re

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*(.*?)\s*```$", flags=re.DOTALL)


DERIVED_FIELDS = (
    "closed_response",
    "response_value",
    "refusal",
    "valid_response_format",
    "valid_response_value",
)


def split_thinking(response: str) -> tuple[list[str], str]:
    traces = [match.group(1).strip() for match in THINK_RE.finditer(response)]
    without_thinking = THINK_RE.sub("", response).strip()
    return traces, without_thinking


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None, str]:
    parse_text = strip_code_fence(text)
    try:
        parsed = json.loads(parse_text)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} at line {exc.lineno} column {exc.colno}", parse_text
    if not isinstance(parsed, dict):
        return None, "parsed JSON is not an object", parse_text
    return parsed, None, parse_text


def normalize_scalar(value: Any) -> str | None:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str):
        return value.strip()
    return None


def reprocess_item(item: dict[str, Any], stats: Counter[str]) -> dict[str, Any]:
    new_item = copy.deepcopy(item)
    metas = new_item.get("closed_response_meta")
    if not isinstance(metas, list):
        stats["items_without_closed_response_meta_list"] += 1
        return new_item

    new_item["original_closed_response_fields"] = {
        field: copy.deepcopy(item.get(field)) for field in DERIVED_FIELDS if field in item
    }

    closed_response: list[str | None] = []
    refusal: list[str | None] = []
    valid_response_format: list[bool] = []
    valid_response_value: list[bool] = []
    thinking_traces: list[list[str]] = []

    for meta in metas:
        stats["judge_responses_total"] += 1
        if not isinstance(meta, dict):
            stats["non_object_closed_response_meta"] += 1
            closed_response.append(None)
            refusal.append(None)
            valid_response_format.append(False)
            valid_response_value.append(False)
            thinking_traces.append([])
            continue

        raw_response = meta.get("response")
        if not isinstance(raw_response, str):
            stats["missing_or_non_string_judge_response"] += 1
            meta["thinking_traces"] = []
            meta["response_without_thinking"] = raw_response
            meta["parsed_final_json"] = None
            meta["json_parse_error"] = "missing or non-string response"
            closed_response.append(None)
            refusal.append(None)
            valid_response_format.append(False)
            valid_response_value.append(False)
            thinking_traces.append([])
            continue

        traces, final_text = split_thinking(raw_response)
        parsed, parse_error, attempted_text = parse_json_response(final_text)

        meta["thinking_traces"] = traces
        meta["response_without_thinking"] = final_text
        meta["json_parse_attempt_text"] = attempted_text
        meta["parsed_final_json"] = parsed
        meta["json_parse_error"] = parse_error

        thinking_traces.append(traces)
        if traces:
            stats["judge_responses_with_thinking"] += 1
            stats["thinking_traces_total"] += len(traces)

        if parsed is None:
            stats["invalid_json_after_simple_cleanup"] += 1
            closed_response.append(None)
            refusal.append(None)
            valid_response_format.append(False)
            valid_response_value.append(False)
            continue

        stats["valid_json_after_simple_cleanup"] += 1
        score_value = normalize_scalar(parsed.get("score"))
        refusal_value = normalize_scalar(parsed.get("refusal"))
        value_ok = score_value is not None and refusal_value in {"0", "1"}

        closed_response.append(score_value)
        refusal.append(refusal_value)
        valid_response_format.append(True)
        valid_response_value.append(value_ok)
        if not value_ok:
            stats["json_missing_or_invalid_score_refusal"] += 1

    new_item["closed_response_thinking_traces"] = thinking_traces
    new_item["closed_response"] = closed_response
    new_item["response_value"] = closed_response[0] if closed_response else None
    new_item["refusal"] = refusal
    new_item["valid_response_format"] = valid_response_format
    new_item["valid_response_value"] = valid_response_value
    return new_item


def percent(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(part * 100 / total, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("resources/output/final-authsys-en-1_0-10-42-falconh1r7b-rejudge"),
    )
    parser.add_argument("--output-name", default="results_reprocessed.json")
    parser.add_argument("--summary-name", default="reprocess_summary.json")
    args = parser.parse_args()

    root = args.root
    all_stats: Counter[str] = Counter()
    per_file: dict[str, dict[str, Any]] = {}
    invalid_examples: list[dict[str, Any]] = []
    changed_derived_fields = 0

    paths = sorted(path for path in root.rglob("results.json") if path.name == "results.json")
    all_stats["files_total"] = len(paths)

    for path in paths:
        output_path = path.with_name(args.output_name)
        if output_path.exists():
            all_stats["files_skipped_existing_output"] += 1
            continue

        all_stats["files_processed"] += 1
        file_stats: Counter[str] = Counter()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a JSON list")

        new_data = []
        for index, item in enumerate(data):
            all_stats["items_total"] += 1
            file_stats["items_total"] += 1
            if not isinstance(item, dict):
                all_stats["non_object_items"] += 1
                file_stats["non_object_items"] += 1
                new_data.append(item)
                continue

            reprocessed = reprocess_item(item, file_stats)
            new_data.append(reprocessed)

            original = reprocessed.get("original_closed_response_fields", {})
            changed = any(
                field in original and original.get(field) != reprocessed.get(field) for field in DERIVED_FIELDS
            )
            if changed:
                changed_derived_fields += 1
                file_stats["items_with_changed_derived_fields"] += 1

            metas = reprocessed.get("closed_response_meta")
            if isinstance(metas, list):
                for meta_index, meta in enumerate(metas):
                    if (
                        isinstance(meta, dict)
                        and meta.get("json_parse_error") is not None
                        and len(invalid_examples) < 25
                    ):
                        invalid_examples.append(
                            {
                                "path": str(path.relative_to(root)),
                                "item_index": index,
                                "meta_index": meta_index,
                                "json_parse_error": meta.get("json_parse_error"),
                                "response_without_thinking_prefix": str(meta.get("response_without_thinking"))[:500],
                            }
                        )

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(new_data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        all_stats.update(file_stats)
        per_file[str(path.relative_to(root))] = dict(file_stats)

    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    for relative_path, _file_stats in per_file.items():
        model_key = "/".join(Path(relative_path).parts[-3:-1])
        by_model[model_key].update(_file_stats)

    total_judge_responses = all_stats["judge_responses_total"]
    invalid_json = all_stats["invalid_json_after_simple_cleanup"]
    valid_json = all_stats["valid_json_after_simple_cleanup"]
    summary = {
        "root": str(root),
        "output_name": args.output_name,
        "stats": {
            **dict(all_stats),
            "items_with_changed_derived_fields": changed_derived_fields,
            "valid_json_after_simple_cleanup_percent": percent(valid_json, total_judge_responses),
            "invalid_json_after_simple_cleanup_percent": percent(invalid_json, total_judge_responses),
            "judge_responses_with_thinking_percent": percent(
                all_stats["judge_responses_with_thinking"], total_judge_responses
            ),
        },
        "by_model": {
            model: {
                **dict(stats),
                "invalid_json_after_simple_cleanup_percent": percent(
                    stats["invalid_json_after_simple_cleanup"],
                    stats["judge_responses_total"],
                ),
            }
            for model, stats in sorted(by_model.items())
        },
        "invalid_examples": invalid_examples,
        "per_file": per_file,
    }

    summary_path = root / args.summary_name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary["stats"], indent=2))
    print(
        f"Wrote {all_stats['files_processed']} {args.output_name} files "
        f"({all_stats['files_skipped_existing_output']} already existed)"
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
