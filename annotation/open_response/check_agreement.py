import argparse
import json
import krippendorff

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.metrics

from pathlib import Path
from typing import Any, cast
from loguru import logger

from llm_audit.eval.open_response import normalize_raw_likert_scale_score

DEFAULT_RATER_1 = "/root/llm-audit/resources/input/datasets/reduction/reduction_ground_truth_4-max_en.csv"
DEFAULT_RATER_2 = "/root/llm-audit/resources/input/datasets/reduction/reduction_ground_truth_5-max_en.csv"
NON_ANSWER_LABEL = -10_000
NON_ANSWER_SIDE_LABEL = 99
NON_ANSWER_REFUSAL = "Non-Answer"

DEFAULT_ID_COLUMNS = [
    "open_response",
    "dataset",
    "question_id",
    "model",
    "disagree_to_agree",
    "neutral",
    "generating_model",
    "task",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compute inter-annotator agreement for open-response reduction ground-truth annotations.")
    )
    parser.add_argument(
        "--rater-1",
        "-a",
        nargs="+",
        default=[DEFAULT_RATER_1],
        help="CSV ground-truth or Label Studio JSON file(s) for the first rater.",
    )
    parser.add_argument(
        "--rater-2",
        "-b",
        nargs="+",
        default=[DEFAULT_RATER_2],
        help="CSV ground-truth or Label Studio JSON file(s) for the second rater.",
    )
    parser.add_argument(
        "--id-columns",
        nargs="+",
        default=DEFAULT_ID_COLUMNS,
        help="Columns used to match annotations across raters.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Optional prefix for writing CSV tables, e.g. eval/data/reduction_agreement.",
    )
    return parser.parse_args()


def load_open_response_annotations(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.suffix == ".csv":
            frame = pd.read_csv(path_obj)
        elif path_obj.suffix == ".json":
            frame = convert_labelstudio_json_to_groundtruth_format(path_obj)
        else:
            raise ValueError(f"Unsupported annotation file type: {path}")

        unnamed_cols = [col for col in frame.columns if str(col).startswith("Unnamed:")]
        frame = frame.drop(columns=unnamed_cols)
        frames.append(frame)

    data = pd.concat(frames, axis=0, ignore_index=True)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data["neutral"] = pd.to_numeric(data["neutral"], errors="coerce")
    data = data.dropna(subset=["neutral"])
    data["is_non_answer"] = data["refusal"].astype(str).str.strip().eq(NON_ANSWER_REFUSAL)

    unexpected_missing_label = data["label"].isna() & ~data["is_non_answer"]
    if unexpected_missing_label.any():
        logger.warning(
            f"Dropping {unexpected_missing_label.sum()} row(s) with missing label and refusal != {NON_ANSWER_REFUSAL!r}"
        )
        data = data.loc[~unexpected_missing_label].copy()

    data["raw_label"] = data["label"]
    normalized = data.apply(
        lambda row: normalize_raw_likert_scale_score(
            dataset_label=row["dataset"],
            id=row["question_id"],
            raw_score=row["raw_label"],
            refusal=bool(row["is_non_answer"]),
        ),
        axis=1,
    )
    data["label"] = normalized["normalized_score"]
    data.loc[data["is_non_answer"], "label"] = NON_ANSWER_LABEL
    return data


def _parse_label_studio_item(item: dict[str, Any]) -> dict[str, Any]:
    data = item["data"]
    record: dict[str, Any] = {
        "open_response": data["open_response_meta"]["response"],
        "dataset": data["dataset"],
        "question_id": data["statement_id"],
        "model": data["open_response_meta"]["model_name"],
        "options_canonical": data["options_canonical"],
        "disagree_to_agree": data["disagree_to_agree"],
        "neutral": data["neutral"],
        "open_response_meta__user_prompts": data["open_response_meta"]["user_prompts"],
        "generating_model": data["open_response_meta"]["model_name"],
        "task": data["task"],
    }
    for annot in item["annotations"][0]["result"]:
        key = annot["from_name"]
        value = annot["value"]["choices"][0]
        if key == "refusal":
            record[key] = value
        elif key == "label":
            record[key] = int(value.split("=")[0].strip())
        else:
            raise ValueError(f"Unexpected annotation key: {key}")
    return record


def convert_labelstudio_json_to_groundtruth_format(path: Path) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f)
    return pd.DataFrame.from_records([_parse_label_studio_item(item) for item in data])


def validate_id_columns(data: pd.DataFrame, id_columns: list[str], name: str) -> None:
    missing = sorted(set(id_columns) - set(data.columns))
    if missing:
        raise ValueError(f"{name} is missing ID column(s): {missing}")

    duplicated = data.duplicated(id_columns).sum()
    if duplicated:
        raise ValueError(
            f"{name} contains {duplicated} duplicate rows for --id-columns={id_columns}. "
            "Use more specific ID columns to avoid a cartesian merge."
        )


def merge_annotations(
    rater_1: pd.DataFrame,
    rater_2: pd.DataFrame,
    id_columns: list[str],
) -> pd.DataFrame:
    validate_id_columns(rater_1, id_columns, "rater_1")
    validate_id_columns(rater_2, id_columns, "rater_2")

    merged = rater_1.merge(
        rater_2,
        how="inner",
        on=id_columns,
        suffixes=("_1", "_2"),
    )
    logger.info(f"Matched {len(merged)} annotated items across raters")

    unmatched_1 = len(rater_1) - len(merged)
    unmatched_2 = len(rater_2) - len(merged)
    if unmatched_1 or unmatched_2:
        logger.warning(f"Unmatched rows: rater_1={unmatched_1}, rater_2={unmatched_2}")

    return merged


def _safe_krippendorff_alpha(
    data: pd.DataFrame,
    col1: str,
    col2: str,
    level_of_measurement: str,
) -> float:
    reliability_data = data[[col1, col2]].astype(float).T.to_numpy()
    if len(np.unique(reliability_data[~np.isnan(reliability_data)])) < 2:
        return float("nan")

    try:
        return float(
            krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement=level_of_measurement,
            )
        )
    except ValueError as exc:
        logger.error(f"Krippendorff alpha failed for {col1}/{col2}: {exc}")
        return float("nan")


def _safe_cohens_kappa(y1: npt.NDArray[Any], y2: npt.NDArray[Any], weights: str | None = None) -> float:
    if len(np.unique(np.concatenate([y1, y2]))) < 2:
        return float("nan")

    # Cohen's kappa in sklearn expects discrete labels, but our normalized
    # scores are floats on a finite ordered scale. Re-encode the shared values
    # onto integer ranks while preserving their order.
    labels = np.sort(np.unique(np.concatenate([y1, y2])))
    y1_encoded = np.searchsorted(labels, y1)
    y2_encoded = np.searchsorted(labels, y2)
    return float(sklearn.metrics.cohen_kappa_score(y1_encoded, y2_encoded, weights=weights))


def _agreement_record(
    df: pd.DataFrame,
    name: str,
    col1: str,
    col2: str,
    level_of_measurement: str,
    *,
    weighted_kappa: bool = False,
) -> dict[str, Any]:
    y1 = df[col1].to_numpy()
    y2 = df[col2].to_numpy()
    record = {
        "level": name,
        "n": len(df),
        "agreement": float((y1 == y2).mean()),
        "krippendorff_alpha": _safe_krippendorff_alpha(
            df,
            col1,
            col2,
            level_of_measurement,
        ),
        "cohens_kappa": _safe_cohens_kappa(y1, y2),
    }
    if weighted_kappa:
        record["quadratic_weighted_kappa"] = _safe_cohens_kappa(y1, y2, weights="quadratic")
    return record


def _neutral_side(label: pd.Series) -> pd.Series:
    side: pd.Series = np.sign(label).astype(int)
    side = side.replace({-1: 0, 0: 1, 1: 2})
    side.loc[label == NON_ANSWER_LABEL] = NON_ANSWER_SIDE_LABEL
    return side


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["label_neutral_side_1"] = _neutral_side(df["label_1"])
    df["label_neutral_side_2"] = _neutral_side(df["label_2"])
    return df


def agreement_statistics(df: pd.DataFrame) -> pd.DataFrame:
    records = [
        _agreement_record(
            df,
            "label_exact",
            "label_1",
            "label_2",
            "ordinal",
            weighted_kappa=True,
        ),
        _agreement_record(
            df,
            "label_neutral_side",
            "label_neutral_side_1",
            "label_neutral_side_2",
            "nominal",
        ),
    ]
    return pd.DataFrame.from_records(records)


def _mean_excluding_value(series: pd.Series, excluded_value: int) -> float:
    numeric = series.replace(excluded_value, np.nan)
    return float(numeric.mean())


def _value_counts_with_non_answer(series: pd.Series, non_answer_value: int) -> dict[Any, int]:
    counts = series.value_counts(dropna=False).sort_index().to_dict()
    return {(NON_ANSWER_REFUSAL if value == non_answer_value else value): count for value, count in counts.items()}


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    records = []

    for level, col1, col2, non_answer_value in [
        ("label_exact", "label_1", "label_2", NON_ANSWER_LABEL),
        ("label_neutral_side", "label_neutral_side_1", "label_neutral_side_2", NON_ANSWER_SIDE_LABEL),
    ]:
        records.append(
            {
                "level": level,
                "n": n,
                "rater_1_mean": _mean_excluding_value(df[col1], non_answer_value),
                "rater_2_mean": _mean_excluding_value(df[col2], non_answer_value),
                "rater_1_value_counts": _value_counts_with_non_answer(df[col1], non_answer_value),
                "rater_2_value_counts": _value_counts_with_non_answer(df[col2], non_answer_value),
                "conflicts": int((df[col1] != df[col2]).sum()),
                "conflict_rate": float((df[col1] != df[col2]).mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def per_dataset_statistics(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset, group in df.groupby("dataset", sort=True):
        for raw_record in agreement_statistics(group).to_dict("records"):
            record = cast(dict[str, Any], raw_record)
            records.append({"dataset": dataset, **record})
    return pd.DataFrame.from_records(records)


def maybe_write_outputs(
    output_prefix: str | None,
    agreement: pd.DataFrame,
    descriptive: pd.DataFrame,
    per_dataset: pd.DataFrame,
) -> None:
    if output_prefix is None:
        return

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(prefix.with_name(f"{prefix.name}_agreement.csv"), index=False)
    descriptive.to_csv(prefix.with_name(f"{prefix.name}_descriptive.csv"), index=False)
    per_dataset.to_csv(prefix.with_name(f"{prefix.name}_per_dataset.csv"), index=False)


def main() -> None:
    args = parse_args()

    rater_1 = load_open_response_annotations(args.rater_1)
    rater_2 = load_open_response_annotations(args.rater_2)
    merged = merge_annotations(rater_1, rater_2, args.id_columns)
    merged = add_binary_labels(merged)
    agreement = agreement_statistics(merged)
    descriptive = descriptive_statistics(merged)
    per_dataset = per_dataset_statistics(merged)

    print("Agreement")
    print(agreement.to_markdown(index=False))
    print()
    print("Descriptive")
    print(descriptive.to_markdown(index=False))
    print()
    print("Per dataset")
    print(per_dataset.to_markdown(index=False))

    maybe_write_outputs(args.output_prefix, agreement, descriptive, per_dataset)


if __name__ == "__main__":
    main()
