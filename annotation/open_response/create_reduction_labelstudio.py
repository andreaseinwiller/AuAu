import argparse
import json
import random

import pandas as pd

from typing import Hashable, Any, cast
from pathlib import Path

from llm_audit.datasets import DATASETS

# Datasets where the negative pole comes first in the raw scale definition,
# so options_canonical is the same as options (no reversal needed).
DATASETS_WITH_NEGATIVE_OPTIONS_FIRST = {
    "ASC",
    "BDW",
    "CSM",
    "KSA3",
    "LAS",
    "PI",
    "PISD",
    "RWA",
    "RWA3D",
    "SDO7",
}


def build_datasets_df() -> pd.DataFrame:
    """
    Build a dataframe with one row per dataset containing its Likert scale info:
      - options          : raw ordered list of {"value": "key = description"} dicts
      - disagree_to_agree: bool — whether the scale runs from disagree→agree
      - neutral          : the agreement discriminator threshold
      - options_canonical: options ordered from negative->positive pole consistently
    """
    records = []
    for dataset_name, cls in DATASETS.items():
        ds = cls(reverse_scale=False)
        options = [{"value": f"{key} = {description}"} for key, description in ds.item_label_map["en"].items()]
        options_canonical = options if dataset_name in DATASETS_WITH_NEGATIVE_OPTIONS_FIRST else list(reversed(options))
        records.append(
            {
                "dataset": dataset_name,
                "options": options,
                "options_canonical": options_canonical,
                "disagree_to_agree": ds.disagree_to_agree,
                "neutral": ds.get_agreement_discriminator_threshold(),
            }
        )
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample open_question responses by auth score and produce Label Studio annotation files."
    )
    parser.add_argument(
        "--scores-csv",
        default="/root/llm-audit/eval/data/tidy/construct_scores.csv",
        help="Path to construct_scores.csv",
    )
    parser.add_argument(
        "--data-dir",
        default="resources/output/final-en-1_0-10-42",
        help="Root directory that contains dataset/experiment_type/statement_id/model subdirs",
    )
    parser.add_argument(
        "--output-auth",
        default="labelstudio_auth1.json",
        help="Output file for auth=1 samples",
    )
    parser.add_argument(
        "--output-non-auth",
        default="labelstudio_auth0.json",
        help="Output file for auth=0 samples",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=50,
        type=int,
        help="Number of samples per output file",
    )
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def model_name_to_path(model: str) -> Path:
    """
    Convert a fully qualified model name like 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    into a two-level path 'Qwen/Qwen3-30B-A3B-Instruct-2507'.
    Models without a '/' are placed directly as a single path component.
    """
    return Path(model)


def find_json_file(
    data_dir: Path,
    dataset: str,
    experiment_type: str,
    statement_id: int | str,
    model: str,
    repetition_id: int | str,
) -> Path | None:
    """
    Build the expected path to results.json for a given row.

    Directory layout:
      <data_dir>/<dataset>/<experiment_type>/<statement_id>/<model_path>/results.json

    The repetition_id is encoded inside the JSON file itself (as array index),
    not as a directory level — but we include it here for callers that need it.
    """
    model_path = model_name_to_path(model)
    candidate = data_dir / dataset / experiment_type / str(statement_id) / model_path / "results.json"
    return candidate if candidate.exists() else None


def extract_record(
    json_path: Path,
    repetition_id: int,
    row: "pd.Series",
) -> dict[str, Any] | None:
    """
    Load results.json and extract the entry at index `repetition_id`.
    Returns a flat dict ready for Label Studio, or None on failure.
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: could not read {json_path}: {exc}")
        return None

    if not isinstance(records, list) or repetition_id >= len(records):
        print(
            f"  WARNING: repetition_id={repetition_id} out of range for {json_path} "
            f"(len={len(records) if isinstance(records, list) else 'N/A'})"
        )
        return None

    entry = records[repetition_id]

    open_response: str = entry.get("open_response", "")
    meta: dict[str, object] = entry.get("open_response_meta", {})
    user_prompts: list[str] = cast(list[str], meta.get("user_prompts", []))
    task: str = user_prompts[0] if user_prompts else ""

    return {
        # Provenance / metadata useful for annotators
        "model": row["model"],
        "dataset": row["dataset"],
        "statement_id": int(row["statement_id"]),
        "repetition_id": int(row["repetition_id"]),
        "experiment_type": row["experiment_type"],
        "language": row["language"],
        "score": float(row["score"]),
        "refusal": bool(row["refusal"]),
        "auth": float(row["auth"]) if pd.notna(row["auth"]) else None,
        # Likert scale info for this dataset
        "options": row["options"],
        "options_canonical": row["options_canonical"],
        "disagree_to_agree": bool(row["disagree_to_agree"]),
        "neutral": row["neutral"],
        # The fields annotators actually need
        "task": task,
        "text": open_response,
        # Full meta in case the annotation tool needs it
        "open_response_meta": meta,
    }


def build_labelstudio_sample(record: dict[str, object]) -> dict[str, object]:
    """Wrap a record in the Label Studio {"data": {...}} envelope."""
    return {"data": record}


def sample_and_build(
    df: pd.DataFrame,
    data_dir: Path,
    n: int,
    rng: random.Random,
    label: str,
) -> list[dict[str, object]]:
    """
    Randomly sample `n` rows from `df`, resolve each to its JSON file,
    extract the relevant record, and return a list of Label Studio dicts.

    Sampling is done with replacement from successfully resolved rows so
    that the output always contains exactly `n` items when possible.
    """
    rows = df.sample(frac=1, random_state=rng.randint(0, 2**31)).reset_index(drop=True)

    results: list[dict[str, object]] = []
    missing: list[Hashable] = []  # row indices we couldn't resolve

    for idx, row in rows.iterrows():
        if len(results) >= n:
            break

        json_path = find_json_file(
            data_dir=data_dir,
            dataset=row["dataset"],
            experiment_type=row["experiment_type"],
            statement_id=row["statement_id"],
            model=row["model"],
            repetition_id=row["repetition_id"],
        )

        if json_path is None:
            expected = (
                data_dir
                / row["dataset"]
                / row["experiment_type"]
                / str(row["statement_id"])
                / model_name_to_path(row["model"])
                / "results.json"
            )
            print(f"  [{label}] MISSING file: {expected}")
            missing.append(int(cast(Any, idx)))
            continue

        record = extract_record(json_path, int(row["repetition_id"]), row)
        if record is None:
            continue

        results.append(build_labelstudio_sample(record))

    if len(results) < n:
        print(
            f"  [{label}] WARNING: only {len(results)}/{n} samples could be resolved "
            f"({len(missing)} files missing / unreadable)."
        )
    else:
        print(f"  [{label}] Successfully collected {len(results)} samples.")
    return results


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Load and filter the scores dataframe
    print(f"Loading scores from: {args.scores_csv}")
    df = pd.read_csv(args.scores_csv)
    df = df.loc[(df["experiment_type"] == "open_question") & (df["language"] == "en")].copy()
    print(f"  Rows after filtering to open_question / en: {len(df)}")

    # Merge Likert scale info
    print("Building dataset scale info from DATASETS registry …")
    ds_info = build_datasets_df()
    df = df.merge(ds_info, on="dataset", how="left")
    missing_scale = df["options"].isna().sum()
    if missing_scale:
        print(f"  WARNING: {missing_scale} rows could not be matched to a dataset scale.")

    # Drop rows where auth is NaN - we can't assign them to either group
    df = df.dropna(subset=["auth"])
    print(f"  Rows with non-null auth: {len(df)}")

    df_auth1 = df.loc[df["auth"] == 1.0]
    df_auth0 = df.loc[df["auth"] == 0.0]
    print(f"  auth=1 rows: {len(df_auth1)},  auth=0 rows: {len(df_auth0)}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Sample and build output for auth=1
    print(f"\nSampling {args.num_samples} auth=1 records …")
    auth1_samples = sample_and_build(df_auth1, data_dir, args.num_samples, rng, "auth=1")
    with open(args.output_auth, "w", encoding="utf-8") as f:
        json.dump(auth1_samples, f, indent=2, ensure_ascii=False)
    print(f"  Written to: {args.output_auth}")

    # Sample and build output for auth=0
    print(f"\nSampling {args.num_samples} auth=0 records …")
    auth0_samples = sample_and_build(df_auth0, data_dir, args.num_samples, rng, "auth=0")
    with open(args.output_non_auth, "w", encoding="utf-8") as f:
        json.dump(auth0_samples, f, indent=2, ensure_ascii=False)
    print(f"  Written to: {args.output_non_auth}")


if __name__ == "__main__":
    main()
