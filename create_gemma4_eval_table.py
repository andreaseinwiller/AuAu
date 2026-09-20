import argparse

import pandas as pd

from pathlib import Path
from typing import cast


DEFAULT_INPUT = Path(
    "resources/output/bootstrap_results/default_issuebench_gemma4sft/issuebench_judge_eval__cis.parquet"
)

DIMENSION_ORDER = ["aggression", "conventionalism", "submission", "any"]
DIMENSION_LABELS = {
    "aggression": "Aggression",
    "conventionalism": "Conventionalism",
    "submission": "Submission",
    "any": "Any",
}

METRIC_ORDER = ["tpr", "fpr"]
METRIC_LABELS = {
    "tpr": "TPR",
    "fpr": "FPR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a LaTeX table for Gemma 4 IssueBench judge TPR/FPR.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to issuebench_judge_eval__cis.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table. Prints to stdout by default.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places for point estimates.",
    )
    return parser.parse_args()


def format_estimate(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "--"
    return f"${value:.{decimals}f}$"


def create_latex_table(df: pd.DataFrame, decimals: int = 2) -> str:
    required_columns = {"dimension", "metric", "point_estimate"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {sorted(missing_columns)}")

    filtered = df.loc[df["dimension"].isin(DIMENSION_ORDER) & df["metric"].isin(METRIC_ORDER)]
    table = filtered.pivot(
        index="metric",
        columns="dimension",
        values="point_estimate",
    )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Metric & " + " & ".join(DIMENSION_LABELS[dimension] for dimension in DIMENSION_ORDER) + r" \\",
        r"\midrule",
    ]

    for metric in METRIC_ORDER:
        row = [METRIC_LABELS[metric]]
        for dimension in DIMENSION_ORDER:
            value = table.loc[metric, dimension] if dimension in table.columns else pd.NA
            row.append(format_estimate(cast(float, value), decimals))
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    latex = create_latex_table(df, decimals=args.decimals)

    if args.output is None:
        print(latex)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex + "\n")


if __name__ == "__main__":
    main()
