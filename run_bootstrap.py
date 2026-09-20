"""Run bootstrap analyses for psychometrics, vignettes, and IssueBench.

Expected upstream steps:
1. Run the psychometric experiments via main.py.
2. Run eval/tidy_df_preprocessing.ipynb to combine results into tidy dataframes.
3. Run the open response judge evaluation via eval/eval_reduction_judge.py.
4. Run the IssueBench judge evaluation for IssueBench bootstrapping.
5. Run this script.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from loguru import logger


if TYPE_CHECKING:
    from llm_audit.eval.issuebench import IssueBenchJudgePredictions
    from llm_audit.eval.open_response import CIAggResults, JudgeEvalPredictions

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "resources/output/bootstrap_results"
DEFAULT_PSYCHOMETRIC_SCORES = REPO_ROOT / "eval/data/tidy/construct_scores_ensemble.csv"
DEFAULT_VIGNETTE_SCORES = REPO_ROOT / "eval/data/tidy/vignette_scores.csv"
DEFAULT_JUDGE_EVAL_RESULTS = [
    REPO_ROOT
    / "resources/output/reduction_eval_datav6-en-1_0-1-42/IssueBench/judge/tiiuae/Falcon-H1R-7b-reprocessed/unknown_model/results.json",
    REPO_ROOT
    / "resources/output/reduction_eval_datav6-en-1_0-1-42/IssueBench/judge/gemma4-31b-it/unknown_model/results.json",
    REPO_ROOT
    / "resources/output/reduction_eval_datav6-en-1_0-1-42/IssueBench/judge/deepseek/deepseek-v4-flash/unknown_model/results.json",
]
DEFAULT_ISSUEBENCH_GENERATIONS = REPO_ROOT / "resources/output/issuebench_gen_jan13_en-en-1_0-1-42"
DEFAULT_ISSUEBENCH_GROUND_TRUTH = (
    REPO_ROOT / "resources/input/datasets/issuebench/ground_truth_data_rebuttal_v3_en.json"
)
DEFAULT_ISSUEBENCH_JUDGE_RESULTS = [
    REPO_ROOT
    / "resources/output/issuebench_judge_jan14_en-en-1_0-1-42/IssueBench/judge/merged_exports/2026-05-22_103509_final_fold-0"
]
DEFAULT_ISSUEBENCH_SELECTOR_JUDGE_RESULTS = (
    REPO_ROOT
    / "resources/output/issuebench_judge_jan25_auth_en-en-1_0-1-42/IssueBench/judge/moonshotai/kimi-k2-thinking"
)

BOOTSTRAPS = ("psychometrics", "vignettes", "issuebench")
ExistingResultsPolicy = Literal["skip", "overwrite"]


@dataclass(frozen=True)
class OutputSet:
    name: str
    paths: tuple[Path, ...]

    def is_complete(self) -> bool:
        return all(path.exists() for path in self.paths)

    def missing_paths(self) -> list[Path]:
        return [path for path in self.paths if not path.exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstraps",
        nargs="+",
        choices=(*BOOTSTRAPS, "all"),
        default=["all"],
        help="Bootstrap analyses to run. Defaults to all.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where bootstrap parquet outputs should be stored.",
    )
    parser.add_argument(
        "--existing-results",
        choices=("skip", "overwrite"),
        default="skip",
        help="Use complete existing outputs to skip computation, or overwrite them.",
    )
    parser.add_argument("--psychometric-scores", type=Path, default=DEFAULT_PSYCHOMETRIC_SCORES)
    parser.add_argument("--vignette-scores", type=Path, default=DEFAULT_VIGNETTE_SCORES)
    parser.add_argument(
        "--judge-eval-results",
        type=Path,
        nargs="+",
        default=DEFAULT_JUDGE_EVAL_RESULTS,
        help="One or more judge evaluation results.json paths. Multiple paths are ensembled conservatively.",
    )
    parser.add_argument("--issuebench-generations", type=Path, default=DEFAULT_ISSUEBENCH_GENERATIONS)
    parser.add_argument("--issuebench-ground-truth", type=Path, default=DEFAULT_ISSUEBENCH_GROUND_TRUTH)
    parser.add_argument(
        "--issuebench-judge-results",
        type=Path,
        nargs="+",
        default=DEFAULT_ISSUEBENCH_JUDGE_RESULTS,
        help="One or more IssueBench judge result JSON files or directories for the target prevalence counts.",
    )
    parser.add_argument(
        "--issuebench-task-filter",
        type=Path,
        default=None,
        help="Optional JSON list of exact IssueBench task strings to keep for target prevalence counts.",
    )
    parser.add_argument(
        "--issuebench-task-exclude",
        type=Path,
        default=None,
        help="Optional JSON list of exact IssueBench task strings to exclude from target prevalence counts.",
    )
    parser.add_argument(
        "--issuebench-judge-eval-results",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional IssueBench judge result JSON files or directories to estimate judge TPR/FPR against "
            "the ground truth. Defaults to --issuebench-judge-results."
        ),
    )
    parser.add_argument(
        "--issuebench-selector-judge-results",
        type=Path,
        default=DEFAULT_ISSUEBENCH_SELECTOR_JUDGE_RESULTS,
        help="IssueBench selector judge outputs used to estimate the predicted_as_autho population rate.",
    )
    parser.add_argument(
        "--issuebench-group-cols",
        nargs="+",
        default=["generating_model_name"],
        help="Columns to preserve as comparison groups in the IssueBench judge-count bootstrap.",
    )
    parser.add_argument("--psychometric-repetitions", type=int, default=1000)
    parser.add_argument("--vignette-repetitions", type=int, default=1000)
    parser.add_argument("--judge-repetitions", type=int, default=1000)
    parser.add_argument("--issuebench-repetitions", type=int, default=1000)
    parser.add_argument("--issuebench-n-jobs", type=int, default=64)
    return parser.parse_args()


def selected_bootstraps(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return BOOTSTRAPS
    return tuple(dict.fromkeys(values))


def ci_output_paths(results_dir: Path, experiment: str) -> tuple[Path, ...]:
    views = ("across_datasets", "per_dataset", "per_dataset_and_factor", "per_factor_across_datasets")
    return tuple(results_dir / f"{experiment}__{view}.parquet" for view in views)


def psychometrics_outputs(results_dir: Path) -> OutputSet:
    return OutputSet(
        "psychometrics",
        (
            *ci_output_paths(results_dir, "closed"),
            *ci_output_paths(results_dir, "open_adjusted"),
        ),
    )


def vignettes_outputs(results_dir: Path) -> OutputSet:
    return OutputSet("vignettes", ci_output_paths(results_dir, "vignettes"))


def issuebench_outputs(results_dir: Path) -> OutputSet:
    return OutputSet(
        "issuebench",
        (
            results_dir / "issuebench_judge_eval__bootstrap_distribution.parquet",
            results_dir / "issuebench_judge_eval__cis.parquet",
            results_dir / "issuebench__point_estimates.parquet",
            results_dir / "issuebench__bootstrap_distribution.parquet",
            results_dir / "issuebench__adjusted_bootstrap_distribution.parquet",
            results_dir / "issuebench__cis.parquet",
        ),
    )


def should_skip(output_set: OutputSet, existing_results: ExistingResultsPolicy) -> bool:
    if existing_results == "overwrite":
        return False
    if output_set.is_complete():
        logger.info(f"Skipping {output_set.name}: all outputs already exist.")
        return True

    missing = ", ".join(str(path) for path in output_set.missing_paths())
    logger.info(f"Running {output_set.name}: missing outputs: {missing}")
    return False


def write_parquet(df: pd.DataFrame, path: Path, existing_results: ExistingResultsPolicy) -> None:
    if existing_results == "skip" and path.exists():
        logger.info(f"Keeping existing result: {path}")
        return
    df.to_parquet(path)
    logger.info(f"Wrote {path}")


def write_ci_results(
    results: CIAggResults,
    results_dir: Path,
    experiment: str,
    existing_results: ExistingResultsPolicy,
) -> None:
    for view in ("across_datasets", "per_dataset", "per_dataset_and_factor", "per_factor_across_datasets"):
        view_results = getattr(results, view)
        write_parquet(view_results, results_dir / f"{experiment}__{view}.parquet", existing_results)


def judge_name_from_path(path: Path) -> str:
    try:
        return path.parts[-3]
    except IndexError as exc:
        raise ValueError(f"Could not derive judge name from path: {path}") from exc


def create_conservative_judge_ensemble(
    judge_dfs: dict[str, JudgeEvalPredictions],
    models: list[str],
    id_cols: tuple[str, ...] = ("dataset", "question_id", "open_response"),
) -> JudgeEvalPredictions:
    from llm_audit.eval.open_response import JudgeEvalPredictions

    missing = [model for model in models if model not in judge_dfs]
    if missing:
        raise ValueError(f"Missing models in judge_dfs: {missing}")

    parts: list[pd.DataFrame] = []
    for model_name in models:
        df = judge_dfs[model_name].data
        if df is None:
            raise ValueError(f"Judge predictions for {model_name} were not loaded.")

        required_cols = set(id_cols) | {"pred_auth", "pred_normalized_score"}
        missing_cols = sorted(required_cols - set(df.columns))
        if missing_cols:
            raise ValueError(f"Judge predictions for {model_name} are missing columns: {missing_cols}")

        part = df.loc[:, [*id_cols, "pred_auth", "pred_normalized_score"]].rename(
            columns={
                "pred_auth": model_name,
                "pred_normalized_score": f"{model_name}_score",
            }
        )
        parts.append(part)

    merged_df = parts[0]
    for part in parts[1:]:
        merged_df = merged_df.merge(part, on=list(id_cols))

    score_cols = [f"{model}_score" for model in models]
    valid_ensemble_rows = merged_df[models + score_cols].notna().all(axis=1)
    merged_df["ensemble_all"] = np.where(
        valid_ensemble_rows,
        merged_df[models].astype(bool).all(axis=1).astype(float),
        np.nan,
    )
    merged_df["ensemble_all_score"] = np.where(
        valid_ensemble_rows,
        merged_df[score_cols].min(axis=1).astype(float),
        np.nan,
    )

    base_df = judge_dfs[models[0]].data
    if base_df is None:
        raise ValueError(f"Judge predictions for {models[0]} were not loaded.")

    ensemble_df = base_df.copy().merge(merged_df, on=list(id_cols))
    ensemble_df["pred_auth"] = ensemble_df["ensemble_all"]
    ensemble_df["pred_normalized_score"] = ensemble_df["ensemble_all_score"]

    return JudgeEvalPredictions(Path(), data=ensemble_df)


def load_judge_eval_predictions(paths: list[Path]) -> JudgeEvalPredictions:
    from llm_audit.eval.open_response import JudgeEvalPredictions

    if not paths:
        raise ValueError("At least one judge evaluation results path is required.")

    judge_names = [judge_name_from_path(path) for path in paths]
    duplicate_names = sorted({name for name in judge_names if judge_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Judge names derived from paths must be unique. Duplicates: {duplicate_names}")

    judge_dfs = {judge_name: JudgeEvalPredictions(path=path) for judge_name, path in zip(judge_names, paths)}

    if len(paths) == 1:
        logger.info(f"Using single judge eval results: {judge_names[0]} ({paths[0]})")
        return judge_dfs[judge_names[0]]

    logger.info(f"Creating conservative judge ensemble from: {', '.join(judge_names)}")
    return create_conservative_judge_ensemble(judge_dfs, judge_names)


def filter_issuebench_judge_predictions_by_task(
    judge_preds: IssueBenchJudgePredictions,
    task_filter_path: Path | None,
    *,
    exclude: bool = False,
) -> None:
    if task_filter_path is None:
        return

    data = getattr(judge_preds, "data", None)
    if data is None:
        raise ValueError("IssueBench judge predictions were not loaded.")
    if "task" not in data.columns:
        raise ValueError("IssueBench judge data must contain a task column to use task filtering.")

    with task_filter_path.open("r", encoding="utf-8") as f:
        selected_tasks_raw = json.load(f)
    if not isinstance(selected_tasks_raw, list) or not all(isinstance(task, str) for task in selected_tasks_raw):
        raise ValueError(f"Expected {task_filter_path} to contain a JSON list of task strings.")

    selected_tasks = set(selected_tasks_raw)
    original_rows = len(data)
    matched_tasks = set(data["task"].dropna()) & selected_tasks
    task_matches = data["task"].isin(selected_tasks)
    if exclude:
        filtered = data.loc[~task_matches].reset_index(drop=True)
        action = "Excluded"
        detail = f"{original_rows - len(filtered)}/{original_rows} rows removed"
    else:
        filtered = data.loc[task_matches].reset_index(drop=True)
        action = "Filtered"
        detail = f"{len(filtered)}/{original_rows} rows kept"

    logger.info(
        f"{action} IssueBench target judge rows by task: "
        f"{detail}; {len(matched_tasks)}/{len(selected_tasks)} selected tasks matched."
    )
    if filtered.empty:
        action_description = "excluded all" if exclude else "matched zero"
        raise ValueError(f"IssueBench task filter {task_filter_path} {action_description} target judge rows.")

    judge_preds.data = filtered


def run_psychometrics(args: argparse.Namespace) -> None:
    output_set = psychometrics_outputs(args.results_dir)
    if should_skip(output_set, args.existing_results):
        return

    from llm_audit.eval.open_response import (
        AuthoritarianResponseRateBootstrap,
        ConfidenceIntervalResult,
        JudgePerformanceBootstrap,
        PsychometricPredictions,
    )

    logger.info("Bootstrapping judge eval statistics")
    judge_preds = load_judge_eval_predictions(args.judge_eval_results)
    judge_bs = JudgePerformanceBootstrap(judge_preds, args.judge_repetitions)
    judge_bs.bootstrap()
    judge_cis = judge_bs.cis()
    judge_distribution = judge_bs.bootstrap_distribution
    if judge_distribution is None:
        raise ValueError("Judge bootstrap distribution was not computed.")

    write_parquet(
        judge_distribution,
        args.results_dir / "judge_eval__bootstrap_distribution.parquet",
        args.existing_results,
    )
    write_parquet(judge_cis, args.results_dir / "judge_eval__cis.parquet", args.existing_results)

    logger.info("Bootstrapping psychometric results")
    psychometric_preds = PsychometricPredictions(args.psychometric_scores, repetition_col="repetition_id")
    psychometric_bs = AuthoritarianResponseRateBootstrap(psychometric_preds, args.psychometric_repetitions)
    psychometric_bs.bootstrap()
    psychometric_bs.bootstrap_adjusted(judge_distribution)
    results = psychometric_bs.cis(alpha=0.05, tpr_fpr_ci_df=judge_cis)
    if not isinstance(results, ConfidenceIntervalResult):
        raise TypeError("Expected psychometric bootstrap to return ConfidenceIntervalResult.")
    if results.open_adjusted is None:
        raise ValueError("Adjusted psychometric results were not computed.")

    write_ci_results(results.closed, args.results_dir, "closed", args.existing_results)
    write_ci_results(results.open_adjusted, args.results_dir, "open_adjusted", args.existing_results)


def run_vignettes(args: argparse.Namespace) -> None:
    output_set = vignettes_outputs(args.results_dir)
    if should_skip(output_set, args.existing_results):
        return

    from llm_audit.eval.open_response import AuthoritarianResponseRateBootstrap, CIAggResults, VignettePredictions

    logger.info("Bootstrapping vignette results")
    preds = VignettePredictions(args.vignette_scores, repetition_col="repetition_id")
    bs = AuthoritarianResponseRateBootstrap(preds, args.vignette_repetitions)
    bs.bootstrap()
    results = bs.cis()
    if not isinstance(results, CIAggResults):
        raise TypeError("Expected vignette bootstrap to return CIAggResults.")

    write_ci_results(results, args.results_dir, "vignettes", args.existing_results)


def run_issuebench(args: argparse.Namespace) -> None:
    output_set = issuebench_outputs(args.results_dir)
    if should_skip(output_set, args.existing_results):
        return

    from llm_audit.eval.issuebench import (
        IssueBenchJudgeCountBootstrap,
        IssueBenchJudgePerformanceBootstrap,
        IssueBenchJudgePredictions,
    )

    if args.issuebench_task_filter is not None and args.issuebench_task_exclude is not None:
        raise ValueError("Use only one of --issuebench-task-filter or --issuebench-task-exclude.")

    logger.info("Loading IssueBench target judge results")
    judge_preds = IssueBenchJudgePredictions(args.issuebench_judge_results)
    filter_issuebench_judge_predictions_by_task(judge_preds, args.issuebench_task_filter)
    filter_issuebench_judge_predictions_by_task(judge_preds, args.issuebench_task_exclude, exclude=True)

    judge_eval_results = args.issuebench_judge_eval_results or args.issuebench_judge_results
    logger.info("Loading IssueBench judge-evaluation results")
    judge_eval_preds = IssueBenchJudgePredictions(judge_eval_results)

    logger.info("Loading IssueBench selector judge results")
    selector_preds = IssueBenchJudgePredictions(args.issuebench_selector_judge_results)

    logger.info("Bootstrapping weighted IssueBench judge-performance statistics")
    judge_perf_bs = IssueBenchJudgePerformanceBootstrap(
        judge_eval_preds,
        args.issuebench_ground_truth,
        selector_preds,
        n_repetitions=args.issuebench_repetitions,
    )
    judge_perf_bs.bootstrap()
    judge_perf_cis = judge_perf_bs.cis()
    judge_perf_distribution = judge_perf_bs.bootstrap_distribution
    if judge_perf_distribution is None:
        raise ValueError("IssueBench judge-performance bootstrap distribution was not computed.")

    write_parquet(
        judge_perf_distribution,
        args.results_dir / "issuebench_judge_eval__bootstrap_distribution.parquet",
        args.existing_results,
    )
    write_parquet(judge_perf_cis, args.results_dir / "issuebench_judge_eval__cis.parquet", args.existing_results)

    bs = IssueBenchJudgeCountBootstrap(
        judge_preds,
        group_cols=tuple(args.issuebench_group_cols),
        n_repetitions=args.issuebench_repetitions,
    )
    logger.info("Computing IssueBench judge-count point estimates")
    bs.point_estimate()
    logger.info("Bootstrapping IssueBench judge counts")
    bs.bootstrap()
    logger.info("Bootstrapping adjusted IssueBench judge counts")
    adjusted_distribution = bs.bootstrap_adjusted(judge_perf_distribution)
    cis = bs.cis()

    write_parquet(cis.point_estimates, args.results_dir / "issuebench__point_estimates.parquet", args.existing_results)
    write_parquet(
        cis.bootstrap_distribution,
        args.results_dir / "issuebench__bootstrap_distribution.parquet",
        args.existing_results,
    )
    write_parquet(
        adjusted_distribution,
        args.results_dir / "issuebench__adjusted_bootstrap_distribution.parquet",
        args.existing_results,
    )
    write_parquet(cis.intervals, args.results_dir / "issuebench__cis.parquet", args.existing_results)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        "psychometrics": run_psychometrics,
        "vignettes": run_vignettes,
        "issuebench": run_issuebench,
    }
    for bootstrap in selected_bootstraps(args.bootstraps):
        runners[bootstrap](args)


if __name__ == "__main__":
    main()
