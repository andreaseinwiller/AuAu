import sys
import importlib.util

import numpy as np
import pandas as pd

from typing import Any, TypedDict
from pathlib import Path

from llm_audit.eval.issuebench import (
    IssueBenchClassifierPrevalenceBootstrap,
    IssueBenchJudgeCountBootstrap,
    IssueBenchJudgePerformanceBootstrap,
    IssueBenchJudgePredictions,
    choose_threshold_by_j_statistic,
    evaluate_classifier_predictions,
    load_issuebench_ground_truth,
    parse_issuebench_judge_response,
    probabilistic_count,
    repeated_stratified_cv_predictions,
)


def _load_run_bootstrap_filter() -> Any:
    module_path = Path(__file__).resolve().parents[1] / "run_bootstrap.py"
    spec = importlib.util.spec_from_file_location("run_bootstrap", module_path)
    assert spec is not None
    assert spec.loader is not None
    run_bootstrap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = run_bootstrap
    spec.loader.exec_module(run_bootstrap)
    return run_bootstrap.filter_issuebench_judge_predictions_by_task


def test_probabilistic_count_adjusts_with_spa() -> None:
    prevalence = probabilistic_count(
        [0.7, 0.9],
        {"avg_proba_for_positives": 0.9, "avg_proba_for_negatives": 0.1},
    )

    assert prevalence == 0.875


def test_evaluate_classifier_predictions_returns_expected_rates() -> None:
    preds = pd.DataFrame({"label": [1, 1, 0, 0], "prob": [0.9, 0.2, 0.8, 0.1]})

    stats = evaluate_classifier_predictions(preds, threshold=0.5)

    assert stats["tpr"] == 0.5
    assert stats["fpr"] == 0.5
    assert np.isclose(stats["observed_prevalence"], 0.5)


def test_threshold_selection_constrains_fpr_and_maximizes_j_statistic() -> None:
    preds = pd.DataFrame(
        {
            "label": [1, 1, 1] + [0] * 20,
            "prob": [0.9, 0.8, 0.6] + [0.65] + [0.5 - i * 0.01 for i in range(19)],
        }
    )

    threshold_stats = choose_threshold_by_j_statistic(preds, max_fpr=0.053)
    stats = evaluate_classifier_predictions(preds, threshold_max_fpr=0.053)

    assert np.isclose(threshold_stats["threshold"], 0.6)
    assert np.isclose(stats["threshold"], 0.6)
    assert stats["fpr"] < 0.053
    assert np.isclose(stats["tpr"], 1.0)
    assert np.isclose(stats["j_statistic"], 0.95)


def test_grouped_target_bootstrap_preserves_groups() -> None:
    labelled = pd.DataFrame(
        {
            "text": [
                "obey the leader",
                "respect authority",
                "open discussion",
                "free inquiry",
            ],
            "label_combined": [1.0, 1.0, -1.0, -1.0],
        }
    )
    target = pd.DataFrame(
        {
            "text": ["obey authority", "free inquiry", "respect authority", "open discussion"],
            "model": ["a", "a", "b", "b"],
        }
    )
    runner = IssueBenchClassifierPrevalenceBootstrap(
        labelled,
        target,
        group_cols=("model",),
        n_repetitions=2,
        n_splits=2,
        random_state=11,
    )

    boot_target = runner._bootstrap_target_data(np.random.default_rng(123))

    assert set(boot_target["model"]) == {"a", "b"}
    assert boot_target.groupby("model").size().to_dict() == {"a": 2, "b": 2}


def test_grouped_prediction_bootstrap_preserves_groups() -> None:
    labelled = pd.DataFrame(
        {
            "text": [
                "obey the leader",
                "respect authority",
                "open discussion",
                "free inquiry",
            ],
            "label_combined": [1.0, 1.0, -1.0, -1.0],
        }
    )
    target = pd.DataFrame(
        {
            "text": ["obey authority", "free inquiry", "respect authority", "open discussion"],
            "model": ["a", "a", "b", "b"],
        }
    )
    runner = IssueBenchClassifierPrevalenceBootstrap(labelled, target, group_cols=("model",), n_splits=2)
    target_predictions = pd.DataFrame({"model": ["a", "a", "b", "b"], "prob": [0.1, 0.2, 0.8, 0.9]})

    boot_predictions = runner._bootstrap_target_predictions(target_predictions, np.random.default_rng(123))

    assert set(boot_predictions["model"]) == {"a", "b"}
    assert boot_predictions.groupby("model").size().to_dict() == {"a": 2, "b": 2}
    assert set(boot_predictions["prob"]).issubset(set(target_predictions["prob"]))


def test_grouped_cv_keeps_bootstrap_duplicates_in_same_fold() -> None:
    texts = [
        "positive one",
        "positive one",
        "positive two",
        "positive two",
        "positive three",
        "positive three",
        "negative one",
        "negative one",
        "negative two",
        "negative two",
        "negative three",
        "negative three",
    ]
    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    groups = ["p1", "p1", "p2", "p2", "p3", "p3", "n1", "n1", "n2", "n2", "n3", "n3"]

    oof = repeated_stratified_cv_predictions(
        texts,
        labels,
        groups=groups,
        n_splits=3,
        n_repeats=2,
        random_state=7,
    )

    assigned_folds = oof.groupby(["repeat", "row_id"])["fold"].first().reset_index()
    assigned_folds["group"] = assigned_folds["row_id"].map(dict(enumerate(groups)))

    fold_counts = assigned_folds.groupby(["repeat", "group"])["fold"].nunique()
    assert fold_counts.eq(1).all()


def test_point_estimate_includes_regular_and_calibrated_spa_methods() -> None:
    labelled = pd.DataFrame(
        {
            "text": [
                "obey the leader",
                "respect authority",
                "follow orders",
                "strict hierarchy",
                "open discussion",
                "free inquiry",
                "question leaders",
                "shared decisions",
            ],
            "label_combined": [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    target = pd.DataFrame(
        {
            "text": ["obey authority", "free inquiry", "respect authority", "open discussion"],
            "model": ["a", "a", "b", "b"],
        }
    )
    runner = IssueBenchClassifierPrevalenceBootstrap(
        labelled,
        target,
        group_cols=("model",),
        n_splits=2,
        calibration_cv=2,
        threshold_max_fpr=None,
        quantification_methods=("sgd_spa", "calibrated_sgd_spa"),
    )

    point_estimates = runner.point_estimate()

    assert set(point_estimates["quantification_method"]) == {"sgd_spa", "calibrated_sgd_spa"}
    assert point_estimates.groupby("quantification_method").size().to_dict() == {
        "calibrated_sgd_spa": 2,
        "sgd_spa": 2,
    }


class _BootstrapKwargs(TypedDict):
    group_cols: tuple[str, ...]
    n_repetitions: int
    n_splits: int
    random_state: int
    quantification_methods: tuple[str, ...]


def test_parallel_bootstrap_matches_serial_results() -> None:
    labelled = pd.DataFrame(
        {
            "text": [
                "obey the leader",
                "respect authority",
                "follow orders",
                "strict hierarchy",
                "discipline first",
                "loyalty above all",
                "open discussion",
                "free inquiry",
                "question leaders",
                "shared decisions",
                "debate policy",
                "independent judgment",
            ],
            "label_combined": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    target = pd.DataFrame(
        {
            "text": ["obey authority", "free inquiry", "respect authority", "open discussion"],
            "model": ["a", "a", "b", "b"],
        }
    )
    kwargs: _BootstrapKwargs = {
        "group_cols": ("model",),
        "n_repetitions": 2,
        "n_splits": 2,
        "random_state": 11,
        "quantification_methods": ("sgd_spa",),
    }
    serial = IssueBenchClassifierPrevalenceBootstrap(labelled, target, n_jobs=1, **kwargs).bootstrap()
    parallel = IssueBenchClassifierPrevalenceBootstrap(labelled, target, n_jobs=2, **kwargs).bootstrap()

    pd.testing.assert_frame_equal(serial, parallel)


class CountingIssueBenchBootstrap(IssueBenchClassifierPrevalenceBootstrap):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fit_predict_calls = 0

    def _fit_predict_target(
        self, labelled_data: pd.DataFrame, target_data: pd.DataFrame, random_state: int, method: str = "sgd_spa"
    ) -> pd.DataFrame:
        self.fit_predict_calls += 1
        preds = (
            target_data.loc[:, list(self.group_cols)].copy()
            if self.group_cols
            else pd.DataFrame(index=target_data.index)
        )
        preds["prob"] = [0.1, 0.2, 0.8, 0.9][: len(target_data)]
        return preds

    def _eval_stats(self, labelled_data: pd.DataFrame, random_state: int, method: str = "sgd_spa") -> dict[str, float]:
        return {
            "tpr": 0.9,
            "fpr": 0.1,
            "auc": 1.0,
            "brier": 0.0,
            "observed_prevalence": 0.5,
            "mean_predicted_prevalence": 0.5,
            "calibration_bias": 0.0,
            "avg_proba_for_positives": 0.9,
            "avg_proba_for_negatives": 0.1,
        }


def test_non_classifier_bootstrap_predicts_target_once() -> None:
    labelled = pd.DataFrame({"text": ["yes", "no"], "label_combined": [1.0, -1.0]})
    target = pd.DataFrame({"text": ["a", "b", "c", "d"], "model": ["a", "a", "b", "b"]})
    runner = CountingIssueBenchBootstrap(
        labelled,
        target,
        group_cols=("model",),
        n_repetitions=3,
        n_splits=2,
        bootstrap_classifier=False,
        quantification_methods=("sgd_spa",),
    )

    runner.bootstrap()

    assert runner.fit_predict_calls == 1


def test_point_estimate_supports_quapy_direct_quantifiers() -> None:
    labelled = pd.DataFrame(
        {
            "text": [
                "obey the leader",
                "respect authority",
                "follow orders",
                "strict hierarchy",
                "discipline and loyalty",
                "submit to command",
                "open discussion",
                "free inquiry",
                "question leaders",
                "shared decisions",
                "civil disagreement",
                "independent judgment",
            ],
            "label_combined": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    target = pd.DataFrame(
        {
            "text": [
                "obey authority",
                "follow command",
                "free inquiry",
                "open discussion",
                "respect hierarchy",
                "question authority",
            ],
            "model": ["a", "a", "a", "b", "b", "b"],
        }
    )
    runner = IssueBenchClassifierPrevalenceBootstrap(
        labelled,
        target,
        group_cols=("model",),
        n_splits=2,
        threshold_max_fpr=None,
        quantification_methods=("hdy", "pacc"),
    )

    point_estimates = runner.point_estimate()

    assert set(point_estimates["quantification_method"]) == {"hdy", "pacc"}
    assert point_estimates.groupby("quantification_method").size().to_dict() == {"hdy": 2, "pacc": 2}
    assert point_estimates["prevalence"].between(0, 1).all()
    assert point_estimates["raw_prevalence"].between(0, 1).all()
    assert {"auc", "brier", "avg_proba_for_positives", "avg_proba_for_negatives"}.issubset(point_estimates.columns)


def test_parse_issuebench_judge_response_supports_prefixed_and_unprefixed_keys() -> None:
    prefixed = parse_issuebench_judge_response(
        '{"label_aggression": "yes", "label_submission": "no", "label_conventionalism": "no", "label_refusal": "no"}'
    )
    unprefixed = parse_issuebench_judge_response(
        '{"aggression": "no", "submission": "yes", "conventionalism": "no", "refusal": "no"}'
    )

    assert prefixed == {
        "aggression": 1.0,
        "submission": -1.0,
        "conventionalism": -1.0,
        "refusal": -1.0,
    }
    assert unprefixed == {
        "aggression": -1.0,
        "submission": 1.0,
        "conventionalism": -1.0,
        "refusal": -1.0,
    }


def test_parse_issuebench_judge_response_extracts_embedded_json() -> None:
    parsed = parse_issuebench_judge_response(
        'The answer is:\n{"aggression": "no", "submission": "no", "conventionalism": "yes", "refusal": "no"}\nDone.'
    )

    assert parsed["aggression"] == -1.0
    assert parsed["submission"] == -1.0
    assert parsed["conventionalism"] == 1.0
    assert parsed["refusal"] == -1.0


def test_issuebench_judge_predictions_loads_files_and_adds_any(tmp_path: Path) -> None:
    result_path = tmp_path / "results.json"
    result_path.write_text(
        '[{"model_name": "judge", "generating_model_name": "generator-a", '
        '"user_prompts": ["judge prompt"], "generating_user_prompts": ["task"], '
        '"response": "{\\"aggression\\": \\"no\\", \\"submission\\": \\"yes\\", '
        '\\"conventionalism\\": \\"no\\", \\"refusal\\": \\"no\\"}"}]'
    )

    preds = IssueBenchJudgePredictions(tmp_path)

    assert preds.data is not None
    assert len(preds.data) == 1
    row = preds.data.iloc[0]
    assert row["judge_model"] == "judge"
    assert row["generating_model_name"] == "generator-a"
    assert row["prompt"] == "judge prompt"
    assert row["any"] == 1.0


def test_filter_issuebench_judge_predictions_by_task_keeps_matching_tasks(tmp_path: Path) -> None:
    preds = IssueBenchJudgePredictions(
        data=pd.DataFrame(
            {
                "generating_model_name": ["generator-a", "generator-a", "generator-a"],
                "task": ["keep", "drop", "also keep"],
                "response": [
                    _issuebench_response(aggression="yes"),
                    _issuebench_response(submission="yes"),
                    _issuebench_response(conventionalism="yes"),
                ],
            }
        )
    )
    task_filter_path = tmp_path / "task_filter.json"
    task_filter_path.write_text('["keep", "also keep", "missing"]')

    filter_issuebench_judge_predictions_by_task = _load_run_bootstrap_filter()

    filter_issuebench_judge_predictions_by_task(preds, task_filter_path)

    assert preds.data is not None
    assert preds.data["task"].tolist() == ["keep", "also keep"]


def test_filter_issuebench_judge_predictions_by_task_excludes_matching_tasks(tmp_path: Path) -> None:
    preds = IssueBenchJudgePredictions(
        data=pd.DataFrame(
            {
                "generating_model_name": ["generator-a", "generator-a", "generator-a"],
                "task": ["keep", "exclude", "also keep"],
                "response": [
                    _issuebench_response(aggression="yes"),
                    _issuebench_response(submission="yes"),
                    _issuebench_response(conventionalism="yes"),
                ],
            }
        )
    )
    task_filter_path = tmp_path / "task_filter.json"
    task_filter_path.write_text('["exclude", "missing"]')

    filter_issuebench_judge_predictions_by_task = _load_run_bootstrap_filter()

    filter_issuebench_judge_predictions_by_task(preds, task_filter_path, exclude=True)

    assert preds.data is not None
    assert preds.data["task"].tolist() == ["keep", "also keep"]


def test_filter_issuebench_judge_predictions_by_task_rejects_zero_matches(tmp_path: Path) -> None:
    preds = IssueBenchJudgePredictions(
        data=pd.DataFrame(
            {
                "generating_model_name": ["generator-a"],
                "task": ["drop"],
                "response": [_issuebench_response()],
            }
        )
    )
    task_filter_path = tmp_path / "task_filter.json"
    task_filter_path.write_text('["missing"]')

    filter_issuebench_judge_predictions_by_task = _load_run_bootstrap_filter()

    try:
        filter_issuebench_judge_predictions_by_task(preds, task_filter_path)
    except ValueError as exc:
        assert "matched zero target judge rows" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a task filter with zero matching rows.")


def test_issuebench_judge_count_excludes_refusals_from_auth_dimensions() -> None:
    data = pd.DataFrame(
        {
            "generating_model_name": ["generator-a"] * 3,
            "response": [
                '{"aggression": "yes", "submission": "no", "conventionalism": "no", "refusal": "no"}',
                '{"aggression": "yes", "submission": "yes", "conventionalism": "yes", "refusal": "yes"}',
                '{"aggression": "no", "submission": "no", "conventionalism": "no", "refusal": "no"}',
            ],
        }
    )
    runner = IssueBenchJudgeCountBootstrap(data, n_repetitions=2)

    point = runner.point_estimate()
    aggression = point.loc[point["dimension"] == "aggression"].iloc[0]
    refusal = point.loc[point["dimension"] == "refusal"].iloc[0]
    any_row = point.loc[point["dimension"] == "any"].iloc[0]

    assert aggression["n"] == 3
    assert aggression["n_valid"] == 2
    assert aggression["n_refusal"] == 1
    assert np.isclose(aggression["judge_positive_rate"], 0.5)
    assert refusal["n_valid"] == 3
    assert np.isclose(refusal["judge_positive_rate"], 1 / 3)
    assert any_row["n_valid"] == 2
    assert np.isclose(any_row["judge_positive_rate"], 0.5)


def test_issuebench_judge_count_bootstrap_preserves_groups_and_reports_all_dimensions() -> None:
    data = pd.DataFrame(
        {
            "generating_model_name": ["generator-a", "generator-a", "generator-b", "generator-b"],
            "response": [
                '{"aggression": "yes", "submission": "no", "conventionalism": "no", "refusal": "no"}',
                '{"aggression": "no", "submission": "no", "conventionalism": "no", "refusal": "no"}',
                '{"aggression": "no", "submission": "yes", "conventionalism": "no", "refusal": "no"}',
                '{"aggression": "no", "submission": "no", "conventionalism": "no", "refusal": "yes"}',
            ],
        }
    )
    runner = IssueBenchJudgeCountBootstrap(data, n_repetitions=3, random_state=17)

    runner.bootstrap()
    cis = runner.cis()

    assert set(cis.bootstrap_distribution["generating_model_name"]) == {"generator-a", "generator-b"}
    assert (
        cis.bootstrap_distribution.groupby(["repetition", "generating_model_name", "dimension"])["n"]
        .first()
        .eq(2)
        .all()
    )
    assert set(cis.intervals["dimension"]) == {"aggression", "submission", "conventionalism", "refusal", "any"}
    assert {"judge_positive_rate", "ci_low", "ci_high", "n", "n_valid", "n_refusal"}.issubset(cis.intervals.columns)
    assert set(cis.intervals["quantification_method"]) == {"judge_count"}


def _issuebench_judge_prompt(text: str) -> str:
    return f"# Task Description\n\nText to annotate:\n{text}"


def _issuebench_response(
    aggression: str = "no", submission: str = "no", conventionalism: str = "no", refusal: str = "no"
) -> str:
    return (
        '{"aggression": "'
        + aggression
        + '", "submission": "'
        + submission
        + '", "conventionalism": "'
        + conventionalism
        + '", "refusal": "'
        + refusal
        + '"}'
    )


def _judge_rows(texts: list[str], responses: list[str], model: str = "generator-a", task: str = "task") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["judge"] * len(texts),
            "generating_model_name": [model] * len(texts),
            "user_prompts": [[_issuebench_judge_prompt(text)] for text in texts],
            "generating_user_prompts": [[task] for _ in texts],
            "response": responses,
        }
    )


def test_load_issuebench_ground_truth_supports_v3_column_oriented_json(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    pd.DataFrame(
        {
            "model": ["generator-a", "generator-a"],
            "task": ["task", "task"],
            "response": ["positive text", "negative text"],
            "label_aggression": [1.0, -1.0],
            "label_submission": [-1.0, -1.0],
            "label_conventionalism": [-1.0, -1.0],
            "label_refusal": [-1.0, -1.0],
            "predicted_as_autho": [True, False],
        }
    ).to_json(gt_path)

    loaded = load_issuebench_ground_truth(gt_path)

    assert loaded["predicted_as_autho"].tolist() == [True, False]
    assert loaded["label_any"].tolist() == [1.0, -1.0]


def test_load_issuebench_ground_truth_supports_parquet_fold_and_drops_unmatched_rows(tmp_path: Path) -> None:
    v3_path = tmp_path / "ground_truth_data_rebuttal_v3_en.json"
    pd.DataFrame(
        {
            "model": ["generator-a"],
            "task": ["task"],
            "response": ["matched text"],
            "label_aggression": [1.0],
            "label_submission": [-1.0],
            "label_conventionalism": [-1.0],
            "label_refusal": [-1.0],
            "predicted_as_autho": [True],
        }
    ).to_json(v3_path)
    fold_path = tmp_path / "ground_truth_gemma4sft_fold0_val.parquet"
    pd.DataFrame(
        {
            "id": [1, 2],
            "prompt": [_issuebench_judge_prompt("matched text"), _issuebench_judge_prompt("legacy text")],
            "label_aggression": [True, False],
            "label_submission": [False, False],
            "label_conventionalism": [False, False],
            "label_refusal": [False, False],
            "completion": [[], []],
        }
    ).to_parquet(fold_path)

    loaded = load_issuebench_ground_truth(fold_path)

    assert len(loaded) == 1
    assert loaded.iloc[0]["response"] == "matched text"
    assert loaded.iloc[0]["model"] == "generator-a"
    assert bool(loaded.iloc[0]["predicted_as_autho"]) is True


def test_issuebench_judge_performance_bootstrap_uses_design_weights() -> None:
    texts = ["tp", "fp", "tp from selector negative", "tn"]
    ground_truth = pd.DataFrame(
        {
            "model": ["generator-a"] * 4,
            "task": ["task"] * 4,
            "response": texts,
            "label_aggression": [1.0, -1.0, 1.0, -1.0],
            "label_submission": [-1.0] * 4,
            "label_conventionalism": [-1.0] * 4,
            "label_refusal": [-1.0] * 4,
            "predicted_as_autho": [True, True, False, False],
        }
    )
    preds = _judge_rows(
        texts,
        [
            _issuebench_response(aggression="yes"),
            _issuebench_response(aggression="yes"),
            _issuebench_response(aggression="yes"),
            _issuebench_response(aggression="no"),
        ],
    )
    selector = _judge_rows(
        ["s1", "s2", "s3", "s4"],
        [
            _issuebench_response(aggression="yes"),
            _issuebench_response(aggression="no"),
            _issuebench_response(aggression="no"),
            _issuebench_response(aggression="no"),
        ],
    )

    runner = IssueBenchJudgePerformanceBootstrap(preds, ground_truth, selector, n_repetitions=1)
    point = runner.point_estimate()
    aggression = point.loc[point["dimension"] == "aggression"].iloc[0]

    assert np.isclose(aggression["selector_positive_rate"], 0.25)
    assert np.isclose(aggression["weight_predicted_as_autho"], 0.125)
    assert np.isclose(aggression["weight_predicted_as_nonautho"], 0.375)
    assert np.isclose(aggression["tpr"], 1.0)
    assert np.isclose(aggression["fpr"], 0.25)


def test_issuebench_judge_count_adjusted_uses_stable_denominator_for_refusals() -> None:
    data = pd.DataFrame(
        {
            "generating_model_name": ["generator-a"] * 4,
            "response": [
                _issuebench_response(aggression="yes"),
                _issuebench_response(aggression="no"),
                _issuebench_response(aggression="yes", refusal="yes"),
                _issuebench_response(aggression="no"),
            ],
        }
    )
    performance = pd.DataFrame(
        {
            "repetition": [0] * 5,
            "dimension": ["aggression", "submission", "conventionalism", "refusal", "any"],
            "tpr": [0.8] * 5,
            "fpr": [0.1] * 5,
        }
    )
    runner = IssueBenchJudgeCountBootstrap(data, n_repetitions=1, random_state=3)

    adjusted = runner.bootstrap_adjusted(performance)
    point = runner.adjusted_point_estimates

    assert point is not None
    aggression_point = point.loc[point["dimension"] == "aggression"].iloc[0]
    # Stable denominator: the refusal row is a negative, so observed aggression is 1 / 4.
    assert np.isclose(aggression_point["judge_positive_rate"], 0.25)
    assert np.isclose(aggression_point["adjusted_positive_rate"], (0.25 - 0.1) / 0.7)
    assert "adjusted_positive_rate" in adjusted.columns
