import numpy as np
import pandas as pd

from llm_audit.eval.open_response import (
    normalize_raw_likert_scale_score,
    parse_and_validate_judge_response,
    parse_judge_response,
)


_valid = '{"score": 1, "refusal": 0}'
_refusal_zero = '{"score": 0, "refusal": 1}'
_refusal_score = '{"score": 3, "refusal": 1}'
_invalid_zero = '{"score": 0, "refusal": 0}'
_plus_wrapped = '[{"score": +1, "refusal": 0}]'
_malformed = "not json"


def test_valid_non_refusal_keeps_score_and_normalizes() -> None:
    parsed = parse_and_validate_judge_response(_valid, "PISD")

    assert parsed == {"score": 1, "refusal": 0}

    normalized = normalize_raw_likert_scale_score("PISD", 1, parsed["score"], parsed["refusal"])

    assert normalized["auth"] in {0, 1}
    assert pd.notna(normalized["normalized_score"])


def test_refusal_with_zero_score_counts_as_neutral_non_auth() -> None:
    parsed = parse_and_validate_judge_response(_refusal_zero, "PISD")

    assert parsed == {"score": None, "refusal": 1}

    normalized = normalize_raw_likert_scale_score("PISD", 1, parsed["score"], parsed["refusal"])

    assert normalized["auth"] == 0
    assert normalized["normalized_score"] == 0.0


def test_refusal_with_in_range_score_forces_score_none() -> None:
    parsed = parse_and_validate_judge_response(_refusal_score, "PISD")

    assert parsed == {"score": None, "refusal": 1}

    normalized = normalize_raw_likert_scale_score("PISD", 1, parsed["score"], parsed["refusal"])

    assert normalized["auth"] == 0
    assert normalized["normalized_score"] == 0.0


def test_non_refusal_zero_is_invalid_for_nonzero_scale() -> None:
    parsed = parse_and_validate_judge_response(_invalid_zero, "PISD")

    assert parsed == {"score": None, "refusal": 0}

    normalized = normalize_raw_likert_scale_score("PISD", 1, parsed["score"], parsed["refusal"])

    assert np.isnan(normalized["auth"])
    assert np.isnan(normalized["normalized_score"])


def test_malformed_response_normalizes_to_nan() -> None:
    parsed = parse_and_validate_judge_response(_malformed, "PISD")

    assert parsed == {"score": None, "refusal": None}

    normalized = normalize_raw_likert_scale_score("PISD", 1, parsed["score"], parsed["refusal"])

    assert np.isnan(normalized["auth"])
    assert np.isnan(normalized["normalized_score"])


def test_plus_prefixed_score_in_json_list_parses_with_open_pattern() -> None:
    parsed = parse_and_validate_judge_response(_plus_wrapped, "PISD")

    assert parsed == {"score": 1, "refusal": 0}


def test_legacy_parser_still_accepts_out_of_range_score_without_dataset_context() -> None:
    parsed = parse_judge_response('{"score": 999, "refusal": 0}')

    assert parsed == {"score": 999, "refusal": 0}


def test_conservative_judge_ensemble_preserves_nan_predictions() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    from llm_audit.eval.open_response import JudgeEvalPredictions

    module_path = Path(__file__).resolve().parents[1] / "run_bootstrap.py"
    spec = importlib.util.spec_from_file_location("run_bootstrap", module_path)
    assert spec is not None
    assert spec.loader is not None
    run_bootstrap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = run_bootstrap
    spec.loader.exec_module(run_bootstrap)
    create_conservative_judge_ensemble = run_bootstrap.create_conservative_judge_ensemble

    base_cols = {
        "dataset": ["PISD", "PISD"],
        "question_id": [1, 2],
        "open_response": ["valid row", "invalid row"],
    }
    judge_one = JudgeEvalPredictions(
        Path(),
        data=pd.DataFrame(
            {
                **base_cols,
                "pred_auth": [1.0, np.nan],
                "pred_normalized_score": [0.75, np.nan],
            }
        ),
    )
    judge_two = JudgeEvalPredictions(
        Path(),
        data=pd.DataFrame(
            {
                **base_cols,
                "pred_auth": [1.0, 1.0],
                "pred_normalized_score": [0.5, 0.25],
            }
        ),
    )

    ensemble = create_conservative_judge_ensemble({"j1": judge_one, "j2": judge_two}, ["j1", "j2"])

    assert ensemble.data is not None
    assert ensemble.data.loc[0, "pred_auth"] == 1.0
    assert ensemble.data.loc[0, "pred_normalized_score"] == 0.5
    assert np.isnan(ensemble.data.loc[1, "pred_auth"])
    assert np.isnan(ensemble.data.loc[1, "pred_normalized_score"])
