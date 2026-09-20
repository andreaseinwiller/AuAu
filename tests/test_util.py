import pytest

from typing import Any

from llm_audit.util import (
    ExperimentType,
    get_enable_thinking,
    get_key_of_last_process_to_parse_by_experiment_type,
)

"""
get_enable_thinking
"""
testdata_valid_get_enable_thinking = [
    (ExperimentType.CLOSED_QUESTION, False),
    (ExperimentType.OPEN_QUESTION, True),
    (ExperimentType.CASE_VIGNETTE, False),
]


@pytest.mark.parametrize("experiment_type, expected_result", testdata_valid_get_enable_thinking)
def test_get_enable_thinking_valid_input(experiment_type: ExperimentType, expected_result: bool) -> None:
    assert get_enable_thinking(experiment_type) == expected_result


testdata_invalid_get_enable_thinking: Any = [
    "invalid_experiment_type",
    None,
    123,
    [],
    {},
]


@pytest.mark.parametrize("invalid_input", testdata_invalid_get_enable_thinking)
def test_get_enable_thinking_invalid_input(invalid_input: Any) -> None:
    with pytest.raises(AssertionError):
        get_enable_thinking(invalid_input)


"""
get_key_of_last_process_to_parse_by_experiment_type
"""
testdata_valid_get_key_of_last_process_to_parse_by_experiment_type = [
    (ExperimentType.CLOSED_QUESTION, "closed_response"),
    (ExperimentType.OPEN_QUESTION, "closed_response"),
    (ExperimentType.CASE_VIGNETTE, "vignette_response"),
]


@pytest.mark.parametrize(
    "experiment_type, expected_result",
    testdata_valid_get_key_of_last_process_to_parse_by_experiment_type,
)
def test_get_key_of_last_process_to_parse_by_experiment_type_valid_input(
    experiment_type: ExperimentType, expected_result: str
) -> None:
    assert get_key_of_last_process_to_parse_by_experiment_type(experiment_type) == expected_result


testdata_invalid_get_key_of_last_process_to_parse_by_experiment_type: Any = [
    "invalid_experiment_type",
    None,
    123,
    [],
    {},
]


@pytest.mark.parametrize(
    "invalid_input",
    testdata_invalid_get_key_of_last_process_to_parse_by_experiment_type,
)
def test_get_key_of_last_process_to_parse_by_experiment_type_invalid_input(
    invalid_input: Any,
) -> None:
    with pytest.raises(AssertionError):
        get_key_of_last_process_to_parse_by_experiment_type(invalid_input)
