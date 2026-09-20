import json
import math

import numpy as np

from pathlib import Path
from typing import Any, Literal, Match, Pattern, TypedDict, cast
from openai import OpenAI
from vllm import LLM, SamplingParams

from llm_audit.datasets.base import (
    AgreementDataset,
    VignetteDataset,
    VignetteOptionLetters,
)
from llm_audit.inference import AsyncGenerationConfig, ResponseMeta, generate
from llm_audit.util import (
    ErrorDict,
    ExperimentType,
    ModelConfig,
    get_auth_system_prompt,
    get_default_system_prompt,
    get_key_of_last_process_to_parse_by_experiment_type,
)


class LLMExperimentMeta(TypedDict):
    """
    Metadata for a single experimental run within the llm_audit framework.
    Lists contain only one entry in case of closed_question and case_vignette approach.
    """

    experiment_type: str
    open_response: str | None
    closed_response: list[str | None] | None
    vignette_response: str | None
    vignette_letter_mapping: str | None
    response_value: str | None
    refusal: list[str | None] | None
    valid_response_format: list[bool | None] | None
    valid_response_value: list[bool | None] | None
    open_response_meta: ResponseMeta | None
    closed_response_meta: list[ResponseMeta | None] | None
    vignette_response_meta: ResponseMeta | None
    error: ErrorDict | None


def parse_response(
    response_to_parse: list[str | None], pattern_closed_response: Pattern[str]
) -> list[Match[str] | None]:
    valid_response: list[Match[str] | None] = [None] * len(response_to_parse)

    for i, response_to_parse_entry in enumerate(response_to_parse):
        if response_to_parse_entry is None:
            continue
        response_to_parse_entry_cleaned: str = (
            response_to_parse_entry.strip().strip("`").strip()
        )  # Remove ```<...>``` formatting/padding
        try:
            # Try to parse response as JSON
            parsed_reponse_entry = json.loads(response_to_parse_entry_cleaned)
            formatted_response_entry = json.dumps(parsed_reponse_entry)
            valid_response_entry = pattern_closed_response.search(formatted_response_entry)
            valid_response[i] = valid_response_entry
        except json.JSONDecodeError:
            # Fall back to plain regex approach
            valid_response_entry = pattern_closed_response.search(response_to_parse_entry_cleaned)
            valid_response[i] = valid_response_entry
    return valid_response


def add_response_to_experiment_metadata(
    valid_response: list[Match[str] | None],
    experiment_metadata: LLMExperimentMeta,
    dataset: AgreementDataset,
) -> LLMExperimentMeta:
    from llm_audit.datasets.util import (
        get_agreement_case,
    )

    n_entries = len(valid_response)

    if n_entries < 1:
        return experiment_metadata

    response_values: list[str | None] = [None] * n_entries
    refusals: list[str | None] = [None] * n_entries
    valid_response_formats: list[bool | None] = [None] * n_entries
    valid_response_values: list[bool | None] = [None] * n_entries

    for i, response in enumerate(valid_response):
        if response is None:
            response_values[i] = None
            refusals[i] = None
            valid_response_formats[i] = None
            valid_response_values[i] = None
        else:
            try:
                response_value_str = response.group(1).strip()
                refusal_str = response.group(2).strip()
                # None or type str, no int cast! (-> lstrip leading + char required!)
                response_value = None if response_value_str == "null" else response_value_str.lstrip("+").strip('"')
                # lstrip("+"): resolve "+4" != "4"
                # strip('"'): resolve '"low_rwa"' != 'low_rwa'
                refusal = refusal_str
                refusals[i] = refusal
                valid_response_formats[i] = True

                # Validate results
                # lstrip("+"): resolve "+4" != "4"
                if (
                    refusal == "0"
                    and response_value is not None
                    and response_value in [x.lstrip("+") for x in dataset.get_scale_items()]
                ):
                    response_values[i] = response_value
                    valid_response_values[i] = True
                else:
                    response_values[i] = None
                    valid_response_values[i] = False
            except Exception as e:
                experiment_metadata["error"] = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
    response_values_cast: list[int | None] = [
        v if v is None else int(v) for v in response_values
    ]  # str to int if not None
    response_value_reduced: str | None = None  # Safeguard for n_entries < 1
    if n_entries == 1:
        response_value_reduced = response_values[0]
    else:
        # n_entries > 1, multiple judges
        if any(x is None for x in response_values):
            # 1. None if any judge fails to complete the reduction task
            response_value_reduced = None
        else:
            # 2. None if judges do not agree on neutral or polarity (e.g., all judges clf the open-ended text as (overall) neutral, agreement, or disagreement)
            agreement_cases = set()
            for response_value in response_values_cast:
                agreement_case = get_agreement_case(
                    raw_score=response_value,
                    threshold=dataset.get_agreement_discriminator_threshold(),
                    positive_polarized=None,  # Judge comp over same instance, eq case comp is invariant to reverse scored item property
                    disagree_to_agree=True,  # Invariant to order
                )
                agreement_cases.add(agreement_case)
            if len(agreement_cases) > 1:
                response_value_reduced = None
            else:
                # 3. Compute closest Likert scale item based on mean (valid, no outlier due to 2. constraint)
                response_values_cast_filtered: list[int] = [v for v in response_values_cast if v is not None]
                assert len(response_values_cast) == len(response_values_cast_filtered)  # see 1.
                response_value_mean = np.mean(response_values_cast_filtered)
                scale_items_cast: list[int] = [int(v) for v in dataset.get_scale_items()]
                items_distance_to_mean: list[tuple[int, float]] = [
                    (v, float(abs(v - response_value_mean))) for v in scale_items_cast
                ]
                closest_distance_to_mean: float = min([d for _, d in items_distance_to_mean])
                item_candidates: list[int] = [
                    v for v, d in items_distance_to_mean if math.isclose(a=d, b=closest_distance_to_mean, abs_tol=1e-5)
                ]

                if len(item_candidates) == 1:
                    response_value_reduced = str(item_candidates[0])
                else:
                    threshold: float = dataset.get_agreement_discriminator_threshold()
                    if not math.isclose(a=closest_distance_to_mean, b=threshold, abs_tol=1e-5):
                        # Case 1: mean != threshold (neutral/polarity discriminator), select candidate closest to threshold
                        # e.g., Likert scale 1,2,3,4; mean 3.5; threshold 2.5; candidates 3 and 4, select 3
                        response_value_reduced = str(min(item_candidates, key=lambda x: abs(x - threshold)))
                    else:
                        # Case 2: mean == threshold, select lower candidate
                        # e.g., Likert scale 1,2,3,4; mean 2.5; threshold 2.5; candidates 2 and 3, select 2
                        # introduces bias, inevitable, special case near neutral, lower :: usually anti-authoritarian
                        response_value_reduced = str(min(item_candidates))
    experiment_metadata["closed_response"] = response_values  # List of LLM-as-a-Judge reductions
    experiment_metadata["response_value"] = (
        response_value_reduced  # Single value reduction (see 1., 2., and 3. comments)
    )
    experiment_metadata["refusal"] = refusals
    experiment_metadata["valid_response_format"] = valid_response_formats
    experiment_metadata["valid_response_value"] = valid_response_values
    return experiment_metadata


def run_experiment(
    experiment_type: ExperimentType,
    clients: dict[str, OpenAI],
    model_cfg: ModelConfig,
    judge_cfgs: list[ModelConfig],
    sampling_params: SamplingParams | None,
    model: LLM | None,
    tokenizer: Any | None,
    temperature: float,
    seed: int | None,
    language: Literal["en", "ru", "de", "zh"],
    dataset: AgreementDataset | VignetteDataset,
    id: int,
    output_dir: Path,
    auth_sys_prompt: bool = False,
    use_tqdm: bool = False,
    use_async: bool = False,
    async_config: AsyncGenerationConfig | None = None,
    num_runs: int = 1,
    prompt_style: Literal["short", "long"] = "short",
) -> list[LLMExperimentMeta]:
    """
    Runs a structured LLM-based experiment based on an open-ended question to be transformed into a closed-form response.

    This function automates the behavior evaluation of LLMs in response to defined system and user prompts, generating responses to closed or open(-ended) questions and
    transforming them into structured, machine-readable closed-form responses. The robustness is ensured based on (a combination of) prompt variations and reruns.

    Parameters:
        experiment_type (str): Type to differentiate between different experiment runs such as "closed_question" and "open_question".
        client (OpenAI | None): An instance of the OpenAI client used to send the request.
        judge_client (OpenAI | None): An instance of the OpenAI client used to send the request.
        judge_name (str): The model called via the judge client.
        model_name (str): The model label defined by OpenRouter.
        model_format (str): The format laybel used to determine the prompt structure (e.g., "text-only", or "multimodal"). None if local setup using vLLM is used for LLM inference.
        model_setup (str): key to determine inference setup (e.g., "OpenRouter" or "vLLM").
        sampling_params <TODO type and description>
        model <TODO type and description>
        tokenizer <TODO type and description>
        temperature (float): Set focus on determinism or randomness.
        seed (int | None): Attempt deterministic sampling, which is not not guaranteed according to OpenAI. Can be set to None to enable repetition output variance during vLLM continuous batch inference.
        language (Literal['en', 'ru', 'de', 'zh']): Language key used to retrieve corresponding instructions.
        dataset <TODO type and description>
        id (int): Unique experiment identifier used for organizing output data.
        output_dir (Path): Directory path to store experiment output.
        auth_sys_prompt (bool): Replace default system prompt with malicious authoritarian system prompt
        use_tqdm (bool): <TODO description>
        use_async (bool): <TODO description>
        async_config (AsyncGenerationConfig | None): <TODO description>
        num_runs (int, optional): Number of experiment runs for each prompt combination. Default is set to 1.
        prompt_style (str): The style of the open question prompt ("short", "long")

    Returns:
        list[ResponseMeta]: Metadata of experiment runs given experiment_type and (scale item) id; IMPORTANT: Empty list in case the computation has already been performed! Thus, handle [] return correctly!
    """
    num_runs = max(1, num_runs)  # Safeguard

    model_name = model_cfg["name"]
    n_items = len(dataset.get_ids(language=language))
    print(f"model={model_name} dataset={dataset.get_label()} {id=}/{n_items} approach={experiment_type.value}")

    results_file_path = output_dir / dataset.get_label() / experiment_type.value / str(id) / model_name / "results.json"
    if results_file_path.exists() and experiment_type != ExperimentType.CASE_VIGNETTE:
        # One comp missing -> redo all for whole dataset
        assert not isinstance(dataset, VignetteDataset), (
            "Cannot skip already computed VignetteDataset experiments due to flattening."
        )
        # TODO: parametrize skip or override
        # IMPORTANT: Handle empty results list and do not override results with empty list!
        print(f"Skipping... {results_file_path} has already been computed!")
        return []

    results: list[LLMExperimentMeta] = []

    experiment_metadatas: list[LLMExperimentMeta] = [
        {
            "experiment_type": experiment_type.value,
            "open_response": None,
            "closed_response": None,
            "vignette_response": None,
            "vignette_letter_mapping": None,
            "response_value": None,
            "refusal": None,
            "valid_response_format": None,
            "valid_response_value": None,
            "open_response_meta": None,
            "closed_response_meta": None,
            "vignette_response_meta": None,
            "error": None,
        }
        for _ in range(num_runs)
    ]

    if experiment_type == ExperimentType.OPEN_QUESTION:
        assert isinstance(dataset, AgreementDataset), "Expected AgreementDataset for OPEN_QUESTION!"
        assert not isinstance(dataset, VignetteDataset), "Expected AgreementDataset, not VignetteDataset!"
        statement = dataset.get_statement(id=id, language=language)
        open_response_metas = generate(
            experiment_type=experiment_type,
            client=clients.get(model_cfg["setup"]),
            model_name=model_name,
            model_format=model_cfg["format"],
            model_setup=model_cfg["setup"],
            sampling_params=sampling_params,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            seed=seed,
            repetitions=num_runs,
            system_prompts=[
                get_auth_system_prompt(language=language)
                if auth_sys_prompt
                else get_default_system_prompt(language=language)
            ],
            batch_user_prompts=[
                [dataset.get_open_question_instruction(statement=statement, language=language, style=prompt_style)]
            ],
            use_tqdm=use_tqdm,
            use_async=use_async,
            async_config=async_config,
        )

        reduction_batch = []
        meta_idx_w_open_response = []
        for idx, (experiment_metadata, open_response_meta) in enumerate(zip(experiment_metadatas, open_response_metas)):
            # Call by ref, mutable :: experiment_metadatas[i]["open_response_meta"] = open_response_meta
            experiment_metadata["open_response_meta"] = open_response_meta

            if open_response_meta["response"] is not None:
                experiment_metadata["open_response"] = open_response_meta["response"]
                meta_idx_w_open_response.append(idx)
                reduction_batch.append(
                    [
                        dataset.get_open_to_closed_judge_reduction_instruction(
                            open_question_instruction=open_response_meta["user_prompts"][0],
                            open_question_response=open_response_meta["response"],
                            language=language,
                        )
                    ]
                )

        judge_response_metas: list[list[ResponseMeta]] = []
        for judge_cfg in judge_cfgs:
            closed_response_metas = generate(
                experiment_type=experiment_type,
                client=clients[judge_cfg["setup"]],
                model_name=judge_cfg["name"],
                model_format=judge_cfg["format"],
                model_setup=judge_cfg["setup"],
                sampling_params=sampling_params,
                model=model,
                tokenizer=tokenizer,
                temperature=judge_cfg.get("temperature") or 1.0,
                seed=seed,
                repetitions=1,
                system_prompts=[get_default_system_prompt(language=language)],
                batch_user_prompts=reduction_batch,
                use_tqdm=use_tqdm,
                use_async=use_async,
                async_config=async_config,
            )
            judge_response_metas.append(closed_response_metas)

        for experiment_metadata in experiment_metadatas:
            # Propagate open_question None
            # (Prev) Resolves bug in function parse_response, where valid_response could become None in case of only open_question None/refusals
            # (Now) valid_response becomes list of None(s) as intended
            # Keep closed meta None
            if experiment_metadata["open_response"] is None:
                experiment_metadata["closed_response"] = [None] * len(judge_cfgs)

        # Pack all judge results into the metas of the experiment as lists
        # (Only those that actually have an open response)
        # judge_response_metas become list of empty lists in case of only "open refusals"
        for closed_response_metas in judge_response_metas:
            for idx, closed_response_meta in zip(meta_idx_w_open_response, closed_response_metas):
                experiment_metadata = experiment_metadatas[idx]
                if experiment_metadata["closed_response_meta"] is None:
                    experiment_metadata["closed_response_meta"] = [closed_response_meta]
                else:
                    experiment_metadata["closed_response_meta"].append(closed_response_meta)
                # closed_response_meta["response"] is None :: judge refusal!
                if experiment_metadata["closed_response"] is None:
                    experiment_metadata["closed_response"] = [closed_response_meta["response"]]
                else:
                    experiment_metadata["closed_response"].append(closed_response_meta["response"])

    elif experiment_type == ExperimentType.CLOSED_QUESTION:
        assert isinstance(dataset, AgreementDataset), "Expected AgreementDataset for CLOSED_QUESTION!"
        assert not isinstance(dataset, VignetteDataset), "Expected AgreementDataset, not VignetteDataset!"
        statement = dataset.get_statement(id=id, language=language)
        closed_response_metas = generate(
            experiment_type=experiment_type,
            client=clients.get(model_cfg["setup"]),
            model_name=model_name,
            model_format=model_cfg["format"],
            model_setup=model_cfg["setup"],
            sampling_params=sampling_params,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            seed=seed,
            repetitions=num_runs,
            system_prompts=[
                get_auth_system_prompt(language=language)
                if auth_sys_prompt
                else get_default_system_prompt(language=language)
            ],
            batch_user_prompts=[[dataset.get_closed_question_instruction(statement=statement, language=language)]],
            use_tqdm=use_tqdm,
            use_async=use_async,
            async_config=async_config,
        )
        for experiment_metadata, closed_response_meta in zip(experiment_metadatas, closed_response_metas):
            experiment_metadata["closed_response_meta"] = [closed_response_meta]

            if closed_response_meta["response"] is not None:
                experiment_metadata["closed_response"] = [closed_response_meta["response"]]
    elif experiment_type == ExperimentType.CASE_VIGNETTE:
        assert isinstance(dataset, VignetteDataset), "Expected VignetteDataset for CASE_VIGNETTE!"
        assert not isinstance(dataset, AgreementDataset), "Expected VignetteDataset, not AgreementDataset!"
        vignette_response_metas = generate(
            experiment_type=experiment_type,
            client=clients.get(model_cfg["setup"]),
            model_name=model_name,
            model_format=model_cfg["format"],
            model_setup=model_cfg["setup"],
            sampling_params=sampling_params,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            seed=seed,
            repetitions=num_runs,
            system_prompts=[
                get_auth_system_prompt(language=language)
                if auth_sys_prompt
                else get_default_system_prompt(language=language)
            ],
            batch_user_prompts=[[dataset.get_case_vignette_instruction(language=language, id=id)]],
            use_tqdm=use_tqdm,
            use_async=use_async,
            async_config=async_config,
        )
        for experiment_metadata, vignette_response_meta in zip(experiment_metadatas, vignette_response_metas):
            experiment_metadata["vignette_response_meta"] = vignette_response_meta

            if vignette_response_meta["response"] is not None:
                experiment_metadata["vignette_response"] = vignette_response_meta["response"]

    for experiment_metadata in experiment_metadatas:
        # dict.get() returns Any or object
        response_to_parse: list[str | None] = cast(
            list[str | None],
            experiment_metadata.get(
                get_key_of_last_process_to_parse_by_experiment_type(experiment_type=experiment_type),
                None,
            ),
        )
        if not isinstance(response_to_parse, list):
            # vignettes
            response_to_parse = [response_to_parse]
        pattern_closed_response: Pattern[str] = dataset.get_closed_response_pattern()
        valid_response: list[Match[str] | None] = parse_response(
            response_to_parse=response_to_parse,
            pattern_closed_response=pattern_closed_response,
        )

        if experiment_type == ExperimentType.CLOSED_QUESTION or experiment_type == ExperimentType.OPEN_QUESTION:
            assert isinstance(dataset, AgreementDataset), (
                "Expected AgreementDataset for CLOSED_QUESTION or OPEN_QUESTION!"
            )
            assert not isinstance(dataset, VignetteDataset), "Expected AgreementDataset, not VignetteDataset!"
            experiment_metadata = add_response_to_experiment_metadata(
                valid_response=valid_response,
                experiment_metadata=experiment_metadata,
                dataset=dataset,
            )
        if experiment_type == ExperimentType.CASE_VIGNETTE:
            assert isinstance(dataset, VignetteDataset), "Expected VignetteDataset for CASE_VIGNETTE!"
            assert not isinstance(dataset, AgreementDataset), "Expected VignetteDataset, not AgreementDataset!"
            assert len(valid_response) == 1
            response: Match[str] | None = valid_response[0]

            response_value: VignetteOptionLetters | None = None
            refusal: str | None = None
            valid_response_format: bool | None = None
            valid_response_value: bool | None = None

            if response is not None:
                try:
                    response_value_str = response.group(1).strip()
                    refusal_str = response.group(2).strip()
                    response_value = cast(
                        VignetteOptionLetters | None,
                        (None if response_value_str == "null" else response_value_str.strip('"')),
                    )
                    # strip('"'): resolve '"low_rwa"' != 'low_rwa'
                    refusal = refusal_str
                    valid_response_format = True

                    # Validate results
                    if (
                        refusal == "0"
                        and response_value is not None
                        and response_value in [x for x in dataset.get_scale_items()]
                    ):
                        valid_response_value = True
                    else:
                        response_value = None
                        valid_response_value = False
                except Exception as e:
                    experiment_metadata["error"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
            experiment_metadata["vignette_letter_mapping"] = str(
                dataset.get_vignette_mapping(language=language, vignette_id=id)
            )
            experiment_metadata["refusal"] = [refusal] if refusal is not None else None
            experiment_metadata["valid_response_format"] = (
                [valid_response_format] if valid_response_format is not None else None
            )
            experiment_metadata["valid_response_value"] = (
                [valid_response_value] if valid_response_value is not None else None
            )

            if response_value is not None:
                experiment_metadata["response_value"] = dataset.decode_response(
                    language=language, vignette_id=id, letter_response=response_value
                )

        # Any dataset (AgreementDataset or VignetteDataset)
        results.append(experiment_metadata)
    return results
