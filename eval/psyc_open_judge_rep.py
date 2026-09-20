import argparse
import json
import os
import re
import warnings

import pandas as pd
import numpy as np

from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llm_audit import BASE_DIR
from llm_audit.datasets import DATASETS
from llm_audit.datasets.base import AgreementDataset
from llm_audit.inference import AsyncGenerationConfig, generate
from llm_audit.util import (
    ExperimentType,
    ModelConfig,
    cleanup_vram,
    construct_output_dir_label,
    get_default_system_prompt,
    get_device,
)
from llm_audit.datasets.util import (
    get_dataset_by_label,
    get_agreement_case,
    is_positive_polarized,
    contains_polarity_or_inverted_variable,
)

# GLOBAL DEFAULTS
EXPERIMENT_TYPE: str = "judge"
SEED: int = 42
GPUS: int = 1
TEMPERATURE: float = 1.0
RUNS: int = 1
MODEL_SELECTION_FILE: str = "judges.json"
TAG: str = "0001"
MAX_SEQUENCE_LENGTH: int = 32000
MAX_CONCURRENT: int = 1
LANGUAGE: str = "en"

AGREEMENT_CASE_AUTH_MAPPING = {
    # None group
    "R": None,
    "R+": None,
    "R-": None,
    "N": None,
    "N+": None,
    "N-": None,
    # 1 group (auth)
    "A": 1,
    "A+": 1,
    "D-": 1,
    # 0 group (non auth)
    "D": 0,
    "A-": 0,
    "D+": 0,
}


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM audit judge experiments with configurable parameters.")
    parser.add_argument("experiment_type", choices=["judge"], default=EXPERIMENT_TYPE)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=GPUS,
        help="Number of GPUs available (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="LLM inference temperature when not configured in model config (default: 1.0).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS,
        help="Number of repetitions (default: 1).",
    )
    parser.add_argument(
        "--model-selection-file",
        type=str,
        default=MODEL_SELECTION_FILE,
        help="Model selection JSON filename (default: complete_model_selection.json).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=TAG,
        help="Custom prefix tag of output directory for the corresponding experiment (default: issuebench_gen).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=MAX_SEQUENCE_LENGTH,
        help="Limit the context length of the model. (default: 32000)",
    )
    parser.add_argument(
        "--max-concurrent",
        default=MAX_CONCURRENT,
        type=int,
        help="Values larger than 1 will lead to async calls to API. Maximal number of open requests.",
    )
    parser.add_argument("-l", "--language", choices=["en", "ru", "zh", "de"], default=LANGUAGE)
    return parser.parse_args()


def extract_score(response: str) -> int | None:
    """
    Returns the score if refusal == 0, otherwise None.
    """
    if response is None:
        return None

    # Clean up escaped whitespace and backtick fences
    cleaned = response.replace("\\n", "\n").strip().strip("`").strip()

    try:
        parsed = json.loads(cleaned)
        if parsed.get("refusal") == 0 and "score" in parsed:
            return int(parsed["score"])
    except json.JSONDecodeError:
        # Fallback: Regex
        score_match = re.search(r'"score"\s*:\s*([+-]?\d+)', cleaned)
        refusal_match = re.search(r'"refusal"\s*:\s*(\d+)', cleaned)
        if score_match and refusal_match and refusal_match.group(1) == "0":
            return int(score_match.group(1))
    return None


def main() -> None:
    args = parse_arguments()
    experiment_type = ExperimentType.JUDGE

    # Load model information
    model_config_path = BASE_DIR / "resources" / "input" / "models" / args.model_selection_file
    with open(model_config_path, "r") as f:
        model_cfgs: list[ModelConfig] = json.load(f)

    clients = setup_clients(model_cfgs)
    async_config = AsyncGenerationConfig(max_concurrent=args.max_concurrent) if args.max_concurrent > 1 else None

    # Load data
    inputs, input_metas = prepare_judge_eval_data(args)

    for model_cfg in model_cfgs:
        model_setup = model_cfg["setup"]
        model_name = model_cfg["name"]

        # vLLM specific
        sampling_params = tokenizer = model = None
        if model_setup == "vLLM":
            sampling_params, tokenizer, model = run_vllm_setup(args, model_cfg)

        response_metas = generate(
            experiment_type=experiment_type,
            client=clients.get(model_setup),
            model_name=model_name,
            model_format=model_cfg["format"],
            model_setup=model_cfg["setup"],
            sampling_params=sampling_params,
            model=model,
            tokenizer=tokenizer,
            temperature=model_cfg.get("temperature") or args.temperature,
            seed=None if model_setup == "vLLM" else args.seed,
            repetitions=args.runs,
            system_prompts=[get_default_system_prompt(language=args.language)],
            batch_user_prompts=[[inp] for inp in inputs],
            use_tqdm=True,
            use_async=True if async_config else False,
            async_config=async_config,
            extra_body={"reasoning": {"enabled": True}}
            if model_cfg["name"] in ["openai/gpt-5.1", "google/gemini-3-pro-preview"]
            else None,
        )

        # Verify data integrity
        if len(response_metas) != len(input_metas):
            # If runs > 1, this logic might need adaptation based on how generate returns lists
            warnings.warn(
                f"Results length ({len(response_metas)}) does not match Metadata length ({len(input_metas)}). Metadata merging may be misaligned if runs > 1."
            )

        _dat = {
            "dataset": [x["dataset"] for x in input_metas],
            "label": [None] * len(response_metas),
            "question_id": [int(x["question_id"]) for x in input_metas],
        }
        for i, _response_meta in enumerate(response_metas):
            _dat["label"][i] = (
                extract_score(response=_response_meta["response"]) if _response_meta["response"] is not None else None
            )
        df = pd.DataFrame(_dat)
        df["response_agreement_case"] = df.apply(compute_agreement_case, axis=1)

        df["response_auth"] = df["response_agreement_case"].map(AGREEMENT_CASE_AUTH_MAPPING)

        r_a_c_l = df["response_agreement_case"].tolist()
        r_a_l = df["response_auth"].tolist()
        for i in range(len(input_metas)):
            input_metas[i]["response_agreement_case"] = r_a_c_l[i]
            input_metas[i]["response_auth"] = r_a_l[i]

        # Group results by the model that generated the original response
        grouped_results = defaultdict(list)

        for res, meta in zip(response_metas, input_metas):
            # Merge the input metadata (which contains info about the generated response)
            # into the judge's result
            _res: dict[str, Any] = dict(res)
            _res.update(meta)

            # Extract the name of the model being judged
            # We check common keys. 'generating_model' is standard if input JSON had 'model'
            if meta:
                # force unknown_model via generating_model_name...
                target_model_name = meta.get("generating_model_name") or "unknown_model"
            else:
                target_model_name = model_name

            # Sanitize filename (e.g., replace '/' in 'meta-llama/Llama-2')
            target_model_safe = target_model_name.replace("/", "_")
            grouped_results[target_model_safe].append(_res)

        # Save results into separate folders per target model
        for target_model_safe, model_results in grouped_results.items():
            # judge
            # Path: .../JudgeName/TargetModelName/results.json
            results_file_path = (
                BASE_DIR
                / "eval"
                / "data"
                / "psych_open_reduction_eval"
                / construct_output_dir_label(
                    output_dir_prefix_tag=args.tag,
                    language=args.language,
                    temperature=args.temperature,
                    runs=args.runs,
                    seed=args.seed,
                )
                / model_name  # The Judge
                / target_model_safe  # The Target (generated) Model
                / f"{args.tag}_results.json"
            )

            results_file_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(model_results).to_json(results_file_path)

            # with results_file_path.open("w", encoding="utf-8") as f:
            #     json.dump(model_results, f, ensure_ascii=False, indent=4)

        # Cleanup
        if model_cfg["setup"] == "vLLM":
            assert model is not None
            assert tokenizer is not None
            cleanup_vram(model=model, tokenizer=tokenizer)


def run_vllm_setup(args: Namespace, model_cfg: ModelConfig) -> tuple[SamplingParams, Any, LLM]:
    seed_info_msg = """ SEED INFO
    vLLM Limitation: Can not set different seeds for batched requests, only global seed for whole batch => issue for repetitions: identical input -> pot. identical output!
    Decide for continuous batching to gain efficiency and thus against (maybe) reproducibility via sequential (single) requests (no batching).
    Not setting a seed defaults to None, which yields non reproducible outcomes (enabling output variance for repetitions).
    We explicitly set seed to None when initalizing SamplingParams and LLM.
    See V0 comment: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/reproducibility.py
    """
    warnings.warn(seed_info_msg)

    # TODO Optimization, lookup and skip if (follow-up) computations have already been done.
    # use complete_results_exist from llm_audit.datasets.util
    model_context_window = model_cfg["context_window"]

    sampling_params = SamplingParams(
        n=1,  # Number of output sequences
        seed=None,  # seed=None, see <SEED INFO> comment above
        temperature=model_cfg.get("temperature") or args.temperature,
        max_tokens=min(model_context_window, args.max_sequence_length),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])  # type: ignore
    # TODO: for continuous batching, set: max_num_seqs: int = 256 (default) to 30?
    # max_num_seqs :: maximum number of sequences per iteration
    model = LLM(
        model_cfg["name"],
        tensor_parallel_size=max(1, args.gpus),  # 1 :: no distributed initialization, 2 :: 2 GPUs, ..., n :: n GPUs
        trust_remote_code=True,
        seed=None,  # seed=None, see <SEED INFO> comment above
        max_model_len=min(model_context_window, args.max_sequence_length),
    )

    return sampling_params, tokenizer, model


def compute_agreement_case(row: pd.Series) -> str:
    dataset_obj = get_dataset_by_label(row["dataset"])
    assert isinstance(dataset_obj, AgreementDataset)

    agreement_case = get_agreement_case(
        raw_score=row["label"],
        threshold=dataset_obj.get_agreement_discriminator_threshold(),
        positive_polarized=is_positive_polarized(
            dataset_obj.get_polarity(
                language="en",
                id=row["question_id"],
                polarity_literal=contains_polarity_or_inverted_variable(dataset=dataset_obj, language="en"),
            )
        ),
        disagree_to_agree=dataset_obj.is_ordered_disagree_to_agree_asc(),
    )
    return agreement_case


def prepare_judge_eval_data(args: Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    data_path = (
        BASE_DIR / "resources" / "input" / "datasets" / "reduction" / f"reduction_ground_truth_3_{args.language}.csv"
    )
    df = pd.read_csv(data_path)

    df["agreement_case"] = df.apply(compute_agreement_case, axis=1)
    df["auth"] = df["agreement_case"].map(AGREEMENT_CASE_AUTH_MAPPING)

    combined_data = []
    for idx, row in df.iterrows():
        dataset_name = row["dataset"]
        dataset = DATASETS[dataset_name](reverse_scale=False)

        text_to_judge = row["open_response"]
        open_response_instruction = row["task"]
        prompt = dataset.get_open_to_closed_judge_reduction_instruction(
            open_response_instruction,
            open_question_response=text_to_judge,
            language=args.language,
        )

        meta = {key: val for key, val in row.items()}
        combined_data.append((prompt, meta))

    combined_data.sort(key=lambda pair: len(pair[0]))
    inputs, metas = zip(*combined_data)
    return list(inputs), list(metas)


def setup_clients(model_cfgs: list[ModelConfig]) -> dict[str, OpenAI]:
    load_dotenv()
    clients = {}
    if any(cfg["setup"] == "Alternative" for cfg in model_cfgs):
        clients["Alternative"] = OpenAI(
            base_url=os.getenv("ALT_BASE_URL"),
            api_key=os.getenv("ALT_API_KEY"),
        )
    if any(cfg["setup"] == "OpenRouter" for cfg in model_cfgs):
        clients["OpenRouter"] = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OR_API_KEY"),
        )
    if any(cfg["setup"] == "vLLM" for cfg in model_cfgs):
        _ = get_device()  # Debug print
        if hf_token := os.getenv("HF_TOKEN"):
            login(token=hf_token)
    return clients


if __name__ == "__main__":
    main()
