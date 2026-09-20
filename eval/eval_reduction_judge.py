import argparse
import json
import warnings

import pandas as pd

from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llm_audit import BASE_DIR
from llm_audit.datasets import DATASETS
from llm_audit.inference import AsyncGenerationConfig, generate
from llm_audit.util import (
    ExperimentType,
    ModelConfig,
    cleanup_vram,
    construct_output_dir_label,
    get_default_system_prompt,
    setup_clients,
)


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM audit experiments with configurable parameters.")
    parser.add_argument("experiment_type", choices=["judge"])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs available (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM inference temperature when not configured in model config (default: 1.0).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repetitions (default: 1).",
    )
    parser.add_argument(
        "--model-selection-file",
        type=str,
        default="complete_model_selection.json",
        help="Model selection JSON filename (default: complete_model_selection.json).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="reduction_eval",
        help="Custom prefix tag of output directory for the corresponding experiment (default: issuebench_gen).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=32000,
        help="Limit the context length of the model. (default: 32000)",
    )
    parser.add_argument(
        "--max-concurrent",
        default=1,
        type=int,
        help="Values larger than 1 will lead to async calls to API. Maximal number of open requests.",
    )
    parser.add_argument("-l", "--language", choices=["en", "ru", "zh", "de"], default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    experiment_type = (
        ExperimentType.GENERATE if args.experiment_type in ["gen", "gen-high-risk"] else ExperimentType.JUDGE
    )

    output_root: Path = (
        BASE_DIR
        / "resources"
        / "output"
        / construct_output_dir_label(
            output_dir_prefix_tag=args.tag,
            language=args.language,
            temperature=args.temperature,
            runs=args.runs,
            seed=args.seed,
        )
    )

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

        if model_setup == "Alternative" or model_setup == "vLLM":
            extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
        elif model_setup == "OpenRouter":
            extra_body = {"reasoning": {"enabled": True}}

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
            extra_body=extra_body,
        )

        # Verify data integrity
        if len(response_metas) != len(input_metas):
            # If runs > 1, this logic might need adaptation based on how generate returns lists
            warnings.warn(
                f"Results length ({len(response_metas)}) does not match Metadata length ({len(input_metas)}). Metadata merging may be misaligned if runs > 1."
            )

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
                target_model_name = meta.get("generating_model_name") or "unknown_model"
            else:
                target_model_name = model_name

            # Sanitize filename (e.g., replace '/' in 'meta-llama/Llama-2')
            target_model_safe = target_model_name.replace("/", "_")
            grouped_results[target_model_safe].append(_res)

        # Save results into separate folders per target model
        for target_model_safe, model_results in grouped_results.items():
            if experiment_type == ExperimentType.GENERATE:  # gen
                results_file_path = output_root / "IssueBench" / experiment_type.value / model_name / "results.json"
            else:  # judge
                # Path: .../JudgeName/TargetModelName/results.json
                results_file_path = (
                    output_root
                    / "IssueBench"
                    / experiment_type.value
                    / model_name  # The Judge
                    / target_model_safe  # The Target (generated) Model
                    / "results.json"
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
        dtype=model_cfg.get("dtype", "auto"),  # type: ignore[arg-type]
    )

    return sampling_params, tokenizer, model


def prepare_judge_eval_data(args: Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    data_path = (
        BASE_DIR
        / "resources"
        / "input"
        / "datasets"
        / "reduction"
        # / f"reduction_ground_truth_3_{args.language}.csv"
        # / f"reduction_ground_truth_4-max_{args.language}.csv"
        / f"reduction_ground_truth_5-max_{args.language}.csv"
    )
    df = pd.read_csv(data_path)

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


if __name__ == "__main__":
    main()
