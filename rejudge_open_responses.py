import argparse
import json
import warnings

from argparse import Namespace
from pathlib import Path
from typing import Any, Literal
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llm_audit import BASE_DIR
from llm_audit.eval.open_response import RELEVANT_DATASETS, parse_judge_response
from llm_audit.inference import AsyncGenerationConfig, ResponseMeta, generate
from llm_audit.util import (
    ExperimentType,
    ModelConfig,
    cleanup_vram,
    get_enable_thinking,
    get_formatted_text_content,
    setup_clients,
)


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description="Re-judge existing LLM responses with a different judge model.")
    parser.add_argument(
        "--model-selection-file",
        type=str,
        default="judges_or_alt.json",
        help="Judge model selection JSON filename (default: judges_or_alt.json).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root input directory with existing results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for new judgments.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=RELEVANT_DATASETS,
        help=f"List of datasets to process (default: {' '.join(RELEVANT_DATASETS)}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42).",
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
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs available (default: 1).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=16000,
        help="Limit the context length of the model. (default: 16000)",
    )
    parser.add_argument(
        "--max-concurrent",
        default=1,
        type=int,
        help="Values larger than 1 will lead to async calls to API. Maximal number of open requests.",
    )
    parser.add_argument(
        "--batch-size",
        default=10,
        type=int,
        help="Number of files to batch before sending requests to the judge (default: 10).",
    )
    return parser.parse_args()


def walk_results_files(input_dir: Path, output_dir: Path, datasets: list[str]) -> list[Path]:
    results_files = []
    for dataset in datasets:
        dataset_path = input_dir / dataset / "open_question"
        if not dataset_path.exists():
            continue
        for item_id_dir in dataset_path.iterdir():
            if not item_id_dir.is_dir():
                continue
            for company_dir in item_id_dir.iterdir():
                if not company_dir.is_dir():
                    continue
                for model_dir in company_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    results_file = model_dir / "results.json"
                    if results_file.exists():
                        rel_path = results_file.relative_to(input_dir)
                        output_file = output_dir / rel_path
                        if output_file.exists():
                            continue
                        results_files.append(results_file)
    return results_files


def run_vllm_setup(args: Namespace, model_cfg: ModelConfig) -> tuple[SamplingParams, Any, LLM]:
    seed_info_msg = """ SEED INFO
    vLLM Limitation: Can not set different seeds for batched requests, only global seed for whole batch => issue for repetitions: identical input -> pot. identical output!
    Decide for continuous batching to gain efficiency and thus against (maybe) reproducibility via sequential (single) requests (no batching).
    Not setting a seed defaults to None, which yields non reproducible outcomes (enabling output variance for repetitions).
    We explicitly set seed to None when initalizing SamplingParams and LLM.
    See V0 comment: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/reproducibility.py
    """
    warnings.warn(seed_info_msg)

    model_context_window = model_cfg["context_window"]

    sampling_params = SamplingParams(
        n=1,
        seed=None,
        temperature=model_cfg.get("temperature") or args.temperature,
        max_tokens=min(model_context_window, args.max_sequence_length),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])  # type: ignore[no-untyped-call]
    model = LLM(
        model_cfg["name"],
        tensor_parallel_size=max(1, args.gpus),
        trust_remote_code=True,
        seed=None,
        max_model_len=min(model_context_window, args.max_sequence_length),
        gpu_memory_utilization=0.7,
    )

    return sampling_params, tokenizer, model


def get_vllm_prompt_token_count(
    tokenizer: Any,
    model_format: Literal["text-only", "multimodal"],
    system_prompt: str,
    user_prompt: str,
) -> int:
    messages = [
        {
            "role": "system",
            "content": get_formatted_text_content(model_format=model_format, content=system_prompt),
        },
        {
            "role": "user",
            "content": get_formatted_text_content(model_format=model_format, content=user_prompt),
        },
    ]
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=get_enable_thinking(experiment_type=ExperimentType.JUDGE),
        )
    )


def make_skipped_response_meta(
    model_name: str,
    temperature: float,
    seed: int | None,
    system_prompt: str,
    user_prompt: str,
    prompt_tokens: int,
    max_prompt_tokens: int,
) -> ResponseMeta:
    return {
        "model_name": model_name,
        "temperature": temperature,
        "seed": seed,
        "system_prompts": [system_prompt],
        "user_prompts": [user_prompt],
        "response": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "total_tokens": prompt_tokens,
        "start_time": None,
        "end_time": None,
        "error": {
            "error_type": "PromptTooLongError",
            "error_message": (
                f"Skipped prompt with {prompt_tokens} tokens because it exceeds "
                f"the maximum model length of {max_prompt_tokens} tokens."
            ),
        },
    }


def main() -> None:
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    model_config_path = BASE_DIR / "resources" / "input" / "models" / args.model_selection_file
    with open(model_config_path, "r") as f:
        model_cfgs: list[ModelConfig] = json.load(f)

    if not model_cfgs:
        raise ValueError("No model configs loaded from model selection file.")

    clients = setup_clients(model_cfgs)
    async_config = AsyncGenerationConfig(max_concurrent=args.max_concurrent) if args.max_concurrent > 1 else None

    results_files = walk_results_files(input_dir, output_dir, args.datasets)
    print(f"Found {len(results_files)} results files to re-judge.")

    for model_cfg in model_cfgs:
        model_setup = model_cfg["setup"]
        model_name = model_cfg["name"]

        sampling_params = tokenizer = model = None
        if model_setup == "vLLM":
            sampling_params, tokenizer, model = run_vllm_setup(args, model_cfg)

        if model_setup == "Alternative" or model_setup == "vLLM":
            extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
        else:
            extra_body = {"reasoning": {"enabled": True}}

        for batch_start in range(0, len(results_files), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(results_files))
            batch_files = results_files[batch_start:batch_end]
            print(f"Processing batch {batch_start // args.batch_size + 1}: files {batch_start + 1}-{batch_end}")

            batch_file_data: list[tuple[Path, list[dict[str, Any]], list[int]]] = []
            pending_user_prompts: list[str] = []
            pending_response_meta_indices: list[int] = []
            all_system_prompts: list[str] = []
            response_metas_by_item: list[ResponseMeta | None] = []
            max_prompt_tokens = min(model_cfg["context_window"], args.max_sequence_length)
            skipped_prompt_count = 0

            for results_file in batch_files:
                with open(results_file, "r", encoding="utf-8") as f:
                    existing_data: list[dict[str, Any]] = json.load(f)

                file_response_meta_indices = []
                system_prompts_for_file = []
                for item in existing_data:
                    response_meta_index = len(response_metas_by_item)
                    response_metas_by_item.append(None)
                    file_response_meta_indices.append(response_meta_index)

                    if not item.get("closed_response_meta") or len(item["closed_response_meta"]) == 0:
                        warnings.warn(f"Did not find closed_response_meta in {results_file}")
                        continue

                    meta = item["closed_response_meta"][0]
                    system_prompt_for_item = meta["system_prompts"][0]
                    user_prompt = meta["user_prompts"][0]
                    system_prompts_for_file.append(system_prompt_for_item)

                    if model_setup == "vLLM":
                        assert tokenizer is not None
                        prompt_tokens = get_vllm_prompt_token_count(
                            tokenizer=tokenizer,
                            model_format=model_cfg["format"],
                            system_prompt=system_prompt_for_item,
                            user_prompt=user_prompt,
                        )
                        if prompt_tokens > max_prompt_tokens:
                            response_metas_by_item[response_meta_index] = make_skipped_response_meta(
                                model_name=model_name,
                                temperature=model_cfg.get("temperature") or args.temperature,
                                seed=None,
                                system_prompt=system_prompt_for_item,
                                user_prompt=user_prompt,
                                prompt_tokens=prompt_tokens,
                                max_prompt_tokens=max_prompt_tokens,
                            )
                            skipped_prompt_count += 1
                            continue

                    pending_response_meta_indices.append(response_meta_index)
                    pending_user_prompts.append(user_prompt)

                if not system_prompts_for_file:
                    warnings.warn(f"No prompts found in {results_file}, skipping.")
                    continue

                system_prompts_set = set(system_prompts_for_file)
                if len(system_prompts_set) > 1:
                    raise ValueError(
                        f"Found {len(system_prompts_set)} different system prompts in {results_file}, but should only be one."
                    )

                batch_file_data.append((results_file, existing_data, file_response_meta_indices))
                all_system_prompts.append(list(system_prompts_set)[0])

            if not pending_user_prompts and not any(response_metas_by_item):
                warnings.warn("No prompts found in batch, skipping.")
                continue

            all_system_prompts = list(set(all_system_prompts))
            if len(all_system_prompts) > 1:
                raise ValueError(
                    f"Found {len(all_system_prompts)} different system prompts across files in batch, but should only be one."
                )
            system_prompt = [all_system_prompts[0]]

            if skipped_prompt_count:
                warnings.warn(
                    f"Skipped {skipped_prompt_count} prompt(s) over the {max_prompt_tokens}-token vLLM limit in this batch."
                )

            if pending_user_prompts:
                generated_response_metas = generate(
                    experiment_type=ExperimentType.JUDGE,
                    client=clients.get(model_setup),
                    model_name=model_name,
                    model_format=model_cfg["format"],
                    model_setup=model_setup,
                    sampling_params=sampling_params,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=model_cfg.get("temperature") or args.temperature,
                    seed=None if model_setup == "vLLM" else args.seed,
                    repetitions=args.runs,
                    system_prompts=system_prompt,
                    batch_user_prompts=[[up] for up in pending_user_prompts],
                    use_tqdm=True,
                    use_async=True if async_config else False,
                    async_config=async_config,
                    extra_body=extra_body,
                )
                for response_meta_index, response_meta in zip(pending_response_meta_indices, generated_response_metas):
                    response_metas_by_item[response_meta_index] = response_meta

            for results_file, existing_data, response_meta_indices in batch_file_data:
                file_response_metas = [response_metas_by_item[idx] for idx in response_meta_indices]

                for item, resp_meta in zip(existing_data, file_response_metas):
                    if resp_meta is None:
                        continue

                    response_text = resp_meta.get("response")
                    judge_response = response_text or ""
                    parsed = parse_judge_response(judge_response) or {}

                    score = parsed.get("score")
                    refusal = parsed.get("refusal")

                    item["closed_response"] = [str(score) if score is not None else None]
                    item["response_value"] = str(score) if score is not None else None
                    item["refusal"] = [str(refusal) if refusal is not None else None]
                    item["valid_response_format"] = [score is not None]
                    item["valid_response_value"] = [refusal is not None or score is not None]

                    new_meta = {
                        "model_name": model_name,
                        "temperature": model_cfg.get("temperature") or args.temperature,
                        "seed": None if model_setup == "vLLM" else args.seed,
                        "system_prompts": resp_meta.get("system_prompts", []),
                        "user_prompts": resp_meta.get("user_prompts", []),
                        "response": response_text,
                        "prompt_tokens": resp_meta.get("prompt_tokens", 0),
                        "completion_tokens": resp_meta.get("completion_tokens", 0),
                        "total_tokens": resp_meta.get("total_tokens", 0),
                        "start_time": resp_meta.get("start_time"),
                        "end_time": resp_meta.get("end_time"),
                        "error": resp_meta.get("error"),
                    }

                    if item.get("closed_response_meta"):
                        item["closed_response_meta"] = [new_meta]
                    else:
                        item["closed_response_meta"] = [new_meta]

                rel_path = results_file.relative_to(input_dir)
                output_file = output_dir / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)

        if model_setup == "vLLM":
            assert model is not None
            assert tokenizer is not None
            cleanup_vram(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
