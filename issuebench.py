import argparse
import json
import random
import warnings
import datasets  # type: ignore[import-untyped]

from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, cast
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llm_audit import BASE_DIR
from llm_audit.inference import AsyncGenerationConfig, generate
from llm_audit.util import (
    ExperimentType,
    ModelConfig,
    cleanup_vram,
    construct_output_dir_label,
    get_auth_system_prompt,
    get_default_system_prompt,
    setup_clients,
)


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM audit experiments with configurable parameters.")
    parser.add_argument("experiment_type", choices=["gen", "judge", "gen-high-risk", "eval"])
    parser.add_argument("-v", "--prompt-version", default="v8")
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
        "--input-dir",
        type=str,
        help="Path to directory with responses to Issuebench prompt. Only needed when judging.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="issuebench_gen",
        help="Custom prefix tag of output directory for the corresponding experiment (default: issuebench_gen).",
    )
    parser.add_argument(
        "--auth-sys-prompt",
        action="store_true",
        default=False,
        help="Replace default system prompt with malicious authoritarian system prompt (default: False).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=32000,
        help="Limit the context length of the model. (default: 32000)",
    )
    parser.add_argument(
        "--num-prompts",
        default=10000,
        type=int,
        help="For how many prompts answers should be generated. Random selection. Only needed when generating. (default: 1000)",
    )
    parser.add_argument(
        "--max-concurrent",
        default=1,
        type=int,
        help="Values larger than 1 will lead to async calls to API. Maximal number of open requests.",
    )
    parser.add_argument("-l", "--language", choices=["en", "ru", "de", "zh"], default="en")
    parser.add_argument(
        "-t",
        "--prompt-template",
        default="instruct-first",
        choices=["instruct-first", "text-first"],
    )
    return parser.parse_args()


def filter_existing_results(
    inputs: list[str],
    input_metas: list[dict[str, Any]],
    output_root: Path,
    experiment_type: ExperimentType,
    current_model_name: str,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Checks if results already exist for specific inputs.
    Returns:
        - inputs_to_run: List of prompts that failed or haven't been run.
        - metas_to_run: Corresponding metadata.
        - existing_results: List of result dicts that were successful and loaded from disk.
    """
    inputs_to_run = []
    metas_to_run = []
    existing_results = []

    # Cache loaded JSON files to avoid re-reading for every input
    # Key: str (filepath), Value: dict (prompt -> result_entry)
    file_cache: dict[str, dict[str, dict[str, Any]]] = {}

    print("Checking for existing results to resume...")

    for inp, meta in zip(inputs, input_metas):
        # Determine where this specific input's result should be stored
        # TODO: this sucks, because it is tightly coupled with the saving part in main()
        if experiment_type == ExperimentType.GENERATE:
            # Path: .../IssueBench/gen/ModelName/results.json
            results_file_path = output_root / "IssueBench" / experiment_type.value / current_model_name / "results.json"
        elif experiment_type == ExperimentType.JUDGE:
            # Path: .../IssueBench/judge/JudgeName/TargetModelName/results.json
            target_model_name = meta.get("generating_model_name") or "unknown_model"
            target_model_safe = target_model_name.replace("/", "_")
            results_file_path = (
                output_root
                / "IssueBench"
                / experiment_type.value
                / current_model_name  # The Judge
                / target_model_safe  # The Target
                / "results.json"
            )

        str_path = str(results_file_path)

        # Load file if not in cache
        if str_path not in file_cache:
            if results_file_path.exists():
                try:
                    with results_file_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Create a lookup map: prompt string -> entry object
                        # We use the first user_prompt as the key
                        lookup = {}
                        for entry in data:
                            if "user_prompts" in entry and len(entry["user_prompts"]) > 0:
                                lookup[entry["user_prompts"][0]] = entry
                        file_cache[str_path] = lookup
                except json.JSONDecodeError:
                    warnings.warn(f"Could not decode JSON at {str_path}. Will re-run these.")
                    file_cache[str_path] = {}
            else:
                file_cache[str_path] = {}

        # Check if this input exists in the loaded data
        existing_entry = file_cache[str_path].get(inp)

        should_rerun = False

        if existing_entry is None:
            # Not found
            should_rerun = True
        else:
            # Found, check for errors
            # Schema says "error" object exists if failed
            if "error" in existing_entry and existing_entry["error"] is not None:
                should_rerun = True
            # Also check if response is accidentally null/empty without explicit error
            elif not existing_entry.get("response"):
                should_rerun = True
            else:
                # Valid existing entry
                existing_results.append(existing_entry)

        if should_rerun:
            inputs_to_run.append(inp)
            metas_to_run.append(meta)

    return inputs_to_run, metas_to_run, existing_results


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
    if experiment_type == ExperimentType.GENERATE and args.experiment_type == "gen":
        inputs, input_metas = prepare_generation_data(args)
    elif experiment_type == ExperimentType.GENERATE and args.experiment_type == "gen-high-risk":
        inputs, input_metas = prepare_high_risk_generation_data(args)
    elif experiment_type == ExperimentType.JUDGE and args.experiment_type == "judge":
        inputs, input_metas = prepare_judgement_data(args)
    else:  # judge eval
        inputs, input_metas = prepare_judge_eval_data(args)

    for model_cfg in model_cfgs:
        model_setup = model_cfg["setup"]
        model_name = model_cfg["name"]

        # --- NEW: Filter inputs to only run failed or missing ones ---
        inputs_to_run, metas_to_run, previous_results = filter_existing_results(
            inputs=inputs,
            input_metas=input_metas,
            output_root=output_root,
            experiment_type=experiment_type,
            current_model_name=model_name,
        )

        print(
            f"Model: {model_name} | Total inputs: {len(inputs)} | To run: {len(inputs_to_run)} | Skipped (valid): {len(previous_results)}"
        )

        if len(inputs_to_run) == 0:
            print(f"Skipping generation for {model_name}, all results exist and are valid.")
            # Even if we skip generation, we might want to ensure the files are saved
            # (though they should already exist).
            # If we strictly skip, we don't re-save.
            continue

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
            system_prompts=[
                get_auth_system_prompt(language=args.language)
                if args.auth_sys_prompt
                else get_default_system_prompt(language=args.language)
            ],
            batch_user_prompts=[[inp] for inp in inputs_to_run],
            use_tqdm=True,
            use_async=True if async_config else False,
            async_config=async_config,
            extra_body={
                "reasoning": {"enabled": True},
                "provider": {
                    "sort": {"by": "price"},
                    "preferred_min_throughput": 90,
                },
            }
            if model_cfg["name"]
            in [
                "moonshotai/kimi-k2-thinking",
                "openai/gpt-5.1",
                "google/gemini-3-pro-preview",
            ]
            else None,
        )

        # Verify data integrity
        if len(response_metas) != len(metas_to_run):
            # If runs > 1, this logic might need adaptation based on how generate returns lists
            warnings.warn(
                f"Results length ({len(response_metas)}) does not match Metadata length ({len(metas_to_run)}). Metadata merging may be misaligned if runs > 1."
            )

        # 1. Attach metadata to NEW results
        new_results_with_meta = []
        for res, meta in zip(response_metas, metas_to_run):
            _res: dict[str, Any] = dict(res)
            _res.update(meta)
            new_results_with_meta.append(_res)

        # 2. Combine NEW results with PREVIOUS successful results
        # We need to re-group everything because output files are split by target model
        combined_results = new_results_with_meta + previous_results

        # Group results by the model that generated the original response
        grouped_results = defaultdict(list)

        for _res in combined_results:
            # Extract the name of the model being judged
            if "generating_model_name" in _res:
                target_model_name = _res["generating_model_name"]
            else:
                # Fallback for Generate mode where metadata might be empty or different
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

            with results_file_path.open("w", encoding="utf-8") as f:
                json.dump(model_results, f, ensure_ascii=False, indent=4)

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


def prepare_judgement_data(args: Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    if not args.input_dir:
        raise ValueError("--input-dir is required when judging")

    if args.language == "ru":
        prompt_path = (
            BASE_DIR / "resources" / "input" / "datasets" / "issuebench" / f"prompt_{args.prompt_version}_ru.txt"
        )
    else:
        prompt_path = BASE_DIR / "resources" / "input" / "datasets" / "issuebench" / f"prompt_{args.prompt_version}.txt"
    with prompt_path.open() as f:
        instruction = f.read()

    data_dir = Path(args.input_dir)
    combined_data = []

    for fpath in data_dir.rglob("*.json"):
        with fpath.open("r") as f:
            data = json.load(f)

            for item in data:
                prompt = f"{instruction}\n{item['response']}"
                meta = {"generating_" + key: val for key, val in item.items() if key != "response"}
                combined_data.append((prompt, meta))

    if not combined_data:
        raise ValueError("Did not find json result files in --input-dir")

    combined_data.sort(key=lambda pair: len(pair[0]))
    inputs, metas = zip(*combined_data)
    return list(inputs), list(metas)


def prepare_judge_eval_data(args: Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    prompt_path = BASE_DIR / "resources" / "input" / "datasets" / "issuebench" / f"prompt_{args.prompt_version}.txt"
    with prompt_path.open() as f:
        instruction = f.read()

    language = args.language
    print(f"Loading ground truth data with responses in language: {language}")
    data_path = (
        BASE_DIR / "resources" / "input" / "datasets" / "issuebench" / f"ground_truth_data_rebuttal_{language}.json"
    )
    with data_path.open() as f:
        data = json.load(f)

    combined_data = []
    response_key = "response" if language == "en" else f"response_{language}"

    for item in data:
        if args.prompt_template == "instruct-first":
            prompt = f"{instruction}\n{item[response_key]}"
        else:
            prompt = f"TEXT:\n{item[response_key]}\n\nINSTRUCTIONS:\n{instruction}"

        meta = {
            "generating_model": item["model"],
            "label_aggression": item["label_aggression"],
            "label_submission": item["label_submission"],
            "label_conventionalism": item["label_conventionalism"],
            "tag": item.get("tag"),
        }
        combined_data.append((prompt, meta))

    combined_data.sort(key=lambda pair: len(pair[0]))
    inputs, metas = zip(*combined_data)
    return list(inputs), list(metas)


def prepare_generation_data(args: Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    random.seed(args.seed)
    dataset = datasets.load_dataset("Paul/IssueBench")
    neutral_prompts = cast(datasets.Dataset, dataset["prompts_sample"]).filter(
        lambda example: example["topic_polarity"] == "neutral"
    )
    selected_prompt_idxs = random.sample(range(len(neutral_prompts)), k=args.num_prompts)
    selected_neutral_prompts = neutral_prompts.select(selected_prompt_idxs)

    if args.language != "en":
        if args.seed != 42 or args.num_prompts != 1000:
            raise ValueError("Translations only available with 1000 prompts gotten with seed 42.")

        with open(
            BASE_DIR / "resources" / "input" / "datasets" / "issuebench" / f"translations_{args.language}.json"
        ) as f:
            translations = json.load(f)
        inputs = [x["text"] for x in translations]
    else:
        inputs = list(selected_neutral_prompts["prompt_text"])
    metas: list[dict[str, Any]] = [{} for _ in inputs]
    return inputs, metas


def prepare_high_risk_generation_data(
    args: Namespace,
) -> tuple[list[str], list[dict[str, Any]]]:
    with open(BASE_DIR / "resources/output/prompt_ratings/manual_selection.json") as f:
        inputs = json.load(f)

    if args.language == "ru":
        inputs = [prompt + "\nAnswer in Russian." for prompt in inputs]
    metas: list[dict[str, Any]] = [{} for _ in inputs]
    return inputs, metas


if __name__ == "__main__":
    main()
