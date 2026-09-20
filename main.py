import argparse
import json

from argparse import Namespace
from pathlib import Path
from typing import Any, Type, Union
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llm_audit import BASE_DIR
from llm_audit.datasets.base import AgreementDataset, VignetteDataset
from llm_audit.datasets.util import get_dataset_by_label, get_dataset_label_class_map
from llm_audit.inference import AsyncGenerationConfig
from llm_audit.pipeline import LLMExperimentMeta, run_experiment
from llm_audit.util import (
    ExperimentType,
    ModelConfig,
    cleanup_vram,
    construct_output_dir_label,
    get_device,
    get_supported_languages,
    setup_clients,
)


def str_to_bool(text: str) -> bool:
    return text.strip().lower() in ("1", "true", "t", "y", "yes")


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM audit experiments with configurable parameters.")

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
        help="LLM inference temperature if not specified in model config (default: 1.0).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repetitions (default: 10).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "de", "ru", "zh"],
        help="Language selection for experiments (default: en).",
    )
    parser.add_argument(
        "--model_selection_file",
        type=str,
        default="complete_model_selection.json",
        help="Model selection JSON filename (default: complete_model_selection.json).",
    )
    parser.add_argument(
        "--judge_selection_file",
        type=str,
        default="judges_or_alt.json",
        help="Judge model selection JSON filename (default: judges_or_alt.json).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(get_dataset_label_class_map(filter_test_and_vignettes=False).keys()),
        default=list(get_dataset_label_class_map(filter_test_and_vignettes=True).keys()),
        help="List of datasets to perform experiments with (default: F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS).",
    )
    parser.add_argument(
        "--output_dir_prefix_tag",
        type=str,
        default="final",
        help="Custom prefix tag of output directory for the corresponding experiment (default: final).",
    )
    parser.add_argument(
        "--reverse_scale",
        type=str_to_bool,
        default="False",
        help="Reverse Likert scale items (default: False).",
    )
    parser.add_argument(
        "--auth_sys_prompt",
        type=str_to_bool,
        default="False",
        help="Replace default system prompt with malicious authoritarian system prompt (default: False).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=16000,
        help="Limit the context length of the model.",
    )
    parser.add_argument(
        "--max_concurrent",
        default=1,
        type=int,
        help="Values larger than 1 will lead to async calls to API. Maximal number of open requests.",
    )
    parser.add_argument("--open_q_prompt_style", choices=["short", "long"], default="short", type=str)
    parser.add_argument("--skip-vignettes", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.language not in get_supported_languages():
        raise ValueError(f"Language {args.language} not supported!")

    assert args.seed is not None  # Can become None after output_dir is initialized
    output_dir: Path = (
        BASE_DIR
        / "resources"
        / "output"
        / construct_output_dir_label(
            output_dir_prefix_tag=args.output_dir_prefix_tag,
            language=args.language,
            temperature=args.temperature,
            runs=args.runs,
            seed=args.seed,
        )
    )

    async_config = AsyncGenerationConfig(max_concurrent=args.max_concurrent) if args.max_concurrent > 1 else None

    # Load model information
    model_dicts_file_path = BASE_DIR / "resources" / "input" / "models" / args.model_selection_file
    with open(model_dicts_file_path, "r") as f:
        model_cfgs: list[ModelConfig] = json.load(f)

    # Load judge info
    judge_dicts_file_path = BASE_DIR / "resources" / "input" / "models" / args.judge_selection_file
    with judge_dicts_file_path.open("r") as f:
        judge_cfgs: list[ModelConfig] = json.load(f)

    clients = setup_clients(model_cfgs + judge_cfgs)

    if any(model_cfg["setup"] == "vLLM" for model_cfg in judge_cfgs):
        raise NotImplementedError("Judges must be called via API, but at least one of them is a vLLM model.")
    for model_cfg in judge_cfgs:
        # Test if the API works
        client = clients[model_cfg["setup"]]
        response = client.chat.completions.create(
            model=model_cfg["name"],
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=16,
            timeout=60,
            extra_body={"reasoning": {"enabled": False}},
        )
        if (
            response is None
            or not response.choices
            or not response.choices[0].message
            or not response.choices[0].message.content
        ):
            raise ValueError(f"Invalid API response for {model_cfg=}. Is your API key valid?")

    # Init datasets
    args.datasets = set(args.datasets) - set(("VignetteRWA3D",)) if args.skip_vignettes else args.datasets
    datasets: list[AgreementDataset | VignetteDataset] = []
    dataset_label_class_map: dict[str, Type[Union[AgreementDataset, VignetteDataset]]] = get_dataset_label_class_map(
        filter_test_and_vignettes=False
    )
    for dataset_label in args.datasets:
        DatasetClass = dataset_label_class_map.get(dataset_label)
        if DatasetClass is None:
            raise ValueError(f"Invalid dataset label: {dataset_label}")
        if issubclass(DatasetClass, AgreementDataset):
            datasets.append(get_dataset_by_label(dataset_label=dataset_label, reverse_scale=args.reverse_scale))
        else:
            # VignetteDataset applies shuffling
            datasets.append(get_dataset_by_label(dataset_label=dataset_label, reverse_scale=False))

    for model_cfg in model_cfgs:
        model_setup = model_cfg["setup"]
        model_name = model_cfg["name"]
        temperature = model_cfg.get("temperature") or args.temperature

        # vLLM specific
        model_context_window = None  # w.r.t. output if specified in model card
        sampling_params: SamplingParams | None = None
        tokenizer: Any | None = None
        model: LLM | None = None

        try:
            if model_setup == "vLLM":
                """ SEED INFO
                vLLM Limitation: Can not set different seeds for batched requests, only global seed for whole batch => issue for repetitions: identical input -> pot. identical output!
                Decide for continuous batching to gain efficiency and thus against (maybe) reproducibility via sequential (single) requests (no batching).
                Not setting a seed defaults to None, which yields non reproducible outcomes (enabling output variance for repetitions).
                We explicitly set seed to None when initalizing SamplingParams and LLM.
                See V0 comment: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/reproducibility.py
                """
                args.seed = None

                # TODO Optimization, lookup and skip if (follow-up) computations have already been done.
                # use complete_results_exist from llm_audit.datasets.util
                model_context_window = model_cfg["context_window"]

                # For default params (e.g., top_k) see: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
                sampling_params = SamplingParams(
                    n=1,  # Number of output sequences
                    seed=args.seed,  # seed=None, see <SEED INFO> comment above
                    temperature=temperature,
                    max_tokens=min(model_context_window, args.max_sequence_length),
                )

                tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore
                # TODO: for continuous batching, set: max_num_seqs: int = 256 (default) to 30?
                # max_num_seqs :: maximum number of sequences per iteration
                model = LLM(
                    model_name,
                    tensor_parallel_size=max(
                        1, args.gpus
                    ),  # 1 :: no distributed initialization, 2 :: 2 GPUs, ..., n :: n GPUs
                    trust_remote_code=True,
                    seed=args.seed,  # seed=None, see <SEED INFO> comment above
                    max_model_len=min(model_context_window, args.max_sequence_length),
                )

            for dataset in datasets:
                ids = dataset.get_ids(language=args.language)

                for experiment_type in dataset.get_valid_experiment_types():
                    results_flattened = []
                    for id in ids:
                        results: list[LLMExperimentMeta] = run_experiment(
                            experiment_type=experiment_type,
                            clients=clients,
                            model_cfg=model_cfg,
                            judge_cfgs=judge_cfgs,
                            sampling_params=sampling_params,
                            model=model,
                            tokenizer=tokenizer,
                            temperature=temperature,
                            seed=args.seed,
                            language=args.language,
                            dataset=dataset,
                            id=id,
                            output_dir=output_dir,
                            auth_sys_prompt=args.auth_sys_prompt,
                            use_tqdm=True,
                            use_async=True if async_config else False,
                            async_config=async_config,
                            num_runs=args.runs,
                            prompt_style=args.open_q_prompt_style,
                        )
                        results_flattened += results

                        # Store non case vignette results if results is not empty
                        # If results = [], then skip saving step, otherwise the previous results would be overwritten with an empty list!
                        if results and experiment_type != ExperimentType.CASE_VIGNETTE:
                            assert isinstance(dataset, AgreementDataset), (
                                "Expected AgreementDataset for CLOSED_QUESTION!"
                            )
                            assert not isinstance(dataset, VignetteDataset), (
                                "Expected AgreementDataset, not VignetteDataset!"
                            )
                            results_file_path = (
                                output_dir
                                / dataset.get_label()
                                / experiment_type.value
                                / str(id)
                                / model_name
                                / "results.json"
                            )
                            results_file_path.parent.mkdir(parents=True, exist_ok=True)

                            with results_file_path.open("w", encoding="utf-8") as f:
                                json.dump(results, f, ensure_ascii=False, indent=4)

                    # Store flattened results of case vignettes, concat by item_id
                    if experiment_type == ExperimentType.CASE_VIGNETTE:
                        assert isinstance(dataset, VignetteDataset), "Expected VignetteDataset for CASE_VIGNETTE!"
                        assert not isinstance(dataset, AgreementDataset), (
                            "Expected VignetteDataset, not AgreementDataset!"
                        )
                        # Concat flattened results
                        n_item_vignette_stats = dataset.get_item_vignette_stats(language=args.language)
                        l = 0
                        while l < len(results):
                            for item_id, n_vignettes in sorted(n_item_vignette_stats.items()):
                                vignette_item_results = []  # vignettes x (re)runs; for corresponding scale item (item_id)
                                for _ in range(n_vignettes * args.runs):
                                    vignette_item_results.append(results_flattened[l])
                                    l += 1
                                results_file_path = (
                                    output_dir
                                    / dataset.get_label()
                                    / experiment_type.value
                                    / str(item_id)
                                    / model_name
                                    / "results.json"
                                )
                                results_file_path.parent.mkdir(parents=True, exist_ok=True)

                                with results_file_path.open("w", encoding="utf-8") as f:
                                    json.dump(
                                        vignette_item_results,
                                        f,
                                        ensure_ascii=False,
                                        indent=4,
                                    )
        finally:
            # Cleanup
            if model_setup == "vLLM":
                assert model is not None
                assert tokenizer is not None
                cleanup_vram(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
