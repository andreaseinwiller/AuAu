import argparse
import json
import shutil
import warnings

from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_audit import BASE_DIR
from llm_audit.eval.open_response import parse_judge_response
from llm_audit.inference import AsyncGenerationConfig, ResponseMeta, generate
from llm_audit.util import ExperimentType, ModelConfig, cleanup_vram, setup_clients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry failed closed-response LLM requests in existing results.json files. "
            "Each touched file is first copied to backup.json."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory to scan for results.json files.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent API requests for OpenRouter/Alternative backends.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of failed requests to retry together within a group.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=("Model name to use for all retries. When provided, failed request metadata model names are ignored."),
    )
    parser.add_argument(
        "--model-selection-file",
        type=str,
        default=None,
        help=(
            "Optional model config JSON filename under resources/input/models/ to load "
            "the model from. If omitted, all model config files are searched."
        ),
    )
    return parser.parse_args()


def load_model_configs(
    model_selection_file: str | None = None,
    model_name_filter: str | None = None,
) -> dict[str, ModelConfig]:
    model_dir = BASE_DIR / "resources" / "input" / "models"
    config_paths = [model_dir / model_selection_file] if model_selection_file else sorted(model_dir.glob("*.json"))
    configs_by_name: dict[str, list[ModelConfig]] = defaultdict(list)

    for path in config_paths:
        if not path.exists():
            raise FileNotFoundError(f"Model selection file does not exist: {path}")
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for cfg in loaded:
            if model_name_filter is not None and cfg["name"] != model_name_filter:
                continue
            configs_by_name[cfg["name"]].append(cfg)

    resolved: dict[str, ModelConfig] = {}
    for model_name, cfgs in configs_by_name.items():
        unique_cfgs = {json.dumps(cfg, sort_keys=True, ensure_ascii=False): cfg for cfg in cfgs}
        if len(unique_cfgs) > 1:
            raise ValueError(f"Found multiple model configs for {model_name}: {list(unique_cfgs.values())}")
        resolved[model_name] = next(iter(unique_cfgs.values()))
    return resolved


def build_extra_body(model_setup: str) -> dict[str, Any]:
    if model_setup in {"Alternative", "vLLM"}:
        return {"chat_template_kwargs": {"enable_thinking": True}}
    return {"reasoning": {"enabled": True}}


def find_failed_requests(
    root: Path,
    model_name_override: str | None = None,
) -> tuple[
    dict[Path, list[dict[str, Any]]],
    dict[tuple[Any, ...], list[dict[str, Any]]],
]:
    file_cache: dict[Path, list[dict[str, Any]]] = {}
    grouped_failures: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for results_path in sorted(root.rglob("results.json")):
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        file_cache[results_path] = data

        for item_index, item in enumerate(data):
            metas = item.get("closed_response_meta")
            if not metas:
                continue

            meta = metas[0]
            if not meta.get("error"):
                continue

            system_prompts = meta.get("system_prompts") or []
            user_prompts = meta.get("user_prompts") or []
            if not system_prompts and not user_prompts:
                warnings.warn(f"Skipping {results_path} item {item_index}: missing prompts in closed_response_meta.")
                continue

            original_model_name = meta["model_name"]
            group_key = (
                model_name_override or original_model_name,
                tuple(system_prompts),
                meta.get("temperature"),
                meta.get("seed"),
            )
            grouped_failures[group_key].append(
                {
                    "results_path": results_path,
                    "item_index": item_index,
                    "user_prompts": user_prompts,
                    "original_model_name": original_model_name,
                }
            )

    return file_cache, grouped_failures


def update_item_with_retry(
    item: dict[str, Any],
    response_meta: ResponseMeta,
    original_model_name: str | None = None,
) -> None:
    judge_response = response_meta.get("response", "") or ""
    parsed = parse_judge_response(judge_response)
    assert parsed is not None

    score = parsed.get("score")
    refusal = parsed.get("refusal")

    item["closed_response"] = [str(score) if score is not None else None]
    item["response_value"] = str(score) if score is not None else None
    item["refusal"] = [str(refusal) if refusal is not None else None]
    item["valid_response_format"] = [score is not None]
    item["valid_response_value"] = [refusal is not None or score is not None]
    updated_meta = dict(response_meta)
    if original_model_name is not None:
        updated_meta["model_name"] = original_model_name
    item["closed_response_meta"] = [updated_meta]


def write_updated_files(file_cache: dict[Path, list[dict[str, Any]]], touched_files: set[Path]) -> None:
    for results_path in sorted(touched_files):
        backup_path = results_path.with_name("backup.json")
        if not backup_path.exists():
            shutil.copy2(results_path, backup_path)

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(file_cache[results_path], f, ensure_ascii=False, indent=4)


def main() -> None:
    args = parse_args()
    root = args.directory.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    model_configs = load_model_configs(args.model_selection_file, args.model_name)
    if args.model_name is not None and args.model_name not in model_configs:
        raise ValueError(f"No matching model config found for --model-name {args.model_name!r}.")

    model_name_override = args.model_name
    if model_name_override is None and args.model_selection_file and len(model_configs) == 1:
        model_name_override = next(iter(model_configs))

    file_cache, grouped_failures = find_failed_requests(root, model_name_override)

    total_failures = sum(len(group) for group in grouped_failures.values())
    print(f"Found {total_failures} failed closed-response requests across {len(file_cache)} files.")
    if total_failures == 0:
        return

    missing_models = sorted(
        {model_name for model_name, *_ in grouped_failures.keys() if model_name not in model_configs}
    )
    if missing_models:
        raise ValueError(f"No matching model config found for failed requests: {missing_models}")

    clients = setup_clients([model_configs[model_name] for model_name, *_ in grouped_failures.keys()])
    async_config = AsyncGenerationConfig(max_concurrent=args.max_concurrent) if args.max_concurrent > 1 else None

    touched_files: set[Path] = set()
    sorted_groups = sorted(
        grouped_failures.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]),
    )

    active_vllm_model_name: str | None = None
    sampling_params = None
    tokenizer = None
    model = None

    try:
        for group_key, failures in sorted_groups:
            model_name, system_prompts_t, temperature, seed = group_key
            model_cfg = model_configs[model_name]
            model_setup = model_cfg["setup"]
            model_format = model_cfg["format"]

            if model_setup == "vLLM":
                if active_vllm_model_name != model_name:
                    if model is not None and tokenizer is not None:
                        cleanup_vram(model=model, tokenizer=tokenizer)
                    from rejudge_open_responses import run_vllm_setup

                    vllm_args = argparse.Namespace(
                        temperature=temperature if temperature is not None else 1.0,
                        max_sequence_length=model_cfg["context_window"],
                        gpus=1,
                    )
                    sampling_params, tokenizer, model = run_vllm_setup(vllm_args, model_cfg)
                    active_vllm_model_name = model_name
            elif model is not None and tokenizer is not None:
                cleanup_vram(model=model, tokenizer=tokenizer)
                active_vllm_model_name = None
                sampling_params = None
                tokenizer = None
                model = None

            for batch_start in range(0, len(failures), args.batch_size):
                batch = failures[batch_start : batch_start + args.batch_size]
                response_metas = generate(
                    experiment_type=ExperimentType.JUDGE,
                    client=clients.get(model_setup),
                    model_name=model_name,
                    model_format=model_format,
                    model_setup=model_setup,
                    sampling_params=sampling_params,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature if temperature is not None else 1.0,
                    seed=seed,
                    repetitions=1,
                    system_prompts=list(system_prompts_t),
                    batch_user_prompts=[failure["user_prompts"] for failure in batch],
                    use_tqdm=True,
                    use_async=bool(async_config and model_setup in {"OpenRouter", "Alternative"}),
                    async_config=async_config,
                    extra_body=build_extra_body(model_setup),
                )

                for failure, response_meta in zip(batch, response_metas):
                    results_path = failure["results_path"]
                    item_index = failure["item_index"]
                    item = file_cache[results_path][item_index]
                    update_item_with_retry(
                        item,
                        response_meta,
                        original_model_name=failure["original_model_name"],
                    )
                    touched_files.add(results_path)

        write_updated_files(file_cache, touched_files)
        print(f"Updated {len(touched_files)} files.")
    finally:
        if model is not None and tokenizer is not None:
            cleanup_vram(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
