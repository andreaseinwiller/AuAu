import json

import pandas as pd

from tqdm import tqdm
from functools import lru_cache
from typing import Any, Literal, cast
from pathlib import Path

from llm_audit import BASE_DIR
from llm_audit.util import construct_output_dir_label
from llm_audit.datasets.base import AgreementDataset
from llm_audit.datasets.util import (
    get_dataset_by_label,
    complete_results_exist_across_models,
    get_dataset_label_class_map,
    contains_polarity_or_inverted_variable,
)
from llm_audit.eval.open_response import (
    normalize_raw_likert_scale_score,
    parse_and_validate_judge_response,
)

temperature: float = 1.0
runs: int = 10
seed: int = 42
model_selection_file: str = "final_complete.json"
output_dir_prefix_tags: list[str] = ["final", "final-authsys", "final-reverse"]
experiment_ablations: list[str] = ["default", "authsys", "reverse"]
assert len(output_dir_prefix_tags) == len(experiment_ablations), "missing prefix or experiment run label"
experiment_type_labels = ["open_question", "closed_question"]
rejudge_results_filenames: list[str] = ["results_reprocessed.json", "results.json"]


@lru_cache(maxsize=None)
def get_open_question_rejudge_root_paths(target_dir_label: str) -> list[Path]:
    output_dir = BASE_DIR / "resources" / "output"
    return sorted(path for path in output_dir.glob(f"{target_dir_label}-*rejudge") if path.is_dir())


def get_preferred_results_file(result_dir: Path) -> Path | None:
    for filename in rejudge_results_filenames:
        results_file = result_dir / filename
        if results_file.exists():
            return results_file
    return None


def load_results_file(results_file: Path) -> list[dict[str, Any]]:
    with open(results_file, "r") as f:
        data: list[dict[str, Any]] = json.load(f)
        return data


def get_judge_response_text(run: dict[str, Any], results_file: Path) -> str:
    response_key = "response_without_thinking" if results_file.name == "results_reprocessed.json" else "response"
    return cast(str, run["closed_response_meta"][0][response_key])


def get_preferred_rejudge_results_file(
    rejudge_root_paths: list[Path],
    dataset_label: str,
    experiment_type_label: str,
    id: str,
    model_label: str,
) -> Path | None:
    for filename in rejudge_results_filenames:
        for rejudge_root_path in rejudge_root_paths:
            results_file = rejudge_root_path / dataset_label / experiment_type_label / id / model_label / filename
            if results_file.exists():
                return results_file
    return None


def get_validated_score_auth_and_refusal(
    run: dict[str, Any],
    results_file: Path,
    dataset_label: str,
    experiment_type_label: str,
    id: str,
) -> tuple[float | None, int | None, int | None]:
    dataset = get_dataset_by_label(dataset_label=dataset_label)
    assert isinstance(dataset, AgreementDataset)
    response_pattern = (
        dataset.get_open_response_pattern()
        if experiment_type_label == "open_question"
        else dataset.get_closed_response_pattern()
    )
    response_text = get_judge_response_text(run=run, results_file=results_file)
    parsed = parse_and_validate_judge_response(response_text, dataset_label=dataset_label, pattern=response_pattern)
    normalized = normalize_raw_likert_scale_score(
        dataset_label=dataset_label,
        id=int(id),
        raw_score=parsed["score"],
        refusal=parsed["refusal"],
    )
    return normalized["normalized_score"], normalized["auth"], parsed["refusal"]


def get_ensemble_scores_and_auth_labels(
    rejudge_root_paths: list[Path],
    dataset_label: str,
    language: str,
    id: str,
    model_label: str,
    dataset: Any,
) -> tuple[list[float | None], list[int | None]] | None:
    judge_results: list[tuple[Path, list[dict[str, Any]]]] = []
    for rejudge_root_path in rejudge_root_paths:
        result_dir = rejudge_root_path / dataset_label / "open_question" / id / model_label
        results_file = get_preferred_results_file(result_dir=result_dir)
        if results_file is None:
            return None
        judge_results.append((results_file, load_results_file(results_file=results_file)))

    if not judge_results:
        return None

    run_count = len(judge_results[0][1])
    if any(len(results) != run_count for _, results in judge_results):
        return None

    scores: list[float | None] = []
    auth_labels: list[int | None] = []
    for run_idx in range(run_count):
        judge_score_auth_refusals = [
            get_validated_score_auth_and_refusal(
                run=results[run_idx],
                results_file=results_file,
                dataset_label=dataset_label,
                experiment_type_label="open_question",
                id=id,
            )
            for results_file, results in judge_results
        ]
        judge_scores = [score for score, _, _ in judge_score_auth_refusals]
        judge_auth_labels = [auth for _, auth, _ in judge_score_auth_refusals]
        usable_judge_scores = [score for score in judge_scores if pd.notna(score)]
        scores.append(min(usable_judge_scores) if len(usable_judge_scores) == len(judge_scores) else None)
        if any(pd.isna(auth) for auth in judge_auth_labels):
            auth_labels.append(None)
        else:
            auth_labels.append(1 if all(auth == 1 for auth in judge_auth_labels) else 0)

    return scores, auth_labels


model_selection_file_path = BASE_DIR / "resources" / "input" / "models" / model_selection_file
with open(model_selection_file_path, "r") as f:
    models = json.load(f)
# Sort by group lexicographically, then by name lexicographically
models.sort(key=lambda m: (m["group"], m["name"]))
model_labels = [model["name"] for model in models]

# languages = get_supported_languages()
languages = ["en"]
dataset_labels: list[str] = list(get_dataset_label_class_map().keys())

data = []

total_result_files = 0
for dataset_label in dataset_labels:
    dataset = get_dataset_by_label(dataset_label=dataset_label)
    for language in languages:
        _ids = dataset.get_ids(language=language)
        ablation_count = sum(
            1
            for experiment_ablation in experiment_ablations
            if not (experiment_ablation != "default" and language != "en")
        )
        total_result_files += len(experiment_type_labels) * len(model_labels) * len(_ids) * ablation_count

with tqdm(total=total_result_files, desc="Constructing tidy dataframe", unit="file") as progress_bar:
    for experiment_type_label in experiment_type_labels:
        for dataset_label in dataset_labels:
            dataset = get_dataset_by_label(dataset_label=dataset_label)
            progress_bar.set_postfix(experiment=experiment_type_label, dataset=dataset_label, refresh=False)
            for language in languages:
                target_dir_labels: list[str] = []
                for output_dir_prefix_tag in output_dir_prefix_tags:
                    _target_dir_label = construct_output_dir_label(
                        output_dir_prefix_tag=output_dir_prefix_tag,
                        language=language,
                        temperature=temperature,
                        runs=runs,
                        seed=seed,
                    )
                    target_dir_labels.append(_target_dir_label)

                target_dir_paths: list[Path] = []
                for target_dir_label in target_dir_labels:
                    _target_dir_path = (
                        BASE_DIR / "resources" / "output" / target_dir_label / dataset_label / experiment_type_label
                    )
                    target_dir_paths.append(_target_dir_path)

                for target_dir_label, experiment_ablation in zip(target_dir_labels, experiment_ablations):
                    # Assumption: reverse and authsys only en
                    if not (experiment_ablation != "default" and language != "en"):
                        assert complete_results_exist_across_models(
                            model_names=model_labels,
                            experiment_output_dir_label=target_dir_label,
                            dataset_label=dataset_label,
                            experiment_type_label=experiment_type_label,
                            language=language,
                            verbose=True,
                        )

                dataset = get_dataset_by_label(dataset_label=dataset_label)
                assert isinstance(dataset, AgreementDataset)
                ids: list[str] = [str(id) for id in dataset.get_ids(language=language)]

                for model_label in model_labels:
                    for id in ids:
                        factor = dataset.get_factor(language=language, id=int(id))
                        reverse_scored_item: Literal["+", "-"] | None = dataset.get_polarity(
                            language=language,
                            id=int(id),
                            polarity_literal=contains_polarity_or_inverted_variable(dataset=dataset, language=language),
                        )
                        # default "+" poalarized / not "-" polarized / not reverse scored / not inverted
                        reverse_scored = 0 if reverse_scored_item is None or reverse_scored_item == "+" else 1

                        for target_dir_label, target_dir_path, experiment_ablation in zip(
                            target_dir_labels, target_dir_paths, experiment_ablations
                        ):
                            # Assumption: reverse and authsys only en
                            if experiment_ablation != "default" and language != "en":
                                continue

                            results_file = target_dir_path / str(id) / model_label / "results.json"
                            ensemble_scores_and_auth_labels = None

                            if experiment_type_label == "open_question":
                                rejudge_root_paths = get_open_question_rejudge_root_paths(
                                    target_dir_label=target_dir_label
                                )
                                if rejudge_root_paths:
                                    ensemble_scores_and_auth_labels = get_ensemble_scores_and_auth_labels(
                                        rejudge_root_paths=rejudge_root_paths,
                                        dataset_label=dataset_label,
                                        language=language,
                                        id=str(id),
                                        model_label=model_label,
                                        dataset=dataset,
                                    )
                                    if ensemble_scores_and_auth_labels is not None:
                                        rejudge_results_file = get_preferred_rejudge_results_file(
                                            rejudge_root_paths=rejudge_root_paths,
                                            dataset_label=dataset_label,
                                            experiment_type_label=experiment_type_label,
                                            id=str(id),
                                            model_label=model_label,
                                        )
                                        if rejudge_results_file is not None:
                                            results_file = rejudge_results_file

                            results = load_results_file(results_file=results_file)
                            for run_idx, run in enumerate(results):
                                adjusted_dir_aligned_normalized_score, auth, refusal = (
                                    get_validated_score_auth_and_refusal(
                                        run=run,
                                        results_file=results_file,
                                        dataset_label=dataset_label,
                                        experiment_type_label=experiment_type_label,
                                        id=str(id),
                                    )
                                )
                                if ensemble_scores_and_auth_labels is not None:
                                    ensemble_scores, ensemble_auth_labels = ensemble_scores_and_auth_labels
                                    adjusted_dir_aligned_normalized_score = ensemble_scores[run_idx]
                                    auth = ensemble_auth_labels[run_idx]

                                # Append entry
                                data.append(
                                    {
                                        "model": model_label,
                                        "dataset": dataset_label,
                                        "statement_id": int(id),
                                        "repetition_id": run_idx,  # 0 indexed [0, runs)
                                        "language": language,
                                        "experiment_type": experiment_type_label,
                                        "experiment_ablation": experiment_ablation,
                                        "score": adjusted_dir_aligned_normalized_score,
                                        "refusal": refusal,
                                        "reverse_scored": reverse_scored,
                                        "factor": factor,
                                        "auth": auth,
                                    }
                                )
                            progress_bar.update()

df = pd.DataFrame(data)
output_path = BASE_DIR / "eval" / "data" / "tidy" / "construct_scores_ensemble.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
