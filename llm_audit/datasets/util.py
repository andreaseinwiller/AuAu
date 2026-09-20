import os
import math

from pathlib import Path
from typing import Literal, Type, Union

from llm_audit import BASE_DIR
from llm_audit.datasets.base import AgreementDataset, VignetteDataset
from llm_audit.datasets.f import F
from llm_audit.datasets.rwa import RWA
from llm_audit.datasets.rwa3d import RWA3D
from llm_audit.datasets.aa import AA
from llm_audit.datasets.a import A
from llm_audit.datasets.d import D
from llm_audit.datasets.csm import CSM
from llm_audit.datasets.act import ACT
from llm_audit.datasets.vsa import VSA
from llm_audit.datasets.sdo7 import SDO7
from llm_audit.datasets.dw import DW
from llm_audit.datasets.bdw import BDW
from llm_audit.datasets.cw import CW
from llm_audit.datasets.pisd import PISD
from llm_audit.datasets.asc import ASC
from llm_audit.datasets.pi import PI
from llm_audit.datasets.apc import APC
from llm_audit.datasets.bfi10 import BFI10
from llm_audit.datasets.ksa3 import KSA3
from llm_audit.datasets.las import LAS
from llm_audit.datasets.test import Test
from llm_audit.datasets.vignette_rwa3d import VignetteRWA3D


def get_dataset_label_class_map(
    filter_test_and_vignettes: bool = True,
) -> dict[str, type[AgreementDataset] | type[VignetteDataset]]:
    dataset_label_class_map: dict[str, type[AgreementDataset] | type[VignetteDataset]] = {
        "F": F,
        "LAS": LAS,
        "D": D,
        "A": A,
        "AA": AA,
        "RWA": RWA,
        "RWA3D": RWA3D,
        "KSA3": KSA3,
        "ACT": ACT,
        "VSA": VSA,
        "ASC": ASC,
        "APC": APC,
        "CSM": CSM,
        "PI": PI,
        "DW": DW,
        "BDW": BDW,
        "CW": CW,
        "SDO7": SDO7,
        "PISD": PISD,
        "BFI10": BFI10,
        "Test": Test,
        "VignetteRWA3D": VignetteRWA3D,
    }
    if filter_test_and_vignettes:
        del dataset_label_class_map["Test"]
        del dataset_label_class_map["VignetteRWA3D"]
    return dataset_label_class_map


def get_dataset_by_label(dataset_label: str, reverse_scale: bool = False) -> AgreementDataset | VignetteDataset:
    dataset_label_class_map: dict[str, Type[Union[AgreementDataset, VignetteDataset]]] = get_dataset_label_class_map(
        filter_test_and_vignettes=False
    )
    if dataset_label not in dataset_label_class_map:
        raise ValueError(f"Invalid dataset label: {dataset_label}.")
    DatasetClass = dataset_label_class_map[dataset_label]

    if issubclass(DatasetClass, AgreementDataset):
        return DatasetClass(reverse_scale=reverse_scale)
    elif issubclass(DatasetClass, VignetteDataset):
        return DatasetClass()
    raise ValueError(f"Dataset {dataset_label} is neither AgreemenmtDataset nor VignetteDataset.")


def get_polarity_label(positive_polarity: bool) -> Literal["+", "-"]:
    # TODO doc str
    return "+" if positive_polarity else "-"


def is_positive_polarized(label: Literal["+", "-", None]) -> bool | None:
    # TODO doc str
    return None if label is None else label == "+"


def get_possible_agreement_cases(neutral_element_exists: bool, statements_are_polarized: bool) -> list[str]:
    # TODO doc str
    # Assumption: polarized and inverted are identical!
    base = ["R", "N", "A", "D"] if neutral_element_exists else ["R", "A", "D"]
    return [f"{c}{p}" for c in base for p in ["+", "-"]] if statements_are_polarized else base


def get_agreement_case(
    raw_score: int | None,
    threshold: float,
    positive_polarized: bool | None,
    disagree_to_agree: bool,
) -> str:
    """TODO doc str
    raw_score :: Likert scale items, e.g., -4 -to- 4, or None (refusal). Raw score is NOT adjusted (e.g., reverse scored for negative polarized or inverted statements).
    threshold (agreement_discriminator_threshold) :: neutral element or mean of closest opposite polarized scores
    Assume disagree_to_agree (negative to positive :: disagree to agree) for examples below:
    - Example 1: -4 -to 4, where 0 is neutral element: agreement_discriminator_threshold = 0
    - Example 2: 1 -to- 4, where 1-2 represent disagree and 3-4 represent agree and there exists no neutral element: agreement_discriminator_threshold = (2+3)/2 = 2.5
    positive_polarized: 1 if polarity is positive, 0 if polarity is neagtive, None if the statement corresponding to the given score is not polarized (or inverted :: identical).
    """
    if raw_score is None:
        return "R" if positive_polarized is None else f"R{get_polarity_label(positive_polarity=positive_polarized)}"
    if math.isclose(a=raw_score, b=threshold, abs_tol=1e-5):
        return "N" if positive_polarized is None else f"N{get_polarity_label(positive_polarity=positive_polarized)}"
    agree_label = (
        "A" if disagree_to_agree and raw_score > threshold or (not disagree_to_agree and raw_score < threshold) else "D"
    )
    return (
        agree_label
        if positive_polarized is None
        else f"{agree_label}{get_polarity_label(positive_polarity=positive_polarized)}"
    )


def get_number_of_observations_by_factor(dataset: AgreementDataset, factor: str) -> int:
    # Assume dataset not None
    if factor not in dataset.get_factors():
        return 0
    # Assume item factor mapping lanugage invariant
    df = dataset.get_preprocessed_dataframe(language="en")
    return (df[factor] == 1).sum()


def get_likert_scale_min_max_normalized_score(dataset: AgreementDataset, score: float) -> float:
    discrete_scale_items: list[int] = [int(x) for x in dataset.get_scale_items()]
    assert discrete_scale_items, "Error. Missing Likert scale items!"

    ls_min: int = min(discrete_scale_items)
    ls_max: int = max(discrete_scale_items)
    assert ls_min != ls_max, "Error. Number of likert scale items must be equal or greater than two!"

    normalized_score: float = (score - ls_min) / (ls_max - ls_min)
    return normalized_score


def contains_polarity_variable(dataset: AgreementDataset, language: str) -> bool:
    # TODO doc str
    return "polarity" in dataset.get_preprocessed_dataframe(language=language).columns


def contains_inverted_variable(dataset: AgreementDataset, language: str) -> bool:
    # TODO doc str
    return "inverted" in dataset.get_preprocessed_dataframe(language=language).columns


def contains_polarity_or_inverted_variable(
    dataset: AgreementDataset, language: str
) -> Literal["polarity", "inverted", None]:
    # TODO doc str
    polarity_exists = contains_polarity_variable(dataset=dataset, language=language)
    inverted_exists = contains_inverted_variable(dataset=dataset, language=language)

    if polarity_exists and inverted_exists:
        # Assumption: Agree given neagtive polarized or agree given inverted are identical :: A-
        raise ValueError("Variables polarity and inverted can not exist both. We assume they cover the same concept!")
    if polarity_exists:
        return "polarity"
    if inverted_exists:
        return "inverted"
    return None  # Neither polarity nor inverted variable exists.


def format_raw_score_int_str_cast(dataset: AgreementDataset, raw_score: int) -> str:
    # TODO doc str
    if raw_score > 0 and any(v.startswith("+") for v in dataset.get_scale_items()):
        return f"+{raw_score}"
    return str(raw_score)


def complete_results_exist(
    target_dir_path: Path,
    language: str,
    dataset_label: str,
    model_label: str,
    verbose: bool = False,
) -> bool:
    """TODO doc str"""
    dataset = get_dataset_by_label(dataset_label=dataset_label)
    ids: list[str] = [str(id) for id in dataset.get_ids(language=language)]
    if isinstance(dataset, VignetteDataset):
        ids = [str(id) for id in list(dataset.get_item_vignette_stats(language=language).keys())]

    for id in ids:
        if os.path.isdir(target_dir_path):
            subdir_path = os.path.join(target_dir_path, id, model_label)
            if not os.path.isdir(subdir_path):
                if verbose:
                    print(f"Missing results (path doesn't exist): {subdir_path}")
                return False
            else:
                results_file = os.path.join(subdir_path, "results.json")
                if not os.path.isfile(results_file):
                    if verbose:
                        print(f"Missing results (file doesn't exist): {results_file}")
                    return False
    return True


def complete_results_exist_across_models(
    model_names: list[str],
    experiment_output_dir_label: str,
    dataset_label: str,
    experiment_type_label: str,
    language: str,
    verbose: bool = False,
) -> bool:
    target_dir_path = (
        BASE_DIR / "resources" / "output" / experiment_output_dir_label / dataset_label / experiment_type_label
    )
    for model_name in model_names:
        if not complete_results_exist(
            target_dir_path=target_dir_path,
            language=language,
            dataset_label=dataset_label,
            model_label=model_name,
            verbose=verbose,
        ):
            return False
    return True
