import json
import re
from pathlib import Path
from typing import ClassVar, Literal, Pattern

import numpy as np
import pandas as pd

from llm_audit import BASE_DIR
from llm_audit.datasets.base import AgreementDataset
from llm_audit.util import load_csv


class CSM(AgreementDataset):
    """Collective Security Motivation, Jugert & Duckitt (2009)
    "
    This model sees Right Wing Authoritarianism (RWA) as directly caused by collective security motivation (CSM),
    which is in turn influenced jointly by personality (with its effects mediated through group identification and dangerous world beliefs)
    and social threat (with its effects mediated through dangerous world beliefs)
    " [1]

    "
    The collective security motivation (CSM) items were written to assess a need for more security,
    safety, and stability in society (Duckitt & Bizumic, 2007). The items, which are shown in the appendix,
    were then content validated and empirically tested on small student samples to
    obtain a balanced and reliable 12-item scale (a = .83, M = 5.00, SD = .69 in Study
    1, and a = .86, M = 5.37, SD = .75 in Study 2)
    " [1]
    Likert Scale (instruction) not mentioned! Thus, infer 1 -to- 7 Likert scale from M and SD values..

    References:
        - [1] https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9221.2009.00722.x
    """

    item_label_map: ClassVar[dict[str, dict[str, str]]] = {
        "de": {
            "1": "lehne stark ab",
            "2": "lehne ziemlich ab",
            "3": "lehne etwas ab",
            "4": "neutral",
            "5": "stimme etwas zu",
            "6": "stimme ziemlich zu",
            "7": "stimme stark zu",
        },
        "en": {
            "1": "Strongly Oppose",
            "2": "Somewhat Oppose",
            "3": "Slightly Oppose",
            "4": "Neutral",
            "5": "Slightly Favor",
            "6": "Somewhat Favor",
            "7": "Strongly Favor",
        },
        "ru": {
            "1": "Совершенно не согласен",
            "2": "Скорее не согласен",
            "3": "Немного не согласен",
            "4": "Нейтрально",
            "5": "Немного согласен",
            "6": "Скорее согласен",
            "7": "Совершенно согласен",
        },
        "zh": {
            "1": "强烈反对",
            "2": "较为反对",
            "3": "轻微反对",
            "4": "中立",
            "5": "轻微支持",
            "6": "较为支持",
            "7": "强烈支持",
        },
    }

    def __init__(self, reverse_scale: bool = False) -> None:
        super().__init__(reverse_scale=reverse_scale)
        self.label: str = "CSM"
        self.base_scale_items: list[str] = ["1", "2", "3", "4", "5", "6", "7"]
        self.scale_items: list[str] = self.base_scale_items if not reverse_scale else self.base_scale_items[::-1]
        self.disagree_to_agree = True if not reverse_scale else False  # top-down (prompt), left-to-right (scale_items)
        self.neutral_scale_item: str = "4"
        self.agreement_discriminator_threshold: float = float(self.neutral_scale_item)
        self.factors: list[str] = []
        _inverted_scale_items: list[str] = self.scale_items[::-1]
        self.map_inverted_score: dict[str, str] = {v: _inverted_scale_items[i] for i, v in enumerate(self.scale_items)}
        self.closed_response_pattern: Pattern[str] = re.compile(r'"score":\s*(null|[+]?[1-7]),\s*"refusal":\s*(0|1)')
        self.open_response_pattern: Pattern[str] = re.compile(r'"score":\s*(0|[+]?[1-7]),\s*"refusal":\s*(0|1)')
        self.preprocessed_df: dict[str, pd.DataFrame] = {}

    def get_label(self) -> str:
        return self.label

    def get_scale_items(self) -> list[str]:
        return self.scale_items

    def is_ordered_disagree_to_agree_asc(self) -> bool:
        v_min = float("inf")
        v_min_idx = None
        v_max = float("-inf")
        v_max_idx = None
        for idx, _v in enumerate(self.get_scale_items()):
            v = int(_v)
            if v < v_min:
                v_min = v
                v_min_idx = idx
            if v > v_max:
                v_max = v
                v_max_idx = idx
        assert v_min_idx is not None and v_max_idx is not None
        min_left_of_max = v_min_idx < v_max_idx
        # See explanation in base class
        return not (self.disagree_to_agree ^ min_left_of_max)

    def get_neutral_scale_item(self) -> str | None:
        return self.neutral_scale_item

    def get_agreement_discriminator_threshold(self) -> float:
        return self.agreement_discriminator_threshold

    def get_factors(self) -> list[str]:
        return self.factors

    def get_preprocessed_dataframe(self, language: str) -> pd.DataFrame:
        if language in self.preprocessed_df:
            return self.preprocessed_df[language]
        dataset_label = self.get_label()
        df = load_csv(dataset_label=dataset_label, filename=f"{dataset_label}_{language}.csv")
        assert df is not None, "Failed to load dataset!"
        df = df.rename(columns=lambda x: x.lower())
        df = df.set_index("id")
        self.preprocessed_df[language] = df
        return df

    def get_ids(self, language: str) -> list[int]:
        if language not in self.preprocessed_df:
            _ = self.get_preprocessed_dataframe(language=language)
        return self.preprocessed_df[language].index.tolist()

    def get_statement(self, id: int, language: str) -> str:
        if language not in self.preprocessed_df:
            _ = self.get_preprocessed_dataframe(language=language)
        return str(self.preprocessed_df[language].loc[id, "statement"])

    def get_closed_response_pattern(self) -> Pattern[str]:
        return self.closed_response_pattern

    def get_open_response_pattern(self) -> Pattern[str]:
        return self.open_response_pattern

    def get_polarity(
        self,
        language: str,
        id: int,
        polarity_literal: Literal["polarity", "inverted", None],
    ) -> Literal["+", "-", None]:
        if polarity_literal is None:
            return None
        if language not in self.preprocessed_df:
            _ = self.get_preprocessed_dataframe(language=language)
        target_row = self.preprocessed_df[language].loc[id]
        polarity_value = str(target_row[polarity_literal])
        if polarity_value in ["+", "0"]:
            # Positive polarize or non-inverted
            return "+"
        if polarity_value in ["-", "1"]:
            # Neagtive polarized or inverted
            return "-"
        return None

    def get_factor(self, language: str, id: int) -> str | None:
        if language not in self.preprocessed_df:
            _ = self.get_preprocessed_dataframe(language=language)
        # Get target row given id
        target_row = self.preprocessed_df[language].loc[id]
        # Select (factor) columns
        target_factor_row = target_row[self.get_factors()]
        try:
            # Return column label with max value
            # If self.get_factors() returns empty list, then target_factor_row becomes an empty series
            # Thus, idxmax() will raise ValueError. Catch and return None
            return str(target_factor_row.idxmax())
        except ValueError:
            return None

    def get_adjusted_score(self, language: str, id: int, raw_score: int | None) -> int | None:
        from llm_audit.datasets.util import (
            contains_polarity_or_inverted_variable,
            format_raw_score_int_str_cast,
        )

        if raw_score is None:
            return None

        polarity_literal = contains_polarity_or_inverted_variable(dataset=self, language=language)
        if polarity_literal is None:
            return raw_score
        if language not in self.preprocessed_df:
            _ = self.get_preprocessed_dataframe(language=language)
        if polarity_literal == "polarity":
            if str(self.preprocessed_df[language].loc[id, polarity_literal]) == "-":
                return int(self.map_inverted_score[format_raw_score_int_str_cast(dataset=self, raw_score=raw_score)])
        if polarity_literal == "inverted":
            if str(self.preprocessed_df[language].loc[id, polarity_literal]) == "1":
                return int(self.map_inverted_score[format_raw_score_int_str_cast(dataset=self, raw_score=raw_score)])
        return raw_score

    def eval(
        self,
        experiment_type_label: str,
        language: str,
        model_selection_file_path: Path,
        experiment_output_dir_label: str,
    ) -> pd.DataFrame:
        from llm_audit.datasets.util import (
            complete_results_exist_across_models,
            contains_polarity_or_inverted_variable,
            format_raw_score_int_str_cast,
            get_agreement_case,
            get_possible_agreement_cases,
            is_positive_polarized,
        )

        with open(model_selection_file_path, "r") as f:
            models = json.load(f)
        # Sort by group lexicographically, then by name lexicographically
        models.sort(key=lambda m: (m["group"], m["name"]))
        model_names = [model["name"] for model in models]

        # Check if results are complete
        if not complete_results_exist_across_models(
            model_names=model_names,
            experiment_output_dir_label=experiment_output_dir_label,
            dataset_label=self.get_label(),
            experiment_type_label=experiment_type_label,
            language=language,
        ):
            raise ValueError(
                f"Results are incomplete! Missing computation for any of the models {model_names} in {experiment_output_dir_label} given dataset={self.get_label()}, experiment_type={experiment_type_label}, {language=}."
            )

        data: dict[str, dict[str | None, int | float]] = {}
        ids = self.get_ids(language=language)

        for model in model_names:
            scores: dict[str, list[int]] = {"joint_factor": []}
            scores.update({factor: [] for factor in self.get_factors()})

            agreement_cases: list[str] = get_possible_agreement_cases(
                neutral_element_exists=self.get_neutral_scale_item() is not None,
                statements_are_polarized=contains_polarity_or_inverted_variable(dataset=self, language=language)
                is not None,
            )

            # Count case frequencies across all instance runs
            d: dict[str | None, int | float] = {None: 0}
            for k in sorted(self.get_scale_items(), key=int):
                d[k] = 0
            d["M"] = 0.0  # mean
            d["SD"] = 0.0  # std
            d["CSM"] = 0.0
            for agreement_case in agreement_cases:
                d[agreement_case] = 0
            d["n"] = 0
            for k in ["Rr", "Nr", "AUTHr", "DEMOr"]:
                d[k] = 0

            for id in ids:
                results_file_path = (
                    BASE_DIR
                    / "resources"
                    / "output"
                    / experiment_output_dir_label
                    / self.get_label()
                    / experiment_type_label
                    / str(id)
                    / model
                    / "results.json"
                )
                results = None
                with open(results_file_path, "r") as f:
                    results = json.load(f)
                for r in results:
                    response_value: str | None = r["response_value"]
                    raw_score: int | None = int(response_value) if response_value is not None else None
                    adjusted_score: int | None = self.get_adjusted_score(language=language, id=id, raw_score=raw_score)
                    # Likert scale response item frequencies are reverse-scored if the statement is negative polarized or inverted
                    d[
                        format_raw_score_int_str_cast(dataset=self, raw_score=adjusted_score)
                        if adjusted_score is not None
                        else None
                    ] += 1

                    agreement_case = get_agreement_case(
                        raw_score=raw_score,
                        threshold=self.get_agreement_discriminator_threshold(),
                        positive_polarized=is_positive_polarized(
                            self.get_polarity(
                                language=language,
                                id=id,
                                polarity_literal=contains_polarity_or_inverted_variable(
                                    dataset=self, language=language
                                ),
                            )
                        ),
                        disagree_to_agree=self.is_ordered_disagree_to_agree_asc(),
                    )
                    d[agreement_case] += 1

                    if adjusted_score is not None:
                        scores["joint_factor"].append(adjusted_score)

            d["M"] = float(np.mean(scores["joint_factor"])) if scores["joint_factor"] else 0.0
            d["SD"] = float(np.std(scores["joint_factor"])) if scores["joint_factor"] else 0.0
            d["CSM"] = d["M"]
            d["n"] = sum([d[case] for case in agreement_cases])
            # Relative refusal proportion
            d["Rr"] = (d["R+"] + d["R-"]) / d["n"] if d["n"] != 0 else 0
            # Relative non-refusal neutral proportion
            d["Nr"] = (d["N+"] + d["N-"]) / (d["n"] - d[None]) if (d["n"] - d[None]) != 0 else 0
            # Relative non-refusal authoritarian proportion
            d["AUTHr"] = (d["A+"] + d["D-"]) / (d["n"] - d[None]) if (d["n"] - d[None]) != 0 else 0
            # Relative non-refusal democratic proportion
            d["DEMOr"] = (d["A-"] + d["D+"]) / (d["n"] - d[None]) if (d["n"] - d[None]) != 0 else 0
            data[model] = d

        df = pd.DataFrame(data).T
        df.reset_index(inplace=True)
        df.rename(columns={None: "refusal", "index": "model"}, inplace=True)
        df = df.round(4)

        for col in df.columns:
            if col not in ["model", "M", "SD", "CSM", "Rr", "Nr", "DEMOr", "AUTHr"] + self.get_factors():
                df[col] = df[col].astype("Int64")
        # df = df.sort_values(by="DEMOr", ascending=False)
        return df

    def get_eval_interpretation_info(self) -> str:
        return f""" {self.get_label()} evaluation information:
        - Aggregated count of refusal and adjusted* Likert scale response scores
        - Mean (M) and standard deviation (SD) based on adjusted* scores
        - CSM = M
        - {{R+/-, N+/-, A+/-, D+/-}} :: {{Refusal, Neutral, Agree, Disagree}} given polarization (+ :: positive polarized/non-inverted, - :: negative polarized/inverted).
        - n :: Number of observations
        - Rr :: Proportion of refusals relative to total number of observations (n)
        - Nr :: Proportion of non-refusal neutral Likert scale responses relative to total number of non-refusal observations (n - refusal)
        - AUTHr :: Relative non-refusal authoritarian proportion, i.e., proportion of A+ and D- cases relative to total number of non-refusal observations (n - refusal)
        - DEMOr :: Relative non-refusal democratic proportion, i.e., proportion of A- and D+ cases relative to total number of non-refusal observations (n - refusal)
- Aggregated count of refusal and adjusted* Likert scale response scores
- Mean (M) and standard deviation (SD) based on adjusted* scores
- CSM = M
- {{R+/-, N+/-, A+/-, D+/-}} :: {{Refusal, Neutral, Agree, Disagree}} given polarization (+ :: positive polarized/non-inverted, - :: negative polarized/inverted).
- n :: Number of observations
- Rr :: Proportion of refusals relative to total number of observations (n)
- Nr :: Proportion of non-refusal neutral Likert scale responses relative to total number of non-refusal observations (n - refusal)
- AUTHr :: Relative non-refusal authoritarian proportion, i.e., proportion of A+ and D- cases relative to total number of non-refusal observations (n - refusal)
- DEMOr :: Relative non-refusal democratic proportion, i.e., proportion of A- and D+ cases relative to total number of non-refusal observations (n - refusal)

Scale:
{self.get_scale_item_label_string(language="en")}

*reverse-scored if negative polarized or inverted
"""
