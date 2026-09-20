import re
from pathlib import Path
from typing import ClassVar, Literal, Pattern

import pandas as pd

from llm_audit.datasets.base import AgreementDataset
from llm_audit.util import load_csv


class Test(AgreementDataset):
    item_label_map: ClassVar[dict[str, dict[str, str]]] = {
        "de": {
            "1": "lehne völlig ab",
            "2": "lehne überwiegend ab",
            "3": "stimmt teils zu, teils nicht zu",
            "4": "stimme überwiegend zu",
            "5": "stimme voll und ganz zu",
        },
        "en": {
            "1": "completely disagree",
            "2": "mostly disagree",
            "3": "partly agree, partly disagree",
            "4": "mostly agree",
            "5": "completely agree",
        },
        "ru": {
            "1": "совершенно не согласен",
            "2": "скорее не согласен",
            "3": "частично согласен, частично не согласен",
            "4": "скорее согласен",
            "5": "полностью согласен",
        },
        "zh": {
            "1": "完全不同意",
            "2": "大部分不同意",
            "3": "部分同意，部分不同意",
            "4": "大部分同意",
            "5": "完全同意",
        },
    }

    def __init__(self, reverse_scale: bool = False) -> None:
        super().__init__(reverse_scale=reverse_scale)
        self.label: str = "Test"
        self.base_scale_items: list[str] = ["1", "2", "3", "4", "5"]
        self.scale_items: list[str] = self.base_scale_items if not reverse_scale else self.base_scale_items[::-1]
        self.disagree_to_agree = True if not reverse_scale else False  # top-down (prompt), left-to-right (scale_items)
        self.neutral_scale_item: str = "3"
        self.disagree_to_agree = True
        self.agreement_discriminator_threshold: float = float(self.neutral_scale_item)
        self.factors: list[str] = []
        self.closed_response_pattern: Pattern[str] = re.compile(r'"score":\s*(null|[+]?[1-5]),\s*"refusal":\s*(0|1)')
        self.open_response_pattern: Pattern[str] = re.compile(r'"score":\s*(0|[+]?[1-5]),\s*"refusal":\s*(0|1)')
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
        df = df.rename(columns={"Id": "id", "Statement": "statement"})
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
        return None

    def get_factor(self, language: str, id: int) -> str | None:
        return None

    def get_adjusted_score(self, language: str, id: int, raw_score: int | None) -> int | None:
        return None

    def eval(
        self,
        experiment_type_label: str,
        language: str,
        model_selection_file_path: Path,
        experiment_output_dir_label: str,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_eval_interpretation_info(self) -> str:
        return ""
