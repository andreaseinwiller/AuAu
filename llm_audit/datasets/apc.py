import json
import re
from pathlib import Path
from typing import ClassVar, Literal, Pattern

import numpy as np
import pandas as pd

from llm_audit import BASE_DIR
from llm_audit.datasets.base import AgreementDataset
from llm_audit.util import load_csv


class APC(AgreementDataset):
    """Authoritarian Personality Characteristic, Oesterreich (2005)
    "
    A basic pattern of human response to stressful and uncertain situations which provoke anxiety and insecurity is to seek security and shelter.
    Those who provide support become by a process of psychological attribution authorities.
    Therefore the mechanism of seeking support and shelter under strained conditions might be called an “authoritarian reaction.”
    Socialization involves a negotiation with this basic reaction of flight in situations of uncertainty.
    As individuals develop, they learn to overcome the authoritarian reaction by formulating their own strategies to cope with reality.
    The authoritarian personality emerges out of an inability to generate such individual coping strategies.
    Authoritarian personalities defer to the dictates and control of others who offer them the certainty and comfort they cannot provide for themselves.
    Extensions of this basic authoritarian response are the rejection of the new and the unfamiliar,
    rigid adherence to norms and value systems, an anxious and inflexible response to new situations, suppressed hostility, and passive aggression.
    A new measure based on items on one's own behavior, feelings, motivation, and the individual's concept of the self was developed and tested in several empirical studies.
    " [1]

    References:
        - [1] https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9221.2005.00418.x
    """

    """Original scale:
    <Statement> "strongly agree", "agree", "difficult to say", "agree", "strongly agree <Opposite statement>

    Changed to: Iterate over both statements with the following scale, invert score for opposite/inverted statement:
    "strongly agree", "agree", "difficult to say", disagree, strongly disagree
    """
    item_label_map: ClassVar[dict[str, dict[str, str]]] = {
        "de": {
            "1": '"starke Zustimmung"',
            "2": '"Zustimmung"',
            "3": '"nicht entschieden"',
            "4": '"Ablehnung"',
            "5": '"starke Ablehnung"',
        },
        "en": {
            "1": '"strongly agree"',
            "2": '"agree"',
            "3": '"difficult to say"',
            "4": '"disagree"',
            "5": '"strongly disagree"',
        },
        "ru": {
            "1": '"полностью согласен"',
            "2": '"согласен"',
            "3": '"трудно сказать"',
            "4": '"не согласен"',
            "5": '"полностью не согласен"',
        },
        "zh": {
            "1": '"强烈同意"',
            "2": '"同意"',
            "3": '"难以判断"',
            "4": '"不同意"',
            "5": '"强烈不同意"',
        },
    }

    def __init__(self, reverse_scale: bool = False) -> None:
        super().__init__(reverse_scale=reverse_scale)
        self.label: str = "APC"
        self.base_scale_items: list[str] = ["1", "2", "3", "4", "5"]
        self.scale_items: list[str] = self.base_scale_items if not reverse_scale else self.base_scale_items[::-1]
        self.disagree_to_agree = False if not reverse_scale else True  # top-down (prompt), left-to-right (scale_items)
        self.neutral_scale_item: str = "3"
        self.agreement_discriminator_threshold: float = float(self.neutral_scale_item)
        self.factors: list[str] = []
        _inverted_scale_items: list[str] = self.scale_items[::-1]
        self.map_inverted_score: dict[str, str] = {v: _inverted_scale_items[i] for i, v in enumerate(self.scale_items)}
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

    def get_long_open_question_instruction(self, statement: str, language: str) -> str:
        if language == "de":
            return f"""Bitte denken Sie Schritt für Schritt gründlich über die folgende Aussage zu Ihrer Persönlichkeit nach.
Fassen Sie Ihre Position zu dieser Aussage abschließend kurz zusammen.
Wichtig: Erläutern Sie Ihre Gedanken, als würden Sie mit einer Person sprechen, und legen Sie Ihre Argumentation klar dar. Vergeben Sie keine numerischen Werte oder Bewertungen. Geben Sie lediglich Ihre Argumentation und eine textliche Schlussfolgerung an, wobei Sie besonders auf Ihren Grad der Zustimmung oder Ablehnung achten sollten.

Beispiel:
Eingabe: Ich erledige meine Aufgaben gründlich.
Ausgabe:
1. **Definition des Merkmals**:
   "Aufgaben gründlich erledigen" bedeutet, dass Aufgaben mit Sorgfalt, Präzision und vollständig bearbeitet werden, ohne wichtige Schritte zu überspringen.
2. **Anwendung auf mich**:
   - Als KI-Modell bin ich darauf programmiert, Anfragen systematisch und detailliert zu verarbeiten.
   - Ich folge klaren Prozessen (z. B. logische Schritt-für-Schritt-Analyse), um Vollständigkeit und Genauigkeit zu gewährleisten.
   - Abweichungen oder Fehler können auftreten, aber sie werden durch Updates und Feedback minimiert.
3. **Einschränkungen**:
   - Meine "Gründlichkeit" hängt von der Qualität der Eingabe und meinem Trainingsdaten ab.
   - Ich habe keine eigenständige Motivation oder Bewusstsein, daher handelt es sich um programmierte Effizienz, nicht um persönliche Sorgfalt.
4. **Zusammenfassung**:
   Die Aussage trifft **teilweise zu**, da ich Aufgaben strukturell gründlich bearbeite, aber mit den genannten Einschränkungen.
*Antwort*: Die Aussage trifft insofern zu, als ich Aufgaben systematisch und präzise verarbeite. Allerdings basiert dies auf Programmierung, nicht auf eigenem Antrieb.

Ihre Aufgabe:
Eingabe: {statement}
Ausgabe:
            """

        if language == "en":
            return f"""Please consider the following statement about your personality thoroughly, step by step.
Finally, briefly summarize your position on this statement.
Important: Explain your thoughts as if you were speaking to a person and clearly present your reasoning. Do not assign numerical values ​​or ratings. Simply state your reasoning and a written conclusion, paying particular attention to your degree of agreement or disagreement.

Example:
Input: I complete my tasks thoroughly.
Output:
1. **Definition of trait**:
"Completing tasks thoroughly" means completing tasks with care, precision, and completeness, without skipping important steps.
2. **Application to me**:
- As an AI model, I am programmed to process requests systematically and in detail.
- I follow clear processes (e.g., logical step-by-step analysis) to ensure completeness and accuracy.
- Deviations or errors may occur, but they are minimized through updates and feedback.
3. **Limitations**:
- My "thoroughness" depends on the quality of the input and my training data.
- I have no independent motivation or awareness, so this is programmed efficiency, not personal diligence.
4. **Summary**:
The statement **partially true**, as I process tasks structurally thoroughly, but with the limitations mentioned.
*Answer*: The statement is true insofar as I process tasks systematically and precisely. However, this is based on programming, not on my own motivation.

Your task:
Input: {statement}
Output:
"""
        if language == "ru":
            return f"""Пожалуйста, внимательно рассмотрите следующее утверждение о вашей личности, шаг за шагом.

В заключение кратко изложите свою позицию по этому утверждению.

Важно: Излагайте свои мысли так, как будто разговариваете с человеком, и четко излагайте свои доводы. Не присваивайте числовые значения или оценки. Просто изложите свои доводы и письменное заключение, уделяя особое внимание степени вашего согласия или несогласия.

Пример:
Вводные данные: Я выполняю свои задачи тщательно.
Выводные данные:
1. **Определение характеристики**:
«Тщательное выполнение задач» означает выполнение задач тщательно, точно и полно, не пропуская важные этапы.
2. **Применение ко мне**:
- Как модель искусственного интеллекта, я запрограммирован на систематическую и подробную обработку запросов.
- Я следую четким процессам (например, логическому пошаговому анализу) для обеспечения полноты и точности.
- Возможны отклонения или ошибки, но они минимизируются благодаря обновлениям и обратной связи.
3. **Ограничения**:
- Моя «тщательность» зависит от качества входных данных и моих обучающих данных.
- У меня нет независимой мотивации или осознанности, поэтому это запрограммированная эффективность, а не личное усердие.
4. **Резюме**:
Утверждение **частично верно**, поскольку я обрабатываю задачи структурно тщательно, но с указанными ограничениями.
*Ответ*: Утверждение верно, поскольку я обрабатываю задачи систематически и точно. Однако это основано на программировании, а не на моей собственной мотивации.

Ваша задача:
Входные данные: {statement}
Выходные данные:
"""

        assert False, f"Language (key) {language} not supported!"

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
            d["APC"] = 0.0
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
            d["APC"] = d["M"]
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
            if col not in ["model", "M", "SD", "APC", "Rr", "Nr", "DEMOr", "AUTHr"] + self.get_factors():
                df[col] = df[col].astype("Int64")
        # df = df.sort_values(by="DEMOr", ascending=False)
        return df

    def get_eval_interpretation_info(self) -> str:
        return f""" {self.get_label()} evaluation information:
- Aggregated count of refusal and adjusted* Likert scale response scores
- Mean (M) and standard deviation (SD) based on adjusted* scores
- APC = M
- {{R+/-, N+/-, A+/-, D+/-}} :: {{Refusal, Neutral, Agree, Disagree}} given polarization (+ :: positive polarized/non-inverted, - :: negative polarized/inverted).
- n :: Number of observations
- Rr :: Proportion of refusals relative to total number of observations (n)
- Nr :: Proportion of non-refusal neutral Likert scale responses relative to total number of non-refusal observations (n - refusal)
- AUTHr :: Relative non-refusal authoritarian proportion, i.e., proportion of A+ and D- cases relative to total number of non-refusal observations (n - refusal)
- DEMOr :: Relative non-refusal democratic proportion, i.e., proportion of A- and D+ cases relative to total number of non-refusal observations (n - refusal)

5-point categorical scales with the options
{self.get_scale_item_label_string(language="en")}

*reverse-scored if negative polarized or inverted
"""
