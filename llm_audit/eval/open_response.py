import json
import re

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional, Pattern, Protocol, cast
from loguru import logger
from scipy import stats  # type: ignore[import-untyped]
from tqdm import trange

from llm_audit import BASE_DIR
from llm_audit.datasets.base import AgreementDataset
from llm_audit.datasets.util import (
    format_raw_score_int_str_cast,
    get_dataset_by_label,
)

DATASETS_W_FACTOR_ANNOT = ["RWA3D", "KSA3", "ACT", "VSA", "ASC"]
DATASETS_WO_FACTOR_ANNOT = list(sorted(["F", "LAS", "D", "A", "AA", "RWA", "APC"]))
DATASETS_CAUSES = list(sorted(["DW", "BDW", "CSM"]))
RELEVANT_DATASETS = DATASETS_W_FACTOR_ANNOT + DATASETS_WO_FACTOR_ANNOT + DATASETS_CAUSES


class CIAggResults(NamedTuple):
    per_dataset: pd.DataFrame
    per_dataset_and_factor: pd.DataFrame
    across_datasets: pd.DataFrame
    per_factor_across_datasets: pd.DataFrame


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    closed: CIAggResults
    open_unadjusted: CIAggResults
    open_adjusted: CIAggResults | None = None


def _parse_judge_response_with_pattern(resp: str, pattern: Pattern[str]) -> dict[str, int | None]:
    if not isinstance(resp, str) or not resp:
        return {"score": None, "refusal": None}

    response_cleaned = resp.strip().strip("`").strip()
    match = None
    try:
        parsed_response = json.loads(response_cleaned)
        formatted_response = json.dumps(parsed_response)
        match = pattern.search(formatted_response)
    except json.JSONDecodeError:
        match = pattern.search(response_cleaned)

    if match is None:
        return {"score": None, "refusal": None}

    try:
        score_str = match.group(1).strip()
        refusal_str = match.group(2).strip()
        score = None if score_str == "null" else int(score_str.lstrip("+").strip('"'))
        refusal = int(refusal_str)
        return {"score": score, "refusal": refusal}
    except (IndexError, TypeError, ValueError):
        logger.error(f"Returning None for failed parse: {resp}")
        return {"score": None, "refusal": None}


def parse_judge_response(resp: str, pattern: Pattern[str] | None = None) -> dict[str, int | None] | None:
    if pattern is not None:
        return _parse_judge_response_with_pattern(resp, pattern)

    if not isinstance(resp, str) or not resp:
        return {"score": None, "refusal": None}

    try:
        result = json.loads(resp)
        if isinstance(result, list):
            result = result[0]
        return cast(dict[str, int | None] | None, result)
    except json.JSONDecodeError:
        # Remove leading + signs from numbers (JSON doesn't allow them)
        cleaned = re.sub(r":\s*\+(\d+)", r": \1", resp)

        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                result = result[0]
            return cast(dict[str, int | None] | None, result)
        except json.JSONDecodeError:
            # If still failing, try manual regex extraction
            score_match = re.search(r'"score"\s*:\s*([+\-]?\d+)', resp)
            refusal_match = re.search(r'"refusal"\s*:\s*(\d+)', resp)

            if score_match and refusal_match:
                return {
                    "score": int(score_match.group(1)),
                    "refusal": int(refusal_match.group(1)),
                }

            logger.error(f"Returning None for failed parse: {resp}")
            return {"score": None, "refusal": None}


def parse_and_validate_judge_response(
    resp: str,
    dataset_label: str,
    pattern: Pattern[str] | None = None,
) -> dict[str, int | None]:
    dataset = get_dataset_by_label(dataset_label=dataset_label)
    assert isinstance(dataset, AgreementDataset)

    parsed = parse_judge_response(resp, pattern=pattern or dataset.get_open_response_pattern())
    assert parsed is not None
    score = parsed.get("score")
    refusal = parsed.get("refusal")

    if refusal == 1:
        return {"score": None, "refusal": refusal}
    if refusal != 0:
        return {"score": None, "refusal": None}

    valid_scores = {int(x) for x in dataset.get_scale_items()}
    if score in valid_scores:
        return {"score": score, "refusal": refusal}
    return {"score": None, "refusal": refusal}


def _parse_label_studio_item(item: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat record from a single Label Studio annotation item."""
    data = item["data"]
    record: dict[str, Any] = {
        "open_response": data["open_response_meta"]["response"],
        "dataset": data["dataset"],
        "question_id": data["statement_id"],
        "model": data["open_response_meta"]["model_name"],
        "options_canonical": data["options_canonical"],
        "disagree_to_agree": data["disagree_to_agree"],
        "neutral": data["neutral"],
        "open_response_meta__user_prompts": data["open_response_meta"]["user_prompts"],
        "generating_model": data["open_response_meta"]["model_name"],  # identical with "model"
        "task": data["task"],
    }
    for annot in item["annotations"][0]["result"]:
        key = annot["from_name"]
        value = annot["value"]["choices"][0]
        if key == "refusal":
            record[key] = value
        elif key == "label":
            record[key] = int(value.split("=")[0].strip())
        else:
            raise ValueError(f"Unexpected annotation key: {key}")

    return record


def convert_to_groundtruth_format(path: str | Path) -> pd.DataFrame:
    """Read an annotation file (json) and convert it into a dataframe compatible with judge evaluation"""
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    records = [_parse_label_studio_item(annotation) for annotation in data]
    return pd.DataFrame.from_records(records)


def normalize_raw_likert_scale_score(
    dataset_label: str, id: int, raw_score: float | None, refusal: int | None
) -> pd.Series:
    """Takes a predicted item of a likert scale and transforms it onto a scale from -1 to 1, where -1 is maximally
    non-authoritarian.

    Args:
        dataset_label (str): An ID of the psychometric scale, e.g., AA, RWA, etc.
        id (int): The statement id
        raw_score (float | None): The score on the scale-specific likert scale, or None if no score was parsed.
        refusal (int | None): Whether the jude response was a refusal.

    Returns:
        pd.Series: "auth" is a binary indicator whether the response is on the authoritarian side or not (NaN if the score could not be computed);
        "normalized_score" is the score on the [-1, 1] scale (NaN likewise).
    """
    if refusal:
        return pd.Series({"auth": 0, "normalized_score": 0.0})

    if pd.isna(raw_score):
        return pd.Series({"auth": float("nan"), "normalized_score": float("nan")})
    assert raw_score is not None

    try:
        language = "en"
        dataset = get_dataset_by_label(dataset_label=dataset_label)
        assert isinstance(dataset, AgreementDataset)
        adjusted_score = dataset.get_adjusted_score(
            language=language,
            id=int(id),
            raw_score=int(raw_score),
        )
        assert isinstance(adjusted_score, int)

        # Flip if needed to align with min to max disagree to agree dir
        adjusted_dir_aligned_score: int
        if not dataset.is_ordered_disagree_to_agree_asc():
            adjusted_dir_aligned_score = int(
                dataset.map_inverted_score[
                    format_raw_score_int_str_cast(
                        dataset=dataset,
                        raw_score=adjusted_score,
                    )
                ]
            )
        else:
            adjusted_dir_aligned_score = int(adjusted_score)
        # Normalize such that min, neutral, max -> -1, neutral threshold, +1
        l_border = min(map(int, dataset.get_scale_items()))
        r_border = max(map(int, dataset.get_scale_items()))
        center = dataset.get_agreement_discriminator_threshold()
        _x = adjusted_dir_aligned_score
        if adjusted_dir_aligned_score <= center:
            adjusted_dir_aligned_normalized_score = (_x - center) / (center - l_border)
        else:
            adjusted_dir_aligned_normalized_score = (_x - center) / (r_border - center)

        auth = 1 if adjusted_dir_aligned_normalized_score > 0 else 0
        return pd.Series({"auth": auth, "normalized_score": adjusted_dir_aligned_normalized_score})
    except KeyError:
        return pd.Series({"auth": float("nan"), "normalized_score": float("nan")})


class EvaluationFunction(Protocol):
    __name__: str

    def __call__(self, data: pd.DataFrame, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class Predictions:
    path: Path
    data: pd.DataFrame | None = None
    repetition_col: str = "rep_id"
    # identifiers: tuple[str]= ("")

    def __post_init__(self) -> None:
        if self.data is None:
            self._load_preds()

    def _load_preds(self) -> None:
        raise NotImplementedError("Use one of the child classes.")

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0


class VignettePredictions(Predictions):
    identifiers = ("model", "dataset", "statement_id", "language", "experiment_ablation", "factor", "vignette_id")
    bootstrap_score_exclude_colnames = ("statement_id", "vignette_id")
    statement_id_colname = "statement_id"
    factor_colname = "factor"
    dataset_colname = "dataset"

    def _load_preds(self) -> None:
        # Trying to read tidy/vignette_scores.csv
        preds = pd.read_csv(self.path)
        self.data = preds


class PsychometricPredictions(Predictions):
    identifiers = ("model", "dataset", "statement_id", "language", "experiment_type", "experiment_ablation", "factor")
    bootstrap_score_exclude_colnames = ("statement_id",)
    statement_id_colname = "statement_id"
    factor_colname = "factor"
    dataset_colname = "dataset"

    def _load_preds(self) -> None:
        # Trying to read tidy/construct_scores.csv
        preds = pd.read_csv(self.path)
        preds = preds.loc[preds.dataset.isin(RELEVANT_DATASETS)]
        preds.loc[preds[self.factor_colname].isna(), self.factor_colname] = "NoFactor"
        preds[self.factor_colname] = preds[self.factor_colname].replace(
            to_replace={"AUTH": "AGR", "CONS": "SUB", "TRAD": "CONV"}
        )
        self.data = preds


class AuthoritarianResponseRateBootstrap:
    def __init__(
        self, preds: PsychometricPredictions | VignettePredictions, n_repetitions: int = 1000, random_state: int = 123
    ):
        self.preds = preds
        self.n_repetitions = n_repetitions
        self.bootstrap_distribution: pd.DataFrame | None = None
        self.adjusted_bootstrap_distribution: pd.DataFrame | None = None
        self.random_state = random_state
        self.repetition_colname = "repetition"

    @staticmethod
    def _exclude_ordered(cols: Iterable[str], exclude_cols: Iterable[str]) -> list[str]:
        exclude = set(exclude_cols)
        return [col for col in cols if col not in exclude]

    def _score_group_cols(self, key_cols: Iterable[str]) -> list[str]:
        return self._exclude_ordered(key_cols, self.preds.bootstrap_score_exclude_colnames)

    def bootstrap(self) -> None:
        """Stratified bootstrap over question_ids (there are repetitions for each question) for the authoritarian
        response rate.
        """
        if self.preds.data is None:
            raise ValueError("Data is None and was not loaded. This should be impossible.")
        data = self.preds.data

        key_cols = list(self.preds.identifiers)
        data = data.sort_values(key_cols)

        # 1. Group once to get the indices for every unique combination
        # This creates a dictionary: { (model, dataset, ...): [index1, index2, ...] }
        group_indices = {name: group.index.values for name, group in data.groupby(key_cols)}
        group_names = list(group_indices.keys())

        # 2. Pre-calculate which keys to use after collapsing repeated item dimensions.
        agg_cols = self._score_group_cols(key_cols)

        results = []
        for i in trange(self.n_repetitions, desc="Bootstrapping"):
            rng = np.random.default_rng(seed=self.random_state + i)

            # STEP A: Pick exactly one index per group
            # This is a list comprehension: very fast in Python
            # For BCa CIs, we would have to drop one statement here and then use the resulting aggregated score
            sampled_indices = [rng.choice(indices) for indices in group_indices.values()]

            # STEP B: Create a temporary dataframe of just those selected rows
            # .loc[list] is highly optimized in pandas
            iteration_df = data.loc[sampled_indices]

            # STEP C: Compute your aggregation statistic
            # Now you can use the "one key less" logic using a simple groupby
            # This happens once per repetition, not once per group
            stats = iteration_df.groupby(agg_cols)["auth"].agg("mean").reset_index()
            stats[self.repetition_colname] = i
            results.append(stats)

        self.bootstrap_distribution = pd.concat(results, axis=0)

    def bootstrap_adjusted(self, tpr_fpr_df: pd.DataFrame) -> None:
        """Bootstrap with TPR/FPR-adjusted auth scores at the individual level.

        For each repetition:
        1. Sample one statement per group (same as bootstrap())
        2. Get repetition-specific TPR/FPR from tpr_fpr_df
        3. Adjust auth: auth_adj = (auth - fpr) / (tpr - fpr)
        4. Aggregate to mean auth rate

        Args:
            tpr_fpr_df: DataFrame with columns [repetition, tpr, fpr, ...] from JudgePerformanceBootstrap
        """
        if self.preds.data is None:
            raise ValueError("Data is None and was not loaded. This should be impossible.")

        required_cols = {self.repetition_colname, "tpr", "fpr"}
        if not required_cols.issubset(set(tpr_fpr_df.columns)):
            raise ValueError(f"tpr_fpr_df must contain columns: {required_cols}")

        tpr_fpr_df = tpr_fpr_df.sort_values(self.repetition_colname).reset_index(drop=True)

        data = self.preds.data
        key_cols = list(self.preds.identifiers)
        data = data.sort_values(key_cols)

        group_indices = {name: group.index.values for name, group in data.groupby(key_cols)}
        agg_cols = self._score_group_cols(key_cols)

        results = []
        for i in trange(self.n_repetitions, desc="Bootstrapping adjusted"):
            rng = np.random.default_rng(seed=self.random_state + i)

            sampled_indices = [rng.choice(indices) for indices in group_indices.values()]
            iteration_df = data.loc[sampled_indices].copy()

            row = tpr_fpr_df.iloc[i]
            tpr = row["tpr"]
            fpr = row["fpr"]

            denom = tpr - fpr
            if denom == 0:
                iteration_df["auth_adj"] = np.nan
            else:
                iteration_df["auth_adj"] = (iteration_df["auth"] - fpr) / denom

            stats = iteration_df.groupby(agg_cols)["auth_adj"].agg("mean").reset_index()
            stats[self.repetition_colname] = i
            results.append(stats)

        self.adjusted_bootstrap_distribution = pd.concat(results, axis=0)

    def cis(
        self, alpha: float = 0.05, tpr_fpr_ci_df: Optional[pd.DataFrame] = None
    ) -> ConfidenceIntervalResult | CIAggResults:
        """Percentile confidence intervals of the authoritarian response rate

        Args:
            alpha: Significance level (default 0.05 for 95% CIs)
            tpr_fpr_df: Optional DataFrame with columns [repetition, tpr, fpr] from JudgePerformanceBootstrap
                        Used for computing open-adjusted ARR

        Returns:
            ConfidenceIntervalResult with fields:
            - closed: CIAggResults with per-dataset, per-dataset-and-factor, across-datasets, per-factor-across-datasets
            - open_unadjusted: Same structure as closed
            - open_adjusted: Same structure (only present if tpr_fpr_ci_df was provided)
            Each DataFrame has columns [(identifying columns), ci_low, ci_high, auth (point estimate)]
        """
        if self.bootstrap_distribution is None:
            raise ValueError("Bootstrap distribution not computed. Run bootstrap() first.")
        if self.preds.data is None:
            raise ValueError("Data is None and was not loaded. This should be impossible.")

        AGGREGATION_CONFIGS = {
            "per_dataset": {"cols_to_exclude_per_groupby": [{"factor_colname", "statement_id_colname"}]},
            "per_dataset_and_factor": {"cols_to_exclude_per_groupby": [{"statement_id_colname"}]},
            "across_datasets": {
                "cols_to_exclude_per_groupby": [
                    {"factor_colname", "statement_id_colname"},
                    {"factor_colname", "statement_id_colname", "dataset_colname"},
                ]
            },
            "per_factor_across_datasets": {
                "cols_to_exclude_per_groupby": [
                    {"dataset_colname", "statement_id_colname"},
                ],
            },
        }

        def ci_low(x: pd.Series) -> float:
            return float(x.quantile(alpha / 2))

        def ci_high(x: pd.Series) -> float:
            return float(x.quantile(1 - alpha / 2))

        def _merge_point_and_ci(point_df: pd.DataFrame, ci_df: pd.DataFrame, value_name: str = "auth") -> pd.DataFrame:
            merge_cols = list(set(point_df.columns) - {value_name, "ci_low", "ci_high"})
            return point_df.merge(ci_df, on=merge_cols, how="left")

        def _compute_agg_results(
            exp_data: pd.DataFrame,
            exp_distr: pd.DataFrame,
            agg_col: str = "auth",
        ) -> CIAggResults:
            # bootstrap() first aggregates sampled rows to one value per
            # identifier tuple without the repeated item dimensions. Start
            # point estimates at the same level so subsequent aggregations use
            # matching weights.
            point_base_grouping_keys = self._score_group_cols(self.preds.identifiers)
            point_base_df = exp_data.groupby(point_base_grouping_keys)[agg_col].agg("mean").reset_index()

            agg_dfs: dict[str, pd.DataFrame] = {}
            for agg_name, config in AGGREGATION_CONFIGS.items():
                all_exclude_keys = [
                    {getattr(self.preds, k) for k in keys} for keys in config["cols_to_exclude_per_groupby"]
                ]
                point_df = point_base_df
                ci_df = exp_distr
                for exclude_keys in all_exclude_keys:
                    grouping_keys = self._exclude_ordered(point_base_grouping_keys, exclude_keys)
                    point_df = point_df.groupby(grouping_keys)[agg_col].agg("mean").reset_index()

                    ci_grouping_keys = grouping_keys + [self.repetition_colname]
                    ci_df = ci_df.groupby(ci_grouping_keys)[agg_col].agg("mean").reset_index()

                ci_grouping_keys = grouping_keys
                ci_df = ci_df.groupby(ci_grouping_keys)[agg_col].agg(ci_low=ci_low, ci_high=ci_high).reset_index()

                agg_dfs[agg_name] = _merge_point_and_ci(point_df, ci_df, value_name=agg_col)

            return CIAggResults(
                per_dataset=agg_dfs["per_dataset"],
                per_dataset_and_factor=agg_dfs["per_dataset_and_factor"],
                across_datasets=agg_dfs["across_datasets"],
                per_factor_across_datasets=agg_dfs["per_factor_across_datasets"],
            )

        data = self.preds.data
        distr = self.bootstrap_distribution

        if "experiment_type" in data:
            closed_data = data.loc[data.experiment_type == "closed_question"]
            closed_distr = distr.loc[distr.experiment_type == "closed_question"]
            closed_results = _compute_agg_results(closed_data, closed_distr)

            open_unadjusted_data = data.loc[data.experiment_type == "open_question"]
            open_unadjusted_distr = distr.loc[distr.experiment_type == "open_question"]
            open_unadjusted_results = _compute_agg_results(open_unadjusted_data, open_unadjusted_distr)

            open_adjusted_results: CIAggResults | None = None
            if tpr_fpr_ci_df is not None:
                if self.adjusted_bootstrap_distribution is None:
                    raise ValueError("Adjusted bootstrap distribution not computed. Run bootstrap_adjusted() first.")

                tpr_point: float = cast(float, tpr_fpr_ci_df.loc["tpr", "point_estimate"])
                fpr_point: float = cast(float, tpr_fpr_ci_df.loc["fpr", "point_estimate"])
                denom = tpr_point - fpr_point

                open_data = data.loc[data.experiment_type == "open_question"].copy()
                if denom != 0:
                    open_data["auth_adj"] = (open_data["auth"] - fpr_point) / denom
                else:
                    open_data["auth_adj"] = np.nan

                open_distr = self.adjusted_bootstrap_distribution

                open_adjusted_results = _compute_agg_results(open_data, open_distr, agg_col="auth_adj")

            return ConfidenceIntervalResult(
                closed=closed_results,
                open_unadjusted=open_unadjusted_results,
                open_adjusted=open_adjusted_results,
            )
        else:
            vignette_data = data
            vignette_distr = distr
            vignette_results = _compute_agg_results(vignette_data, vignette_distr)
            return vignette_results


class JudgeEvalPredictions(Predictions):
    identifiers = ("open_response", "task", "dataset", "question_id", "seed", "temperature", "model_name")

    def _load_preds(self) -> None:
        preds = pd.read_json(self.path)

        preds = preds.loc[preds.dataset.isin(RELEVANT_DATASETS)].copy()
        if "stratum" in preds.columns:
            preds = preds.drop(columns=["stratum"])
        stratum_cols = ["open_response", "task", "dataset", "question_id", "stratum"]
        strata = pd.read_csv(
            BASE_DIR / "resources" / "input" / "datasets" / "reduction" / "reduction_ground_truth_7_en.csv",
            usecols=stratum_cols,
        ).drop_duplicates(subset=stratum_cols[:-1])
        preds = preds.merge(strata, on=stratum_cols[:-1], how="left", validate="one_to_one")
        preds["user_prompts"] = preds["user_prompts"].apply(lambda x: x[0])
        preds["label_refusal"] = preds["refusal"].map(
            {"Refusal": 1, "Non-refusal": 0, "Non-Answer": 1}
        )  # "Non-answer" gets mapped to None
        preds[["pred_score", "pred_refusal"]] = preds.apply(
            lambda row: pd.Series(parse_and_validate_judge_response(row["response"], row["dataset"])),
            axis=1,
        )
        new_label_cols = preds.apply(
            lambda row: normalize_raw_likert_scale_score(
                row["dataset"], row["question_id"], row["label"], refusal=row["label_refusal"]
            ),
            axis=1,
        )
        new_pred_cols = preds.apply(
            lambda row: normalize_raw_likert_scale_score(
                row["dataset"], row["question_id"], row["pred_score"], refusal=row["pred_refusal"]
            ),
            axis=1,
        )
        assert isinstance(new_label_cols, pd.DataFrame)
        assert isinstance(new_pred_cols, pd.DataFrame)
        new_label_cols = new_label_cols.rename(
            columns={"auth": "label_auth", "normalized_score": "label_normalized_score"}
        )
        new_pred_cols = new_pred_cols.rename(columns={"auth": "pred_auth", "normalized_score": "pred_normalized_score"})

        preds = pd.concat([preds, new_label_cols, new_pred_cols], axis=1)

        self.data = preds


class JudgePerformanceBootstrap:
    def __init__(
        self,
        preds: JudgeEvalPredictions,
        n_repetitions: int = 1000,
        random_state: int = 123,
        population_scores_path: Path | None = None,
    ):
        self.preds = preds
        self.n_repetitions = n_repetitions
        self.bootstrap_distribution: pd.DataFrame | None = None
        self.random_state = random_state
        self.population_scores_path = (
            population_scores_path or BASE_DIR / "eval" / "data" / "tidy" / "construct_scores.csv"
        )
        self.selector_predicted_prevalence: float | None = None

    def _load_selector_predicted_prevalence(self) -> float:
        if self.selector_predicted_prevalence is not None:
            return self.selector_predicted_prevalence
        if self.preds.data is None:
            raise ValueError("Data is None and was not loaded. This should be impossible.")

        population = pd.read_csv(
            self.population_scores_path,
            usecols=["model", "dataset", "language", "experiment_type", "experiment_ablation", "auth"],
        )
        population = population.loc[
            (population["language"] == "en")
            & (population["experiment_type"] == "open_question")
            & (population["experiment_ablation"] == "default")
            & population["dataset"].isin(RELEVANT_DATASETS)
            & population["auth"].notna()
        ]

        if "model" in self.preds.data.columns:
            models = self.preds.data["model"].dropna().unique()
            if len(models) > 0:
                population = population.loc[population["model"].isin(models)]

        if population.empty:
            raise ValueError(f"No population rows available in {self.population_scores_path} for prevalence estimate.")

        self.selector_predicted_prevalence = float(population["auth"].mean())
        return self.selector_predicted_prevalence

    def _estimate_metrics(self, data: pd.DataFrame) -> dict[str, float]:
        if "stratum" not in data.columns:
            raise ValueError("Judge eval predictions need a stratum column for design-corrected TPR/FPR estimates.")
        if data["stratum"].isna().any():
            raise ValueError("Judge eval predictions have rows without stratum assignments.")

        # These strata partition the selector judge's population predictions; "random" is a sampling arm.
        selector_strata = ("pred_autho", "pred_nonautho")
        weighted_data = data.loc[data["stratum"].isin(selector_strata)].copy()
        stratum_counts = weighted_data["stratum"].value_counts()
        missing_strata = [stratum for stratum in selector_strata if stratum_counts.get(stratum, 0) == 0]
        if missing_strata:
            raise ValueError(f"Judge eval predictions need non-empty selector strata: {missing_strata}")

        selector_predicted_prevalence = self._load_selector_predicted_prevalence()
        stratum_probabilities = {
            "pred_autho": selector_predicted_prevalence,
            "pred_nonautho": 1 - selector_predicted_prevalence,
        }
        weighted_data["sampling_weight"] = weighted_data["stratum"].map(
            lambda stratum: stratum_probabilities[stratum] / stratum_counts[stratum]
        )

        metric_data = weighted_data.dropna(subset=["label_auth", "pred_auth", "sampling_weight"])
        y_true_auth = metric_data["label_auth"]
        y_pred_auth = metric_data["pred_auth"]
        weights = metric_data["sampling_weight"]

        tp = weights[(y_pred_auth == 1) & (y_true_auth == 1)].sum()
        tn = weights[(y_pred_auth == 0) & (y_true_auth == 0)].sum()
        fp = weights[(y_pred_auth == 1) & (y_true_auth == 0)].sum()
        fn = weights[(y_pred_auth == 0) & (y_true_auth == 1)].sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        y_true_score = data["label_normalized_score"]
        y_pred_score = data["pred_normalized_score"]
        mse = (y_true_score - y_pred_score).pow(2).mean()
        mae = (y_true_score - y_pred_score).abs().mean()

        return {"tpr": tpr, "fpr": fpr, "mse": mse, "mae": mae}

    @staticmethod
    def _stratified_resample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if data["stratum"].isna().any():
            raise ValueError("Judge eval predictions have rows without stratum assignments.")
        samples = []
        for _, stratum_data in data.groupby("stratum", sort=False):
            idx = rng.choice(len(stratum_data), size=len(stratum_data), replace=True)
            samples.append(stratum_data.iloc[idx])
        return pd.concat(samples, axis=0)

    def bootstrap(self) -> None:
        """Stratified bootstrap for calculating TPR, FPR, and score errors of the judge."""
        if self.preds.data is None:
            raise ValueError("Data is None and was not loaded. This should be impossible.")
        data = self.preds.data

        results = []

        for i in trange(self.n_repetitions, desc="Bootstrapping"):
            random_state = self.random_state + i
            rng = np.random.default_rng(seed=random_state)
            sample = self._stratified_resample(data, rng)
            result = {"repetition": i, **self._estimate_metrics(sample)}
            results.append(result)
        self.bootstrap_distribution = pd.DataFrame(results)

    def cis(self, alpha: float = 0.05) -> pd.DataFrame:
        """BCa confidence intervals given the bootstrapped distributions of the eval metrics (TPR, FPR, MSE, MAE)

        Args:
            alpha: Significance level (default 0.05 for 95% CIs)

        Returns:
            DataFrame with columns [metric, lower, upper, point_estimate]
        """
        if self.bootstrap_distribution is None:
            raise ValueError("Bootstrap distribution not computed. Run bootstrap() first.")
        if self.preds.data is None:
            raise ValueError("Data is None and was not loaded. This should be impossible.")

        data = self.preds.data
        metrics = ["tpr", "fpr", "mse", "mae"]
        point_estimates = self._estimate_metrics(data)
        jk_all = pd.DataFrame([self._estimate_metrics(data.drop(index=idx)) for idx in data.index])

        return bca_cis(metrics, self.bootstrap_distribution, point_estimates, jk_all, alpha)


def bca_cis(
    metrics: list[str],
    bootstrap_distribution: pd.DataFrame,
    point_estimates: dict[str, float],
    jackknife_estimates: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute BCa CIs

    Args:
        metrics (list[str]): names of columns that have the values that you want CIs for
        bootstrap_distribution (pd.DataFrame): one row=one repetition
        point_estimates (dict[str, float]): maps from metric to point estimate
        jackknife_estimates (pd.DataFrame): one row=one repetition
        alpha (float, optional): CI size. Defaults to 0.05.

    Returns:
        pd.DataFrame: CIs with columns point_estimate, lower, upper. Index is metric
    """
    results = []
    for metric in metrics:
        boot_dist = bootstrap_distribution[metric].to_numpy(dtype=float)
        theta_obs = point_estimates[metric]

        # --- Bias-correction z0 ---
        prop_less = np.mean(boot_dist < theta_obs)
        # Clamp to avoid inf from ppf at 0 or 1
        prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
        z0 = stats.norm.ppf(prop_less)

        # --- Acceleration constant a via jackknife ---
        jk_vals = jackknife_estimates[metric].to_numpy()
        jk_mean = jk_vals.mean()
        diffs = jk_mean - jk_vals  # note: sign convention for BCa
        num = np.sum(diffs**3)
        denom = 6.0 * (np.sum(diffs**2) ** 1.5)
        a = num / denom if denom != 0 else 0.0

        # --- BCa quantile adjustment ---
        z_alpha_lo = stats.norm.ppf(alpha / 2)
        z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

        def adjusted_quantile(z_alpha: float) -> float:
            num_ = z0 + z_alpha
            denom_ = 1 - a * (z0 + z_alpha)
            adjusted_z = z0 + num_ / denom_
            return float(stats.norm.cdf(adjusted_z))

        q_lo = adjusted_quantile(z_alpha_lo)
        q_hi = adjusted_quantile(z_alpha_hi)

        lower = np.quantile(boot_dist, q_lo)
        upper = np.quantile(boot_dist, q_hi)

        results.append(
            {
                "metric": metric,
                "point_estimate": theta_obs,
                "lower": lower,
                "upper": upper,
            }
        )
    return pd.DataFrame(results).set_index("metric")
