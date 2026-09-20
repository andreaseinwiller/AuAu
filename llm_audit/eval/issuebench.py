import json
import re
import krippendorff

import numpy as np
import pandas as pd
import sklearn.metrics

from typing import Any, Callable, Iterable, NamedTuple
from dataclasses import dataclass
from pathlib import Path
from joblib import Parallel, delayed
from loguru import logger
from quapy.method.aggregative import ACC, CC, MAX, MS2, PACC, PCC, T50, HDy
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import trange


LABEL_DIMENSIONS = ["aggression", "submission", "conventionalism", "refusal"]
AUTHORITARIAN_DIMENSIONS = ["aggression", "submission", "conventionalism"]
JUDGE_COUNT_DIMENSIONS = AUTHORITARIAN_DIMENSIONS + ["refusal", "any"]
CHOICE_TO_VAL: dict[str, float] = {"Yes": 1.0, "No": -1.0}
DEFAULT_VAL = -1.0

# Rename map handles the historical typo in the annotation setup
_LABEL_RENAMES = {"label_agression": "label_aggression"}


def _first_if_list(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _normalize_yes_no_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else -1.0
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1", "+1"}:
            return 1.0
        if normalized in {"no", "n", "false", "0", "-1"}:
            return -1.0
    return float("nan")


def parse_issuebench_judge_response(response: Any) -> dict[str, float]:
    """Parse an IssueBench judge JSON response into yes/no dimension values."""
    parsed: Any | None = None
    if isinstance(response, dict):
        parsed = response
    elif isinstance(response, str) and response.strip():
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None

    parsed = _first_if_list(parsed)
    result = {dim: float("nan") for dim in LABEL_DIMENSIONS}
    if not isinstance(parsed, dict):
        return result

    for dim in LABEL_DIMENSIONS:
        candidates: tuple[str, ...] = (dim, f"label_{dim}")
        if dim == "aggression":
            candidates = (*candidates, "agression", "label_agression")
        for key in candidates:
            if key in parsed:
                result[dim] = _normalize_yes_no_value(parsed[key])
                break
    return result


def _add_issuebench_any_label(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    cols = [f"{prefix}{dim}" for dim in AUTHORITARIAN_DIMENSIONS]
    any_col = f"{prefix}any"
    complete = df[cols].notna().all(axis=1)
    positive = df[cols].eq(1.0).any(axis=1)
    df[any_col] = np.where(complete, np.where(positive, 1.0, -1.0), np.nan)
    return df


def _extract_issuebench_text_to_annotate(prompt: Any) -> Any:
    prompt = _first_if_list(prompt)
    if not isinstance(prompt, str):
        return None
    marker = "Text to annotate:"
    if marker not in prompt:
        return None
    return prompt.rsplit(marker, 1)[1].strip()


def _parse_label_studio_item(item: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat record from a single Label Studio annotation item."""
    data = item["data"]
    record: dict[str, Any] = {
        "judge_model": data["model_name"],
        "judge_prompt": data["user_prompts"][0],
        "model": data["generating_model_name"],
        "task": data["task"],
        "text": data["text"],
    }
    for dim in LABEL_DIMENSIONS:
        # Use the legacy typo key so existing exports still parse correctly
        legacy_key = "agression" if dim == "aggression" else dim
        record[f"label_{legacy_key}"] = DEFAULT_VAL

    for annot in item["annotations"][0]["result"]:
        key = f"label_{annot['from_name']}"
        record[key] = CHOICE_TO_VAL.get(annot["value"]["choices"][0], DEFAULT_VAL)

    return record


def _add_derived_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_LABEL_RENAMES)
    autho_dims = ["aggression", "submission", "conventionalism"]
    df["label_combined"] = df[[f"label_{d}" for d in autho_dims]].eq(1.0).any(axis=1).map({True: 1.0, False: -1.0})
    return df


def build_annot_df(annots: list[dict[str, Any]]) -> pd.DataFrame:
    records = [_parse_label_studio_item(item) for item in annots]
    df = pd.DataFrame.from_records(records)
    return _add_derived_labels(df)


def load_issuebench_annotations(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        with open(p) as f:
            frames.append(build_annot_df(json.load(f)))
    return pd.concat(frames, axis=0, ignore_index=True)


def merge_annotations_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    merged = df1.merge(df2, how="outer", on=["judge_prompt", "task", "model"], suffixes=["_1", "_2"])
    return merged.dropna(axis=0)


def convert_to_groundtruth_format(paths: list[str]) -> list[dict[str, Any]]:
    annots = load_issuebench_annotations(paths)
    annots = annots.rename(
        columns={
            "judge_prompt": "prompt",
            "text": "response",
        }
    )
    data = [row.to_dict() for _, row in annots.iterrows()]
    return data


def _dimension_agreement(df: pd.DataFrame, dimension: str) -> dict[str, Any]:
    col1, col2 = f"label_{dimension}_1", f"label_{dimension}_2"
    y1, y2 = df[col1].to_numpy(), df[col2].to_numpy()

    agreement_rate = (y1 == y2).mean()

    try:
        alpha = krippendorff.alpha(
            reliability_data=df[[col1, col2]].T.to_numpy(),
            level_of_measurement="nominal",
        )
    except ValueError as e:
        logger.error(f"[{dimension}] Krippendorff alpha failed: {e}")
        alpha = float("nan")

    cohens_kappa = sklearn.metrics.cohen_kappa_score(y1, y2)

    return {
        "dimension": dimension,
        "agreement": agreement_rate,
        "krippendorff_alpha": alpha,
        "cohens_kappa": cohens_kappa,
    }


def agreement_statistics(df: pd.DataFrame) -> pd.DataFrame:
    all_dimensions = LABEL_DIMENSIONS + ["combined"]
    scores = []
    for dim in all_dimensions:
        logger.info(f"Calculating agreement statistics for {dim}")
        scores.append(_dimension_agreement(df, dim))
    return pd.DataFrame.from_records(scores)


def describe_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-rater, per-dimension breakdown: counts and rates of positive labels,
    plus a 'conflict' column showing how often the two raters disagree on that dimension.
    """
    all_dimensions = LABEL_DIMENSIONS + ["combined"]
    records = []

    for dim in all_dimensions:
        col1, col2 = f"label_{dim}_1", f"label_{dim}_2"
        y1, y2 = df[col1], df[col2]
        n = len(df)

        pos1, pos2 = (y1 == 1.0).sum(), (y2 == 1.0).sum()
        conflict = (y1 != y2).sum()

        # Cases where exactly one rater fired positive - the "contested" items
        only_1_positive = ((y1 == 1.0) & (y2 != 1.0)).sum()
        only_2_positive = ((y1 != 1.0) & (y2 == 1.0)).sum()

        records.append(
            {
                "dimension": dim,
                "n": n,
                "rater_1_pos": pos1,
                "rater_1_neg": n - pos1,
                "rater_1_pos_rate": pos1 / n,
                "rater_2_pos": pos2,
                "rater_2_neg": n - pos2,
                "rater_2_pos_rate": pos2 / n,
                "conflicts": conflict,
                "conflict_rate": conflict / n,
                "only_rater_1_positive": only_1_positive,
                "only_rater_2_positive": only_2_positive,
                # Prevalence-weighted: if both base rates are near 0 or 1,
                # agreement is trivially high - flag dimensions where this masks low kappa
                "prevalence_imbalance": abs((pos1 / n) - (pos2 / n)),
            }
        )

    return pd.DataFrame.from_records(records)


def agreement(paths1: list[str], paths2: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    a1 = load_issuebench_annotations(paths1)
    a2 = load_issuebench_annotations(paths2)
    merged = merge_annotations_dfs(a1, a2)
    return agreement_statistics(merged), describe_annotations(merged)


# Bootstrapping
class IssueBenchPrevalenceCI(NamedTuple):
    point_estimates: pd.DataFrame
    bootstrap_distribution: pd.DataFrame
    intervals: pd.DataFrame


@dataclass
class IssueBenchGenerations:
    path: Path | str
    data: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.data is None:
            self._load_data()

    def _load_data(self) -> None:
        path = Path(self.path)
        records = []
        for fpath in path.rglob("*.json"):
            with fpath.open("r") as f:
                data = json.load(f)

            for item in data:
                item["user_prompts"] = item["user_prompts"][0]
                item["system_prompts"] = item["system_prompts"][0]
                records.append(item)

        self.data = pd.DataFrame(records)


@dataclass
class IssueBenchJudgePredictions:
    path: Path | str | Iterable[Path | str] | None = None
    data: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.data is None:
            if self.path is None:
                raise ValueError("path is required when data is not provided.")
            self._load_data()
        else:
            self.data = self._prepare_data(self.data.copy())

    def _iter_paths(self) -> Iterable[Path]:
        if self.path is None:
            return []
        paths: Iterable[Path | str]
        if isinstance(self.path, (str, Path)):
            paths = [self.path]
        else:
            paths = self.path

        json_paths: list[Path] = []
        for item in paths:
            path = Path(item)
            if path.is_file():
                json_paths.append(path)
            elif path.is_dir():
                json_paths.extend(sorted(path.rglob("*.json")))
            else:
                raise FileNotFoundError(f"IssueBench judge path does not exist: {path}")
        return json_paths

    def _load_data(self) -> None:
        records: list[dict[str, Any]] = []
        for fpath in self._iter_paths():
            with fpath.open("r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                loaded = [loaded]
            if not isinstance(loaded, list):
                raise ValueError(f"Expected a JSON object or list in {fpath}")
            for item in loaded:
                if not isinstance(item, dict):
                    raise ValueError(f"Expected JSON records in {fpath}")
                record = item.copy()
                record["source_path"] = str(fpath)
                records.append(record)

        self.data = self._prepare_data(pd.DataFrame.from_records(records))

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "response" not in data.columns:
            raise ValueError("IssueBench judge data must contain a response column.")

        data = data.copy()
        if "judge_model" not in data.columns and "model_name" in data.columns:
            data["judge_model"] = data["model_name"]
        if "generating_model_name" not in data.columns:
            for candidate in ("generating_model", "model"):
                if candidate in data.columns:
                    data["generating_model_name"] = data[candidate]
                    break
        if "prompt" not in data.columns and "user_prompts" in data.columns:
            data["prompt"] = data["user_prompts"].apply(_first_if_list)
        if "task" not in data.columns:
            for candidate in ("generating_user_prompts", "user_prompts"):
                if candidate in data.columns:
                    data["task"] = data[candidate].apply(_first_if_list)
                    break
        if "text" not in data.columns:
            if "generating_response" in data.columns:
                data["text"] = data["generating_response"]
            elif "prompt" in data.columns:
                data["text"] = data["prompt"].apply(_extract_issuebench_text_to_annotate)
        if "raw_response" not in data.columns:
            data["raw_response"] = data["response"]

        parsed = data["response"].apply(parse_issuebench_judge_response).apply(pd.Series)
        for dim in LABEL_DIMENSIONS:
            data[dim] = parsed[dim]
        data = _add_issuebench_any_label(data)
        return data


def _read_issuebench_ground_truth_path(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    with path.open("r") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        return pd.DataFrame.from_records(loaded)
    if isinstance(loaded, dict):
        return pd.DataFrame(loaded)
    raise ValueError(f"Expected list or column-oriented JSON object in {path}")


def _enrich_issuebench_ground_truth_from_v3(data: pd.DataFrame, source_path: Path | None) -> pd.DataFrame:
    if source_path is None or "response" not in data.columns:
        return data

    metadata_path = source_path.with_name("ground_truth_data_rebuttal_v3_en.json")
    if not metadata_path.exists() or metadata_path == source_path:
        return data

    missing_metadata_cols = [col for col in ("model", "task", "predicted_as_autho") if col not in data.columns]
    if not missing_metadata_cols:
        return data

    metadata = load_issuebench_ground_truth(metadata_path).loc[:, ["response", "model", "task", "predicted_as_autho"]]
    metadata = metadata.drop_duplicates(subset=["response"])
    enriched = data.merge(metadata, on="response", how="left", suffixes=("", "__metadata"))
    for col in ("model", "task", "predicted_as_autho"):
        metadata_col = f"{col}__metadata"
        if metadata_col not in enriched.columns:
            continue
        if col not in data.columns:
            enriched[col] = enriched[metadata_col]
        else:
            enriched[col] = enriched[col].where(enriched[col].notna(), enriched[metadata_col])
        enriched = enriched.drop(columns=[metadata_col])
    return enriched


def load_issuebench_ground_truth(ground_truth: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Load IssueBench manual annotations and normalize labels for judge evaluation."""
    source_path: Path | None = None
    if isinstance(ground_truth, pd.DataFrame):
        data = ground_truth.copy()
    else:
        source_path = Path(ground_truth)
        data = _read_issuebench_ground_truth_path(source_path)

    data = data.rename(columns=_LABEL_RENAMES).copy()
    if "response" not in data.columns and "prompt" in data.columns:
        data["response"] = data["prompt"].apply(_extract_issuebench_text_to_annotate)
    data = _enrich_issuebench_ground_truth_from_v3(data, source_path)

    required_cols = {
        "response",
        "label_aggression",
        "label_submission",
        "label_conventionalism",
        "label_refusal",
    }
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"IssueBench ground truth is missing required columns: {sorted(missing_cols)}")
    if data["response"].isna().any():
        raise ValueError("IssueBench ground truth has rows without response text.")

    for dim in LABEL_DIMENSIONS:
        data[f"label_{dim}"] = data[f"label_{dim}"].apply(_normalize_yes_no_value)
    data = _add_issuebench_any_label(data, prefix="label_")

    if "predicted_as_autho" in data.columns:
        missing_strata = data["predicted_as_autho"].isna()
        if missing_strata.any():
            logger.warning(
                f"Dropping {int(missing_strata.sum())} IssueBench ground-truth rows without "
                "predicted_as_autho strata after metadata enrichment."
            )
            data = data.loc[~missing_strata].reset_index(drop=True)
        data["predicted_as_autho"] = data["predicted_as_autho"].astype(bool)
    return data


def _issuebench_binary_indicator(data: pd.DataFrame, dimension: str, prefix: str = "") -> pd.Series:
    value_col = f"{prefix}{dimension}"
    refusal_col = f"{prefix}refusal"
    if value_col not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)

    values = data[value_col]
    result = pd.Series(np.nan, index=data.index, dtype=float)
    result.loc[values.notna()] = values.loc[values.notna()].eq(1.0).astype(float)

    if dimension != "refusal" and refusal_col in data.columns:
        refusal_positive = data[refusal_col].eq(1.0)
        result.loc[refusal_positive] = 0.0
    return result


class IssueBenchJudgePerformanceBootstrap:
    """Design-corrected bootstrap of IssueBench judge TPR/FPR against manual labels."""

    def __init__(
        self,
        preds: IssueBenchJudgePredictions | pd.DataFrame,
        ground_truth: Path | str | pd.DataFrame,
        selector_preds: IssueBenchJudgePredictions | pd.DataFrame | Path | str | Iterable[Path | str],
        n_repetitions: int = 1000,
        random_state: int = 123,
    ):
        if isinstance(preds, IssueBenchJudgePredictions):
            if preds.data is None:
                raise ValueError("preds.data is None and was not loaded.")
            self.preds_data = preds.data.reset_index(drop=True).copy()
        else:
            prepared = IssueBenchJudgePredictions(data=preds).data
            if prepared is None:
                raise ValueError("IssueBench judge data was not prepared.")
            self.preds_data = prepared.reset_index(drop=True)

        self.ground_truth = load_issuebench_ground_truth(ground_truth).reset_index(drop=True)
        if isinstance(selector_preds, IssueBenchJudgePredictions):
            if selector_preds.data is None:
                raise ValueError("selector_preds.data is None and was not loaded.")
            self.selector_data = selector_preds.data.reset_index(drop=True).copy()
        elif isinstance(selector_preds, pd.DataFrame):
            prepared = IssueBenchJudgePredictions(data=selector_preds).data
            if prepared is None:
                raise ValueError("IssueBench selector judge data was not prepared.")
            self.selector_data = prepared.reset_index(drop=True)
        else:
            selector = IssueBenchJudgePredictions(selector_preds)
            if selector.data is None:
                raise ValueError("IssueBench selector judge data was not loaded.")
            self.selector_data = selector.data.reset_index(drop=True)

        self.n_repetitions = n_repetitions
        self.random_state = random_state
        self.selector_predicted_prevalence: float | None = None
        self.data = self._merge_predictions_to_ground_truth()
        self.bootstrap_distribution: pd.DataFrame | None = None
        self.point_estimates: pd.DataFrame | None = None

    @staticmethod
    def _merge_key_frame(data: pd.DataFrame, text_col: str) -> pd.DataFrame:
        if text_col not in data.columns:
            raise ValueError(f"IssueBench data is missing text merge column: {text_col}")
        result = data.copy()
        result["__merge_text"] = result[text_col].astype(str)
        return result

    def _ground_truth_identity_cols(self) -> list[str]:
        if (
            {"model", "task"}.issubset(self.ground_truth.columns)
            and self.ground_truth[["model", "task"]].notna().all().all()
            and {"generating_model_name", "task"}.issubset(self.preds_data.columns)
        ):
            return ["__merge_model", "__merge_task", "__merge_text"]
        return ["__merge_text"]

    @staticmethod
    def _validate_sampling_strata(data: pd.DataFrame) -> pd.DataFrame:
        if "predicted_as_autho" not in data.columns:
            raise ValueError("IssueBench ground truth needs predicted_as_autho sampling strata.")
        if data["predicted_as_autho"].isna().any():
            raise ValueError("IssueBench ground truth has rows without predicted_as_autho sampling strata.")
        result = data.copy()
        result["predicted_as_autho"] = result["predicted_as_autho"].astype(bool)
        return result

    def _merge_predictions_to_ground_truth(self) -> pd.DataFrame:
        if "text" not in self.preds_data.columns:
            raise ValueError("IssueBench judge predictions need a text column extracted from the judge prompt.")
        if self.preds_data["text"].isna().any():
            raise ValueError("IssueBench judge predictions have rows without generated text for ground-truth matching.")

        preds = self._merge_key_frame(self.preds_data, "text")
        truth = self._merge_key_frame(self.ground_truth, "response")
        merge_cols = self._ground_truth_identity_cols()
        if "__merge_model" in merge_cols:
            preds["__merge_model"] = preds["generating_model_name"].astype(str)
            truth["__merge_model"] = truth["model"].astype(str)
        if "__merge_task" in merge_cols:
            preds["__merge_task"] = preds["task"].astype(str)
            truth["__merge_task"] = truth["task"].astype(str)

        merged = preds.merge(
            truth,
            on=merge_cols,
            how="inner",
            suffixes=("_pred", "_truth"),
        )
        if merged.empty:
            raise ValueError("No IssueBench judge predictions matched the manual ground truth.")
        if len(merged) < len(truth):
            logger.warning(f"Matched {len(merged)} of {len(truth)} IssueBench ground-truth rows to judge predictions.")
        return self._validate_sampling_strata(merged).reset_index(drop=True)

    def _load_selector_predicted_prevalence(self) -> float:
        if self.selector_predicted_prevalence is not None:
            return self.selector_predicted_prevalence
        valid_any = self.selector_data["any"].dropna()
        if valid_any.empty:
            raise ValueError("IssueBench selector judge predictions have no valid any labels.")
        self.selector_predicted_prevalence = float(valid_any.eq(1.0).mean())
        return self.selector_predicted_prevalence

    def _with_sampling_weights(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        if data["predicted_as_autho"].isna().any():
            raise ValueError("IssueBench judge-performance data has rows without predicted_as_autho strata.")
        weighted = data.copy()
        weighted["predicted_as_autho"] = weighted["predicted_as_autho"].astype(bool)
        stratum_counts = weighted["predicted_as_autho"].value_counts()
        missing = [stratum for stratum in (True, False) if stratum_counts.get(stratum, 0) == 0]
        if missing:
            raise ValueError(f"IssueBench judge-performance data needs non-empty strata: {missing}")

        selector_positive_rate = self._load_selector_predicted_prevalence()
        probabilities = {True: selector_positive_rate, False: 1 - selector_positive_rate}
        weighted["sampling_weight"] = weighted["predicted_as_autho"].map(
            lambda stratum: probabilities[stratum] / stratum_counts[stratum]
        )
        meta = {
            "selector_positive_rate": selector_positive_rate,
            "n_predicted_as_autho": float(stratum_counts[True]),
            "n_predicted_as_nonautho": float(stratum_counts[False]),
            "weight_predicted_as_autho": probabilities[True] / stratum_counts[True],
            "weight_predicted_as_nonautho": probabilities[False] / stratum_counts[False],
        }
        return weighted, meta

    def _estimate_dimension_metrics(self, data: pd.DataFrame, dimension: str, meta: dict[str, float]) -> dict[str, Any]:
        y_true = _issuebench_binary_indicator(data, dimension, prefix="label_")
        y_pred = _issuebench_binary_indicator(data, dimension, prefix="")
        metric_data = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "sampling_weight": data["sampling_weight"],
            }
        ).dropna()
        weights = metric_data["sampling_weight"]
        y_true = metric_data["y_true"]
        y_pred = metric_data["y_pred"]

        tp = float(weights[(y_pred == 1) & (y_true == 1)].sum())
        tn = float(weights[(y_pred == 0) & (y_true == 0)].sum())
        fp = float(weights[(y_pred == 1) & (y_true == 0)].sum())
        fn = float(weights[(y_pred == 0) & (y_true == 1)].sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

        return {
            "dimension": dimension,  # str not float
            "tpr": float(tpr),
            "fpr": float(fpr),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n": float(len(metric_data)),
            **meta,
        }

    def _estimate_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        weighted, meta = self._with_sampling_weights(data)
        records = [self._estimate_dimension_metrics(weighted, dimension, meta) for dimension in JUDGE_COUNT_DIMENSIONS]
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _stratified_resample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        samples = []
        for _, stratum_data in data.groupby("predicted_as_autho", sort=False):
            idx = rng.choice(len(stratum_data), size=len(stratum_data), replace=True)
            samples.append(stratum_data.iloc[idx])
        return pd.concat(samples, axis=0).reset_index(drop=True)

    def point_estimate(self) -> pd.DataFrame:
        self.point_estimates = self._estimate_metrics(self.data)
        return self.point_estimates

    def bootstrap(self) -> pd.DataFrame:
        results = []
        for i in trange(self.n_repetitions, desc="Bootstrapping IssueBench judge performance"):
            rng = np.random.default_rng(seed=self.random_state + i)
            result = self._estimate_metrics(self._stratified_resample(self.data, rng))
            result["repetition"] = i
            results.append(result)
        self.bootstrap_distribution = pd.concat(results, axis=0, ignore_index=True)
        return self.bootstrap_distribution

    def cis(self, alpha: float = 0.05) -> pd.DataFrame:
        if self.bootstrap_distribution is None:
            raise ValueError("Bootstrap distribution not computed. Run bootstrap() first.")
        if self.point_estimates is None:
            self.point_estimate()
        assert self.point_estimates is not None

        records = []
        for metric in ("tpr", "fpr"):
            intervals = (
                self.bootstrap_distribution.groupby("dimension")[metric]
                .quantile(np.array([alpha / 2, 1 - alpha / 2]))
                .unstack()
                .reset_index()
                .rename(columns={alpha / 2: "lower", 1 - alpha / 2: "upper"})
            )
            point = self.point_estimates.loc[:, ["dimension", metric]].rename(columns={metric: "point_estimate"})
            metric_df = point.merge(intervals, on="dimension", how="left")
            metric_df["metric"] = metric
            records.append(metric_df)
        return pd.concat(records, axis=0, ignore_index=True).loc[
            :, ["dimension", "metric", "point_estimate", "lower", "upper"]
        ]


class IssueBenchJudgeCountBootstrap:
    """Bootstrap observed IssueBench judge-positive rates for comparative analysis."""

    def __init__(
        self,
        preds: IssueBenchJudgePredictions | pd.DataFrame,
        *,
        group_cols: tuple[str, ...] = ("generating_model_name",),
        n_repetitions: int = 1000,
        random_state: int = 123,
    ):
        if isinstance(preds, IssueBenchJudgePredictions):
            if preds.data is None:
                raise ValueError("preds.data is None and was not loaded.")
            self.data = preds.data.reset_index(drop=True).copy()
        else:
            predictions_data = IssueBenchJudgePredictions(data=preds).data
            if predictions_data is None:
                raise ValueError("IssueBench judge data was not prepared.")
            self.data = predictions_data.reset_index(drop=True)

        self.group_cols = tuple(group_cols)
        self.n_repetitions = n_repetitions
        self.random_state = random_state
        self.quantification_method = "judge_count"
        self.bootstrap_distribution: pd.DataFrame | None = None
        self.adjusted_bootstrap_distribution: pd.DataFrame | None = None
        self.point_estimates: pd.DataFrame | None = None
        self.adjusted_point_estimates: pd.DataFrame | None = None

        missing_group_cols = set(self.group_cols) - set(self.data.columns)
        if missing_group_cols:
            raise ValueError(f"group_cols are not present in judge data: {sorted(missing_group_cols)}")

    def _valid_mask(self, data: pd.DataFrame, dimension: str) -> pd.Series:
        if dimension == "refusal":
            return data["refusal"].notna()
        return data[dimension].notna() & data["refusal"].notna() & data["refusal"].ne(1.0)

    def _summarize_group(self, group: pd.DataFrame) -> pd.DataFrame:
        records = []
        n = len(group)
        n_refusal = int(group["refusal"].eq(1.0).sum())
        for dimension in JUDGE_COUNT_DIMENSIONS:
            valid = self._valid_mask(group, dimension)
            n_valid = int(valid.sum())
            judge_positive_rate = float(group.loc[valid, dimension].eq(1.0).mean()) if n_valid else float("nan")
            records.append(
                {
                    "dimension": dimension,
                    "quantification_method": self.quantification_method,
                    "judge_positive_rate": judge_positive_rate,
                    "n": n,
                    "n_valid": n_valid,
                    "n_refusal": n_refusal,
                }
            )
        return pd.DataFrame.from_records(records)

    def _summarize(self, data: pd.DataFrame, repetition: int | None = None) -> pd.DataFrame:
        if self.group_cols:
            parts = []
            for group_values, group in data.groupby(list(self.group_cols), dropna=False, sort=False):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                summary = self._summarize_group(group)
                for col, value in zip(self.group_cols, group_values):
                    summary[col] = value
                parts.append(summary)
            result = pd.concat(parts, axis=0, ignore_index=True)
        else:
            result = self._summarize_group(data)

        if repetition is not None:
            result["repetition"] = repetition
        return result

    def point_estimate(self) -> pd.DataFrame:
        self.point_estimates = self._summarize(self.data)
        return self.point_estimates

    def _stable_valid_mask(self, data: pd.DataFrame, dimension: str) -> pd.Series:
        if dimension == "refusal":
            return data["refusal"].notna()
        return data["refusal"].notna() & (data[dimension].notna() | data["refusal"].eq(1.0))

    def _summarize_adjusted_group(self, group: pd.DataFrame) -> pd.DataFrame:
        records = []
        n = len(group)
        n_refusal = int(group["refusal"].eq(1.0).sum())
        for dimension in JUDGE_COUNT_DIMENSIONS:
            valid = self._stable_valid_mask(group, dimension)
            n_valid = int(valid.sum())
            if n_valid:
                observed_positive = _issuebench_binary_indicator(group.loc[valid], dimension, prefix="")
                judge_positive_rate = float(observed_positive.mean())
            else:
                judge_positive_rate = float("nan")
            records.append(
                {
                    "dimension": dimension,
                    "quantification_method": "judge_count_adjusted",
                    "judge_positive_rate": judge_positive_rate,
                    "n": n,
                    "n_valid": n_valid,
                    "n_refusal": n_refusal,
                }
            )
        return pd.DataFrame.from_records(records)

    def _summarize_adjusted_observed(self, data: pd.DataFrame, repetition: int | None = None) -> pd.DataFrame:
        if self.group_cols:
            parts = []
            for group_values, group in data.groupby(list(self.group_cols), dropna=False, sort=False):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                summary = self._summarize_adjusted_group(group)
                for col, value in zip(self.group_cols, group_values):
                    summary[col] = value
                parts.append(summary)
            result = pd.concat(parts, axis=0, ignore_index=True)
        else:
            result = self._summarize_adjusted_group(data)

        if repetition is not None:
            result["repetition"] = repetition
        return result

    @staticmethod
    def _apply_adjusted_count(observed: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
        result = observed.merge(performance, on="dimension", how="left", validate="many_to_one")
        denom = result["tpr"] - result["fpr"]
        result["adjusted_positive_rate"] = np.where(
            denom.ne(0) & denom.notna(),
            (result["judge_positive_rate"] - result["fpr"]) / denom,
            np.nan,
        )
        result["adjusted_positive_rate"] = result["adjusted_positive_rate"].clip(0.0, 1.0)
        return result

    def _adjusted_point_estimate_from_distribution(self, performance_distribution: pd.DataFrame) -> pd.DataFrame:
        performance_point = performance_distribution.groupby("dimension")[["tpr", "fpr"]].mean().reset_index()
        return self._apply_adjusted_count(self._summarize_adjusted_observed(self.data), performance_point)

    def _bootstrap_data(self, rng: np.random.Generator) -> pd.DataFrame:
        if not self.group_cols:
            idx = rng.choice(len(self.data), size=len(self.data), replace=True)
            return self.data.iloc[idx].reset_index(drop=True)

        samples = []
        for _, group in self.data.groupby(list(self.group_cols), dropna=False, sort=False):
            sampled_idx = rng.choice(len(group), size=len(group), replace=True)
            samples.append(group.iloc[sampled_idx])
        return pd.concat(samples, axis=0).reset_index(drop=True)

    def bootstrap(self) -> pd.DataFrame:
        results = []
        for i in trange(self.n_repetitions, desc="Bootstrapping IssueBench judge counts"):
            rng = np.random.default_rng(seed=self.random_state + i)
            results.append(self._summarize(self._bootstrap_data(rng), repetition=i))
        self.bootstrap_distribution = pd.concat(results, axis=0, ignore_index=True)
        return self.bootstrap_distribution

    def bootstrap_adjusted(self, performance_distribution: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"repetition", "dimension", "tpr", "fpr"}
        if not required_cols.issubset(performance_distribution.columns):
            raise ValueError(f"performance_distribution must contain columns: {sorted(required_cols)}")

        results = []
        for i in trange(self.n_repetitions, desc="Bootstrapping adjusted IssueBench judge counts"):
            performance = performance_distribution.loc[
                performance_distribution["repetition"] == i, ["dimension", "tpr", "fpr"]
            ]
            if performance.empty:
                raise ValueError(f"No judge-performance bootstrap rows found for repetition {i}.")
            rng = np.random.default_rng(seed=self.random_state + i)
            observed = self._summarize_adjusted_observed(self._bootstrap_data(rng), repetition=i)
            adjusted = self._apply_adjusted_count(observed, performance)
            results.append(adjusted)

        self.adjusted_bootstrap_distribution = pd.concat(results, axis=0, ignore_index=True)
        self.adjusted_point_estimates = self._adjusted_point_estimate_from_distribution(performance_distribution)
        return self.adjusted_bootstrap_distribution

    def cis(self, alpha: float = 0.05) -> IssueBenchPrevalenceCI:
        if self.bootstrap_distribution is None:
            raise ValueError("Bootstrap distribution not computed. Run bootstrap() first.")
        if self.point_estimates is None:
            self.point_estimate()

        assert self.point_estimates is not None
        grouping_cols = list(self.group_cols) + ["dimension", "quantification_method"]
        intervals = (
            self.bootstrap_distribution.groupby(grouping_cols)["judge_positive_rate"]
            .quantile(np.array([alpha / 2, 1 - alpha / 2]))
            .unstack()
            .reset_index()
        )
        intervals = intervals.rename(columns={alpha / 2: "ci_low", 1 - alpha / 2: "ci_high"})
        intervals = self.point_estimates.merge(intervals, on=grouping_cols, how="left")

        if self.adjusted_bootstrap_distribution is not None:
            adjusted_grouping_cols = list(self.group_cols) + ["dimension", "quantification_method"]
            adjusted_intervals = (
                self.adjusted_bootstrap_distribution.groupby(adjusted_grouping_cols)["adjusted_positive_rate"]
                .quantile(np.array([alpha / 2, 1 - alpha / 2]))
                .unstack()
                .reset_index()
            )
            adjusted_intervals = adjusted_intervals.rename(
                columns={alpha / 2: "adjusted_ci_low", 1 - alpha / 2: "adjusted_ci_high"}
            )
            if self.adjusted_point_estimates is None:
                adjusted_points = (
                    self.adjusted_bootstrap_distribution.groupby(adjusted_grouping_cols)["adjusted_positive_rate"]
                    .mean()
                    .reset_index()
                )
            else:
                adjusted_points = self.adjusted_point_estimates.loc[
                    :, adjusted_grouping_cols + ["adjusted_positive_rate", "tpr", "fpr"]
                ]
            adjusted_summary = adjusted_points.merge(adjusted_intervals, on=adjusted_grouping_cols, how="left")
            adjusted_summary["quantification_method"] = "judge_count"
            adjusted_summary = adjusted_summary.rename(columns={"tpr": "adjustment_tpr", "fpr": "adjustment_fpr"})
            intervals = intervals.merge(
                adjusted_summary,
                on=grouping_cols,
                how="left",
            )

        return IssueBenchPrevalenceCI(
            point_estimates=self.point_estimates,
            bootstrap_distribution=self.bootstrap_distribution,
            intervals=intervals,
        )


def _make_tfidf_classifier(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
            ("classifier", SGDClassifier(loss="log_loss", random_state=random_state, max_iter=1000)),
        ]
    )


def _resolve_text_column(data: pd.DataFrame, text_col: str | None) -> str:
    if text_col is not None:
        if text_col not in data.columns:
            raise ValueError(f"text_col={text_col!r} is not present in data columns.")
        return text_col

    for candidate in ("text", "response", "open_response"):
        if candidate in data.columns:
            return candidate
    raise ValueError("Could not infer text column. Pass text_col explicitly.")


def _labels_to_binary(labels: pd.Series, positive_label: float | int | str = 1.0) -> np.ndarray[Any, np.dtype[Any]]:
    return (labels == positive_label).astype(int).to_numpy()


PROBABILITY_QUANTIFICATION_METHODS = ("sgd_spa", "calibrated_sgd_spa")
QUAPY_QUANTIFICATION_METHODS = ("hdy", "pacc", "quapy_pacc", "cc", "pcc", "acc", "t50", "max", "ms2")
SUPPORTED_QUANTIFICATION_METHODS = PROBABILITY_QUANTIFICATION_METHODS + QUAPY_QUANTIFICATION_METHODS
DEFAULT_QUANTIFICATION_METHODS = SUPPORTED_QUANTIFICATION_METHODS
_BOOTSTRAP_GROUP_COL = "__bootstrap_group_id"


def _positive_class_probs(classifier: Any, texts: Iterable[str]) -> np.ndarray[Any, np.dtype[Any]]:
    probs = classifier.predict_proba(list(texts))
    classes = getattr(classifier, "classes_", None)
    if classes is None and hasattr(classifier, "named_steps"):
        classes = classifier.named_steps["classifier"].classes_
    if classes is None:
        raise ValueError("Classifier does not expose classes_ after fitting.")
    positive_idx = list(classes).index(1)
    return np.asarray(probs[:, positive_idx])


def repeated_stratified_cv_predictions(
    texts: Iterable[str],
    labels: Iterable[int],
    *,
    groups: Iterable[Any] | None = None,
    classifier: Any | None = None,
    classifier_factory: Callable[[int, np.ndarray[Any, np.dtype[Any]]], Any] | None = None,
    n_splits: int = 10,
    n_repeats: int = 1,
    random_state: int = 123,
) -> pd.DataFrame:
    """Out-of-fold probabilities from repeated stratified CV."""
    texts = list(texts)
    labels = np.asarray(list(labels), dtype=int)
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if groups is not None:
        groups = np.asarray(list(groups))
        if len(groups) != len(labels):
            raise ValueError("groups and labels must have the same length.")

    if groups is None:
        class_counts = np.bincount(labels, minlength=2)
    else:
        grouped = pd.DataFrame({"label": labels, "group": groups}).drop_duplicates("group")
        class_counts = np.bincount(grouped["label"].to_numpy(dtype=int), minlength=2)
    max_splits = int(class_counts.min())
    if max_splits < 2:
        raise ValueError("Need at least two examples of each class for stratified CV.")

    n_splits = min(n_splits, max_splits)
    base_classifier = classifier or _make_tfidf_classifier(random_state)
    records: list[dict[str, Any]] = []

    for repeat in range(n_repeats):
        if groups is None:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
            split_iter = splitter.split(texts, labels)
        else:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
            split_iter = splitter.split(texts, labels, groups)
        for fold, (train_idx, val_idx) in enumerate(split_iter):
            fold_random_state = random_state + repeat * n_splits + fold
            if classifier_factory is None:
                fold_classifier = clone(base_classifier)
            else:
                fold_classifier = classifier_factory(fold_random_state, labels[train_idx])
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            fold_classifier.fit(train_texts, labels[train_idx])
            val_probs = _positive_class_probs(fold_classifier, val_texts)

            records.extend(
                {
                    "row_id": int(row_id),
                    "repeat": repeat,
                    "fold": fold,
                    "label": int(labels[row_id]),
                    "prob": float(prob),
                }
                for row_id, prob in zip(val_idx, val_probs)
            )

    return pd.DataFrame.from_records(records)


def choose_threshold_by_j_statistic(oof_predictions: pd.DataFrame, max_fpr: float = 0.053) -> dict[str, float]:
    required_cols = {"label", "prob"}
    if not required_cols.issubset(oof_predictions.columns):
        raise ValueError(f"oof_predictions must contain columns: {required_cols}")

    y_true = oof_predictions["label"].astype(int)
    y_prob = oof_predictions["prob"].astype(float)
    if y_true.nunique() != 2:
        return {"threshold": float("nan"), "threshold_max_fpr": float(max_fpr), "j_statistic": float("nan")}

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_prob)
    eligible = fpr < max_fpr
    if not eligible.any():
        return {"threshold": float("nan"), "threshold_max_fpr": float(max_fpr), "j_statistic": float("nan")}

    eligible_indices = np.flatnonzero(eligible)
    j_values = tpr[eligible_indices] - fpr[eligible_indices]
    best_idx = int(eligible_indices[np.argmax(j_values)])
    return {
        "threshold": float(thresholds[best_idx]),
        "threshold_max_fpr": float(max_fpr),
        "j_statistic": float(tpr[best_idx] - fpr[best_idx]),
    }


def evaluate_classifier_predictions(
    oof_predictions: pd.DataFrame,
    threshold: float = 0.5,
    threshold_max_fpr: float | None = None,
) -> dict[str, float]:
    required_cols = {"label", "prob"}
    if not required_cols.issubset(oof_predictions.columns):
        raise ValueError(f"oof_predictions must contain columns: {required_cols}")

    # Warning: threshold is selected to maximize TPR-FPR on the same data that where TPR and FPR are reported.
    # Basically, the selection step is tested on its own selection data, not a validation set.
    threshold_stats = (
        choose_threshold_by_j_statistic(oof_predictions, threshold_max_fpr)
        if threshold_max_fpr is not None
        else {"threshold": float(threshold), "threshold_max_fpr": float("nan"), "j_statistic": float("nan")}
    )
    selected_threshold = threshold_stats["threshold"]

    y_true = oof_predictions["label"].astype(int)
    y_prob = oof_predictions["prob"].astype(float)
    y_pred = (y_prob >= selected_threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    auc = sklearn.metrics.roc_auc_score(y_true, y_prob) if y_true.nunique() == 2 else float("nan")

    # Statistics needed for the "Scaled Probability Average" (SPA) method from Bella et al., 2010. "Quantification via Probability Estimators"
    avg_positive_proba_for_negatives = y_prob.loc[y_true == 0].mean()
    avg_positive_proba_for_positives = y_prob.loc[y_true == 1].mean()

    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "auc": float(auc),
        "brier": float(sklearn.metrics.brier_score_loss(y_true, y_prob)),
        "observed_prevalence": float(y_true.mean()),
        "mean_predicted_prevalence": float(y_prob.mean()),
        "calibration_bias": float(y_prob.mean() - y_true.mean()),
        "threshold": float(selected_threshold),
        "threshold_max_fpr": threshold_stats["threshold_max_fpr"],
        "j_statistic": float(tpr - fpr),
        "avg_proba_for_negatives": avg_positive_proba_for_negatives,
        "avg_proba_for_positives": avg_positive_proba_for_positives,
    }


def probabilistic_count(
    probs: Iterable[float],
    eval_stats: dict[str, float] | None = None,
    *,
    adjust: bool = True,
    clip: bool = True,
) -> float:
    prevalence = float(np.mean(list(probs)))
    if not adjust or eval_stats is None:
        return prevalence

    # Using the SPA method from Bella et al., 2010. "Quantification via Probability Estimators"
    proba_pos = eval_stats["avg_proba_for_positives"]
    proba_neg = eval_stats["avg_proba_for_negatives"]
    denom = proba_pos - proba_neg
    if not np.isfinite(denom) or denom == 0:
        return float("nan")

    adjusted = (prevalence - proba_neg) / denom
    if clip:
        adjusted = float(np.clip(adjusted, 0.0, 1.0))
    return float(adjusted)


class IssueBenchClassifierPrevalenceBootstrap:
    """Estimate IssueBench label prevalence with classifier and corpus bootstrap uncertainty."""

    def __init__(
        self,
        labelled_data: pd.DataFrame,
        target_data: IssueBenchGenerations | pd.DataFrame,
        *,
        label_col: str = "label_combined",
        text_col: str | None = None,
        target_text_col: str | None = None,
        positive_label: float | int | str = 1.0,
        group_cols: tuple[str, ...] = (),
        n_repetitions: int = 1000,
        n_splits: int = 10,
        n_cv_repeats: int = 1,
        random_state: int = 123,
        threshold: float = 0.5,
        threshold_max_fpr: float | None = 0.053,
        adjust: bool = True,
        bootstrap_classifier: bool = True,
        n_jobs: int | None = 1,
        quantification_methods: tuple[str, ...] = DEFAULT_QUANTIFICATION_METHODS,
        calibration_method: str = "sigmoid",
        calibration_cv: int = 3,
    ):
        self.labelled_data = labelled_data.reset_index(drop=True).copy()
        bootstrap_group_col = _BOOTSTRAP_GROUP_COL
        while bootstrap_group_col in self.labelled_data.columns:
            bootstrap_group_col = f"_{bootstrap_group_col}"
        self._bootstrap_group_col = bootstrap_group_col
        self.labelled_data[self._bootstrap_group_col] = np.arange(len(self.labelled_data))
        if isinstance(target_data, IssueBenchGenerations):
            if target_data.data is None:
                raise ValueError("target_data.data is None and was not loaded.")
            self.target_data = target_data.data.reset_index(drop=True)
        else:
            self.target_data = target_data.reset_index(drop=True)
        self.label_col = label_col
        self.text_col = _resolve_text_column(self.labelled_data, text_col)
        self.target_text_col = _resolve_text_column(self.target_data, target_text_col)
        self.positive_label = positive_label
        self.group_cols = tuple(group_cols)
        self.n_repetitions = n_repetitions
        self.n_splits = n_splits
        self.n_cv_repeats = n_cv_repeats
        self.random_state = random_state
        self.threshold = threshold
        self.threshold_max_fpr = threshold_max_fpr
        self.adjust = adjust
        self.bootstrap_classifier = bootstrap_classifier
        self.n_jobs = n_jobs
        self.quantification_methods = tuple(quantification_methods)
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.bootstrap_distribution: pd.DataFrame | None = None
        self._target_predictions: dict[str, pd.DataFrame] = {}
        self._quapy_quantifiers: dict[str, Any] = {}
        self.point_estimates: pd.DataFrame | None = None

        missing_group_cols = set(self.group_cols) - set(self.target_data.columns)
        if missing_group_cols:
            raise ValueError(f"group_cols are not present in target_data: {sorted(missing_group_cols)}")
        if self.label_col not in self.labelled_data.columns:
            raise ValueError(f"label_col={self.label_col!r} is not present in labelled_data.")
        unknown_methods = set(self.quantification_methods) - set(SUPPORTED_QUANTIFICATION_METHODS)
        if unknown_methods:
            raise ValueError(f"Unknown quantification_methods: {sorted(unknown_methods)}")

    def _text_and_labels(self, data: pd.DataFrame) -> tuple[list[str], np.ndarray[Any, np.dtype[Any]]]:
        texts = data[self.text_col].astype(str).tolist()
        labels = _labels_to_binary(data[self.label_col], positive_label=self.positive_label)
        return texts, labels

    def _calibration_cv_for_labels(self, labels: np.ndarray[Any, np.dtype[Any]]) -> int:
        class_counts = np.bincount(labels, minlength=2)
        max_cv = int(class_counts.min())
        if max_cv < 2:
            raise ValueError("Need at least two examples of each class for calibrated classifier CV.")
        return min(self.calibration_cv, max_cv)

    def _quantifier_val_split_for_labels(self, labels: np.ndarray[Any, np.dtype[Any]]) -> int:
        class_counts = np.bincount(labels, minlength=2)
        max_cv = int(class_counts.min())
        if max_cv < 2:
            raise ValueError("Need at least two examples of each class for quapy validation CV.")
        return min(self.n_splits, max_cv)

    def _is_quapy_quantification_method(self, method: str) -> bool:
        return method in QUAPY_QUANTIFICATION_METHODS

    def _classifier_method_for_eval(self, method: str) -> str:
        if self._is_quapy_quantification_method(method):
            return "sgd_spa"
        return method

    def _make_classifier(
        self,
        method: str,
        random_state: int,
        labels: np.ndarray[Any, np.dtype[Any]] | None = None,
    ) -> Any:
        base_classifier = _make_tfidf_classifier(random_state)
        if method == "sgd_spa":
            return base_classifier
        if method == "calibrated_sgd_spa":
            if labels is None:
                raise ValueError("labels are required to configure calibrated classifier CV.")
            return CalibratedClassifierCV(
                estimator=base_classifier,
                method=self.calibration_method,
                cv=self._calibration_cv_for_labels(labels),
            )
        raise ValueError(f"Unknown classifier-backed probability method: {method}")

    def _make_quapy_quantifier(self, method: str, random_state: int, labels: np.ndarray[Any, np.dtype[Any]]) -> Any:
        base_classifier = _make_tfidf_classifier(random_state)
        if method == "hdy":
            return HDy(base_classifier, val_split=self._quantifier_val_split_for_labels(labels))
        if method in {"pacc", "quapy_pacc"}:
            return PACC(base_classifier, val_split=self._quantifier_val_split_for_labels(labels))
        if method == "cc":
            return CC(base_classifier)
        if method == "pcc":
            return PCC(base_classifier)
        if method == "acc":
            return ACC(base_classifier, val_split=self._quantifier_val_split_for_labels(labels))
        if method == "t50":
            return T50(base_classifier, val_split=self._quantifier_val_split_for_labels(labels))
        if method == "max":
            return MAX(base_classifier, val_split=self._quantifier_val_split_for_labels(labels))
        if method == "ms2":
            return MS2(base_classifier, val_split=self._quantifier_val_split_for_labels(labels))
        raise ValueError(f"Unknown quapy quantification method: {method}")

    def _fit_quapy_quantifier(self, labelled_data: pd.DataFrame, random_state: int, method: str) -> Any:
        texts, labels = self._text_and_labels(labelled_data)
        quantifier = self._make_quapy_quantifier(method, random_state, labels)
        quantifier.fit(texts, labels)
        return quantifier

    def _fit_predict_target(
        self,
        labelled_data: pd.DataFrame,
        target_data: pd.DataFrame,
        random_state: int,
        method: str = "sgd_spa",
    ) -> pd.DataFrame:
        texts, labels = self._text_and_labels(labelled_data)
        classifier = self._make_classifier(method, random_state, labels)
        classifier.fit(texts, labels)

        target_probs = _positive_class_probs(classifier, target_data[self.target_text_col].astype(str).tolist())
        preds = (
            target_data.loc[:, list(self.group_cols)].copy()
            if self.group_cols
            else pd.DataFrame(index=target_data.index)
        )
        preds["prob"] = target_probs
        return preds

    def _eval_stats(
        self,
        labelled_data: pd.DataFrame,
        random_state: int,
        method: str = "sgd_spa",
    ) -> dict[str, float]:
        texts, labels = self._text_and_labels(labelled_data)
        eval_method = self._classifier_method_for_eval(method)
        groups = None
        if self._bootstrap_group_col in labelled_data.columns:
            candidate_groups = labelled_data[self._bootstrap_group_col].to_numpy()
            if pd.Series(candidate_groups).duplicated().any():
                groups = candidate_groups
        oof_predictions = repeated_stratified_cv_predictions(
            texts,
            labels,
            groups=groups,
            classifier_factory=lambda seed, train_labels: self._make_classifier(eval_method, seed, train_labels),
            n_splits=self.n_splits,
            n_repeats=self.n_cv_repeats,
            random_state=random_state,
        )
        return evaluate_classifier_predictions(
            oof_predictions,
            threshold=self.threshold,
            threshold_max_fpr=self.threshold_max_fpr,
        )

    def _summarize_prevalence(
        self,
        target_predictions: pd.DataFrame,
        eval_stats: dict[str, float],
        method: str,
        repetition: int | None = None,
    ) -> pd.DataFrame:
        grouping_cols = list(self.group_cols)

        def summarize(group: pd.DataFrame) -> pd.Series:
            raw = probabilistic_count(group["prob"], adjust=False)
            adjusted = probabilistic_count(group["prob"], eval_stats, adjust=self.adjust)
            return pd.Series({"prevalence": adjusted, "raw_prevalence": raw, "n": len(group)})

        if grouping_cols:
            result = target_predictions.groupby(grouping_cols).apply(summarize, include_groups=False).reset_index()  # type: ignore[call-overload]
        else:
            result = summarize(target_predictions).to_frame().T

        result["quantification_method"] = method
        for metric, value in eval_stats.items():
            result[metric] = value
        if repetition is not None:
            result["repetition"] = repetition
        return pd.DataFrame(result)

    def _fit_predict_full_target(self, method: str) -> pd.DataFrame:
        if method not in self._target_predictions:
            self._target_predictions[method] = self._fit_predict_target(
                self.labelled_data,
                self.target_data,
                self.random_state,
                method,
            )
        return self._target_predictions[method]

    def _fit_full_quapy_quantifier(self, method: str) -> Any:
        if method not in self._quapy_quantifiers:
            self._quapy_quantifiers[method] = self._fit_quapy_quantifier(
                self.labelled_data,
                self.random_state,
                method,
            )
        return self._quapy_quantifiers[method]

    def _positive_prevalence_from_quantifier(self, quantifier: Any, texts: Iterable[str]) -> float:
        prevalences = quantifier.predict(list(texts))
        classes = getattr(quantifier, "classes_", np.array([0, 1]))
        positive_idx = list(classes).index(1)
        return float(prevalences[positive_idx])

    def _raw_prevalence_from_quantifier(self, quantifier: Any, texts: Iterable[str]) -> float:
        text_list = list(texts)
        try:
            return probabilistic_count(_positive_class_probs(quantifier.classifier, text_list), adjust=False)
        except (AttributeError, ValueError):
            predictions = quantifier.classifier.predict(text_list)
            return float(np.mean(np.asarray(predictions) == 1))

    def _summarize_quapy_prevalence(
        self,
        quantifier: Any,
        target_data: pd.DataFrame,
        eval_stats: dict[str, float],
        method: str,
        repetition: int | None = None,
    ) -> pd.DataFrame:
        grouping_cols = list(self.group_cols)

        def summarize(group: pd.DataFrame) -> pd.Series:
            texts = group[self.target_text_col].astype(str).tolist()
            prevalence = self._positive_prevalence_from_quantifier(quantifier, texts)
            raw = self._raw_prevalence_from_quantifier(quantifier, texts)
            return pd.Series({"prevalence": prevalence, "raw_prevalence": raw, "n": len(group)})

        if grouping_cols:
            result = target_data.groupby(grouping_cols).apply(summarize, include_groups=False).reset_index()  # type: ignore[call-overload]
        else:
            result = summarize(target_data).to_frame().T

        result["quantification_method"] = method
        for metric, value in eval_stats.items():
            result[metric] = value
        if repetition is not None:
            result["repetition"] = repetition
        return pd.DataFrame(result)

    def point_estimate(self) -> pd.DataFrame:
        results = []
        for method in self.quantification_methods:
            eval_stats = self._eval_stats(self.labelled_data, self.random_state, method)
            if self._is_quapy_quantification_method(method):
                quantifier = self._fit_full_quapy_quantifier(method)
                results.append(self._summarize_quapy_prevalence(quantifier, self.target_data, eval_stats, method))
            else:
                target_predictions = self._fit_predict_full_target(method)
                results.append(self._summarize_prevalence(target_predictions, eval_stats, method))
        self.point_estimates = pd.concat(results, axis=0, ignore_index=True)
        return self.point_estimates

    def _bootstrap_labelled_data(self, rng: np.random.Generator) -> pd.DataFrame:
        labels = _labels_to_binary(self.labelled_data[self.label_col], positive_label=self.positive_label)
        samples = []
        for label in (0, 1):
            class_data = self.labelled_data.loc[labels == label]
            if class_data.empty:
                raise ValueError("Both positive and negative labels are required for classifier bootstrapping.")
            sampled_idx = rng.choice(len(class_data), size=len(class_data), replace=True)
            samples.append(class_data.iloc[sampled_idx])
        shuffle_seed = int(rng.integers(0, 2**32 - 1))
        return pd.DataFrame(
            pd.concat(samples, axis=0).sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
        )

    def _bootstrap_target_data(self, rng: np.random.Generator) -> pd.DataFrame:
        if not self.group_cols:
            idx = rng.choice(len(self.target_data), size=len(self.target_data), replace=True)
            return self.target_data.iloc[idx].reset_index(drop=True)

        samples = []
        for _, group in self.target_data.groupby(list(self.group_cols), sort=False):
            sampled_idx = rng.choice(len(group), size=len(group), replace=True)
            samples.append(group.iloc[sampled_idx])
        return pd.concat(samples, axis=0).reset_index(drop=True)

    def _bootstrap_target_predictions(self, target_predictions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if not self.group_cols:
            idx = rng.choice(len(target_predictions), size=len(target_predictions), replace=True)
            return target_predictions.iloc[idx].reset_index(drop=True)

        samples = []
        for _, group in target_predictions.groupby(list(self.group_cols), sort=False):
            sampled_idx = rng.choice(len(group), size=len(group), replace=True)
            samples.append(group.iloc[sampled_idx])
        return pd.concat(samples, axis=0).reset_index(drop=True)

    def _bootstrap_repetition(self, i: int, target_predictions: dict[str, Any] | None = None) -> pd.DataFrame:
        random_state = self.random_state + i
        rng = np.random.default_rng(seed=random_state)
        boot_labelled = self._bootstrap_labelled_data(rng)

        results = []
        for method in self.quantification_methods:
            eval_stats = self._eval_stats(boot_labelled, random_state, method)
            if self._is_quapy_quantification_method(method):
                if self.bootstrap_classifier:
                    quantifier = self._fit_quapy_quantifier(boot_labelled, random_state, method)
                elif target_predictions is None:
                    raise ValueError("target_predictions must be provided when bootstrap_classifier=False.")
                else:
                    quantifier = target_predictions[method]
                boot_target_data = self._bootstrap_target_data(rng)
                results.append(
                    self._summarize_quapy_prevalence(quantifier, boot_target_data, eval_stats, method, repetition=i)
                )
                continue

            if self.bootstrap_classifier:
                method_target_predictions = self._fit_predict_target(
                    boot_labelled, self.target_data, random_state, method
                )
            elif target_predictions is None:
                raise ValueError("target_predictions must be provided when bootstrap_classifier=False.")
            else:
                method_target_predictions = target_predictions[method]

            boot_target_predictions = self._bootstrap_target_predictions(method_target_predictions, rng)
            results.append(self._summarize_prevalence(boot_target_predictions, eval_stats, method, repetition=i))
        return pd.concat(results, axis=0, ignore_index=True)

    def bootstrap(self) -> pd.DataFrame:
        target_predictions = (
            None
            if self.bootstrap_classifier
            else {
                method: (
                    self._fit_full_quapy_quantifier(method)
                    if self._is_quapy_quantification_method(method)
                    else self._fit_predict_full_target(method)
                )
                for method in self.quantification_methods
            }
        )
        iterator = trange(self.n_repetitions, desc="Bootstrapping IssueBench classifier prevalence")
        if self.n_jobs == 1:
            results = [self._bootstrap_repetition(i, target_predictions) for i in iterator]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._bootstrap_repetition)(i, target_predictions) for i in iterator
            )

        self.bootstrap_distribution = pd.concat(results, axis=0, ignore_index=True)
        return self.bootstrap_distribution

    def cis(self, alpha: float = 0.05) -> IssueBenchPrevalenceCI:
        if self.bootstrap_distribution is None:
            raise ValueError("Bootstrap distribution not computed. Run bootstrap() first.")
        if self.point_estimates is None:
            self.point_estimate()

        assert self.point_estimates is not None
        grouping_cols = list(self.group_cols) + ["quantification_method"]

        intervals = (
            self.bootstrap_distribution.groupby(grouping_cols)["prevalence"]
            .quantile(np.array([alpha / 2, 1 - alpha / 2]))
            .unstack()
            .reset_index()
        )
        intervals = intervals.rename(columns={alpha / 2: "ci_low", 1 - alpha / 2: "ci_high"})
        intervals = self.point_estimates.merge(intervals, on=grouping_cols, how="left")

        return IssueBenchPrevalenceCI(
            point_estimates=self.point_estimates,
            bootstrap_distribution=self.bootstrap_distribution,
            intervals=intervals,
        )
