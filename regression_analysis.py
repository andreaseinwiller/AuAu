# %%
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, cast
from numpy.random import default_rng
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llm_audit.eval.open_response import RELEVANT_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Ridge regression from construct scores to IssueBench.",
    )
    parser.add_argument("--features", choices=["all", "closed_question", "open_question", "vignette"], default="all")
    parser.add_argument(
        "--feature-granularity",
        choices=["aggregated", "dataset"],
        default="aggregated",
        help="Use 3 experiment-type features or 31 experiment-type/dataset features.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["bootstrap", "mean"],
        default="bootstrap",
        help="Sample target bootstrap replicates or use their model-level means.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=100,
        help="Number of feature/target bootstrap evaluations.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument("--use-lasso", action="store_true", default=False)
    parser.add_argument("--remove-olmos", action="store_true", default=False)
    parser.add_argument(
        "--coefficient-plot",
        type=Path,
        default=Path("figs/coefficient_distribution.png"),
        help="Path for the standardized coefficient distribution plot.",
    )
    parser.add_argument(
        "--residual-plot",
        type=Path,
        default=Path("figs/residual_diagnostics.png"),
        help="Path for the out-of-fold residual diagnostics plot.",
    )
    return parser.parse_args()


args = parse_args()

x1 = pd.read_csv("eval/data/tidy/construct_scores_ensemble.csv")
x2 = pd.read_csv("eval/data/tidy/vignette_scores.csv")
x2["experiment_type"] = "vignette"
df = pd.concat((x1, x2), axis="index")

MODELS_TO_REMOVE = [
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct-DPO",
    "allenai/Olmo-3-7B-Instruct-SFT",
]

# DATASETS_W_FACTOR_ANNOT = ["RWA3D", "KSA3", "ACT", "VSA", "ASC"]
# DATASETS_WO_FACTOR_ANNOT = list(sorted(["F", "LAS", "D", "A", "AA", "RWA", "APC"]))
# DATASETS_CAUSES = list(sorted(["DW", "BDW", "CSM"]))
# RELEVANT_DATASETS = DATASETS_W_FACTOR_ANNOT + DATASETS_WO_FACTOR_ANNOT + DATASETS_CAUSES

df = df.loc[df.experiment_ablation == "default"]
df = df.loc[df.dataset.isin(RELEVANT_DATASETS + ["VignetteRWA3D"])]
df = df.sort_values("model")
df.head(3)

# %%
ib = pd.read_parquet(
    "resources/output/bootstrap_results/default_issuebench_gemma4sft/issuebench__adjusted_bootstrap_distribution.parquet"
)
ib = ib.loc[ib.dimension.eq("any")]

if args.remove_olmos:
    ib = ib.loc[~ib.generating_model_name.isin(MODELS_TO_REMOVE)]
    df = df.loc[~df.model.isin(MODELS_TO_REMOVE)]


# %%
ib

# %%
score_col = "auth"

# "aggregated": one feature per experiment type (3 features).
# "dataset": one feature per experiment type and dataset (31 features).
FEATURE_GRANULARITY = args.feature_granularity

# "bootstrap": independently sample one IssueBench bootstrap replicate per model.
# "mean": use each model's mean over all IssueBench bootstrap replicates.
TARGET_MODE = args.target_mode

N_BOOT = args.n_boot
RIDGE_ALPHA = args.ridge_alpha

psycho_item_cols = [
    "model",
    "experiment_type",
    "dataset",
    "statement_id",
    "experiment_ablation",
]

vignette_item_cols = [
    "model",
    "experiment_type",
    "dataset",
    "statement_id",
    "vignette_id",
    "experiment_ablation",
]


def sample_one_repetition(
    frame: pd.DataFrame,
    item_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Uniformly select one repetition row per item."""
    random_order = rng.permutation(len(frame))

    return frame.iloc[random_order].drop_duplicates(item_cols, keep="first")


def build_features(
    df: pd.DataFrame,
    random_state: int,
    feature_granularity: str,
) -> pd.DataFrame:
    rng = default_rng(random_state)

    psycho = sample_one_repetition(
        df.loc[df["experiment_type"].ne("vignette")],
        psycho_item_cols,
        rng,
    )

    vignette = sample_one_repetition(
        df.loc[df["experiment_type"].eq("vignette")],
        vignette_item_cols,
        rng,
    )

    selected = pd.concat([psycho, vignette], ignore_index=True)

    if feature_granularity == "aggregated":
        feature_cols = ["experiment_type"]
    elif feature_granularity == "dataset":
        feature_cols = ["experiment_type", "dataset"]
    else:
        raise ValueError(f"Unknown feature granularity {feature_granularity!r}; expected 'aggregated' or 'dataset'.")

    return (
        selected.groupby(["model", *feature_cols], sort=False, observed=True)[score_col]
        .mean()
        .unstack(feature_cols)
        .sort_index()
        .sort_index(axis="columns")
    )


def build_targets(
    df: pd.DataFrame,
    random_state: int,
    target_mode: str,
) -> pd.Series:
    if target_mode == "bootstrap":
        rng = default_rng(random_state)
        sample = sample_one_repetition(df, ["generating_model_name"], rng)
        targets = sample.set_index("generating_model_name")["adjusted_positive_rate"]
    elif target_mode == "mean":
        targets = df.groupby("generating_model_name")["adjusted_positive_rate"].mean()
    else:
        raise ValueError(f"Unknown target mode {target_mode!r}; expected 'bootstrap' or 'mean'.")

    return targets.sort_index()


def align_features_and_targets(
    features: pd.DataFrame,
    targets: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_targets = features.index.difference(targets.index)
    missing_features = targets.index.difference(features.index)
    if len(missing_targets) or len(missing_features):
        raise ValueError(
            "Features and targets contain different models. "
            f"Missing targets: {missing_targets.tolist()}; "
            f"missing features: {missing_features.tolist()}."
        )

    targets = targets.reindex(features.index)
    if features.isna().any().any() or targets.isna().any():
        raise ValueError("Features and targets must not contain missing values.")

    return features, targets


def format_feature_name(feature: str | tuple[str, ...]) -> str:
    if isinstance(feature, tuple):
        return " / ".join(map(str, feature))
    return str(feature)


def coefficient_frame(model_coefs: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for model_coef in model_coefs:
        for feature, coefficient in model_coef["coefficients"].items():
            records.append(
                {
                    "bootstrap": model_coef["bootstrap"],
                    "fold": model_coef["fold"],
                    "held_out_model": model_coef["held_out_model"],
                    "feature": format_feature_name(feature),
                    "coefficient": coefficient,
                }
            )
    return pd.DataFrame(records)


def plot_coefficient_distribution(
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    feature_order = coefficients.groupby("feature")["coefficient"].median().sort_values().index
    distributions = [coefficients.loc[coefficients["feature"].eq(feature), "coefficient"] for feature in feature_order]

    figure_height = max(4, 0.35 * len(feature_order))
    fig, ax = plt.subplots(figsize=(10, figure_height))
    ax.boxplot(
        distributions,
        orientation="horizontal",
        tick_labels=[str(f) for f in feature_order],
        showfliers=False,
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Standardized Ridge coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Coefficient distributions across bootstrap samples and LOO folds")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def select_features(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature == "all":
        return data

    selected = data.loc[:, feature]
    if isinstance(selected, pd.Series):
        return selected.to_frame()
    return selected


def bootstrap_model_correlation(
    features: pd.DataFrame,
    targets: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Calculate correlations after sampling models with replacement."""
    rng = default_rng(random_state)
    sampled_positions = rng.integers(0, len(features), size=len(features))

    # Reset indices so duplicate sampled models remain separate observations.
    sampled_features = features.iloc[sampled_positions].reset_index(drop=True)
    sampled_targets = targets.iloc[sampled_positions].reset_index(drop=True)
    return pd.concat((sampled_features, sampled_targets), axis="columns").corr()


def summarize_correlations(
    correlations: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for bootstrap, correlation in enumerate(correlations):
        labels = [format_feature_name(column) for column in correlation.columns]
        values = correlation.to_numpy()
        for row, column in zip(*np.triu_indices_from(values, k=1), strict=True):
            records.append(
                {
                    "bootstrap": bootstrap,
                    "variable_1": labels[row],
                    "variable_2": labels[column],
                    "correlation": values[row, column],
                }
            )

    correlation_results = pd.DataFrame(records)
    correlation_summary = (
        correlation_results.groupby(["variable_1", "variable_2"])["correlation"]
        .agg(
            n_valid="count",
            mean="mean",
            median="median",
            ci_low=lambda values: values.quantile(0.025),
            ci_high=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    return correlation_results, correlation_summary


def summarize_residuals(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_bias = predictions.groupby("bootstrap")["residual"].mean()
    overall_summary = pd.DataFrame(
        [
            {
                "mean_bias": bootstrap_bias.mean(),
                "median_bias": bootstrap_bias.median(),
                "ci_low": bootstrap_bias.quantile(0.025),
                "ci_high": bootstrap_bias.quantile(0.975),
                "fraction_positive": bootstrap_bias.gt(0).mean(),
            }
        ]
    )

    model_summary = (
        cast(
            pd.DataFrame,
            predictions.groupby("model")["residual"].agg(
                mean_bias="mean",
                median_bias="median",
                ci_low=lambda values: values.quantile(0.025),
                ci_high=lambda values: values.quantile(0.975),
                fraction_positive=lambda values: values.gt(0).mean(),
            ),
        )
        .sort_values("mean_bias")
        .reset_index()
    )
    return overall_summary, model_summary


def plot_residual_diagnostics(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    bootstrap_bias = predictions.groupby("bootstrap")["residual"].mean()
    model_order = predictions.groupby("model")["residual"].mean().sort_values().index
    model_residuals = [predictions.loc[predictions["model"].eq(model), "residual"] for model in model_order]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, max(5, 0.35 * len(model_order))),
        gridspec_kw={"width_ratios": [1, 2]},
    )
    axes[0].hist(bootstrap_bias, bins="auto", edgecolor="black")
    axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0].axvline(bootstrap_bias.mean(), color="tab:red", linewidth=2)
    axes[0].set_xlabel("Mean residual per bootstrap")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall out-of-fold bias")

    axes[1].boxplot(
        model_residuals,
        orientation="horizontal",
        tick_labels=model_order,
        showfliers=False,
    )
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Residual (prediction - target)")
    axes[1].set_ylabel("Held-out model")
    axes[1].set_title("Out-of-fold residual distributions by model")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


data_splitter = LeaveOneOut()
scores = []
model_coefs = []
predictions = []
correlations = []
for i in range(N_BOOT):
    X = build_features(
        df,
        random_state=i,
        feature_granularity=FEATURE_GRANULARITY,
    )
    y = build_targets(ib, random_state=N_BOOT + i, target_mode=TARGET_MODE)
    X, y = align_features_and_targets(X, y)
    X = select_features(X, args.features)

    correlations.append(
        bootstrap_model_correlation(
            X,
            y,
            random_state=2 * N_BOOT + i,
        )
    )

    val_predictions = []
    for fold, (train_idx, val_idx) in enumerate(data_splitter.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=RIDGE_ALPHA) if args.use_lasso else Ridge(alpha=RIDGE_ALPHA),
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(X_val)[0])
        val_predictions.append(pred)
        predictions.append(
            {
                "bootstrap": i,
                "model": X.index[val_idx[0]],
                "target": float(y.iloc[val_idx[0]]),
                "prediction": pred,
                "residual": pred - float(y.iloc[val_idx[0]]),
            }
        )

        ridge = model.named_steps["lasso"] if args.use_lasso else model.named_steps["ridge"]
        model_coefs.append(
            {
                "bootstrap": i,
                "fold": fold,
                "held_out_model": X.index[val_idx[0]],
                "coefficients": pd.Series(ridge.coef_, index=X.columns),
                "intercept": float(ridge.intercept_),
            }
        )

    scores.append(
        {
            "r2": r2_score(y, val_predictions),
            "mae": mean_absolute_error(y, val_predictions),
            "bootstrap": i,
            "feature_granularity": FEATURE_GRANULARITY,
            "target_mode": TARGET_MODE,
            "n_features": X.shape[1],
        }
    )

    if N_BOOT == 1:
        plt.scatter(val_predictions, np.arange(len(val_predictions)), label="Prediction")
        plt.scatter(y, np.arange(len(val_predictions)), color="orange", label="Ground truth")
        plt.yticks(ticks=np.arange(len(val_predictions)), labels=[str(i) for i in X.index])
        plt.legend()


# %%
results = pd.DataFrame(scores)
prediction_results = pd.DataFrame(predictions)
coefficient_results = coefficient_frame(model_coefs)
correlation_results, correlation_summary = summarize_correlations(correlations)
residual_summary, model_residual_summary = summarize_residuals(prediction_results)

configuration = pd.DataFrame(
    [
        {
            "feature_granularity": FEATURE_GRANULARITY,
            "target_mode": TARGET_MODE,
            "n_boot": N_BOOT,
            "ridge_alpha": RIDGE_ALPHA,
            "n_models": X.shape[0],
            "n_features": X.shape[1],
        }
    ]
)

score_summary = (
    results[["r2", "mae"]]
    .describe(percentiles=[0.025, 0.5, 0.975])
    .loc[["mean", "std", "min", "2.5%", "50%", "97.5%", "max"]]
    .rename(index={"50%": "median"})
    .T.reset_index(names="metric")
)

print("\n## Regression configuration\n")
print(configuration.to_markdown(index=False, floatfmt=".4f"))
print("\n## Bootstrap score summary\n")
print(score_summary.to_markdown(index=False, floatfmt=".4f"))
print("\n## Bootstrap correlation summary\n")
print(correlation_summary.to_markdown(index=False, floatfmt=".4f"))
print("\n## Out-of-fold residual bias summary\n")
print(residual_summary.to_markdown(index=False, floatfmt=".4f"))
print("\nPositive residuals indicate overprediction; negative residuals indicate underprediction.")
print("\n## Out-of-fold residual bias by model\n")
print(model_residual_summary.to_markdown(index=False, floatfmt=".4f"))

plot_coefficient_distribution(coefficient_results, args.coefficient_plot)
print(f"\nCoefficient distribution plot: `{args.coefficient_plot}`")
plot_residual_diagnostics(prediction_results, args.residual_plot)
print(f"Residual diagnostics plot: `{args.residual_plot}`")

# %%
