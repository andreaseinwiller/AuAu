import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.axes import Axes


MODEL_ORDER = [
    "deepseek/deepseek-v3.2",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "utter-project/EuroLLM-9B-Instruct",
    "mistralai/mistral-large-2512",
    "ai-sage/GigaChat-20B-A3B-instruct",
    "Vikhrmodels/QVikhr-3-8B-Instruction",
    "t-tech/T-pro-it-2.0",
    "yandex/YandexGPT-5-Lite-8B-instruct",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "x-ai/grok-4.1-fast",
    "allenai/Olmo-3.1-32B-Instruct",
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct-SFT",
    "allenai/Olmo-3-7B-Instruct-DPO",
    "allenai/Olmo-3-7B-Instruct",
]
MODEL_LABELS = {
    "deepseek/deepseek-v3.2": "Deepseek V3.2 [CN]",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3 30B-A3B 2507 [CN]",
    "utter-project/EuroLLM-9B-Instruct": "EuroLLM 9B [EU]",
    "mistralai/mistral-large-2512": "Mistral Large 2512 [EU]",
    "ai-sage/GigaChat-20B-A3B-instruct": "GigaChat 20B-A3B [RU]",
    "Vikhrmodels/QVikhr-3-8B-Instruction": "QVikhr 3 8B [RU]",
    "t-tech/T-pro-it-2.0": "T-Pro 2.0 [RU]",
    "yandex/YandexGPT-5-Lite-8B-instruct": "YandexGPT 5 Lite 8B [RU]",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5 [US]",
    "openai/gpt-5-mini": "GPT5 Mini [US]",
    "google/gemini-3-flash-preview": "Gemini 3 Flash Prev [US]",
    "x-ai/grok-4.1-fast": "Grok 4.1 Fast [US]",
    "allenai/Olmo-3.1-32B-Instruct": "Olmo 3.1 32B [US]",
    "allenai/Olmo-3-1025-7B": "Olmo3 7B Base [US]",
    "allenai/Olmo-3-7B-Instruct-SFT": "Olmo3 7B Instruct SFT [US]",
    "allenai/Olmo-3-7B-Instruct-DPO": "Olmo3 7B Instruct DPO [US]",
    "allenai/Olmo-3-7B-Instruct": "Olmo3 7B Instruct RLVR [US]",
}
DISPLAY_MODEL_ORDER = [MODEL_LABELS[model] for model in reversed(MODEL_ORDER)]

DIMENSION_ORDER = ["Any", "Aggression", "Submission", "Conventionalism"]
DIMENSION_LABELS = {
    "any": "Any",
    "aggression": "Aggression",
    "submission": "Submission",
    "conventionalism": "Conventionalism",
}
OUTPUT_DIR = Path("figs")
HIGH_RISK_DIR = Path("resources/output/bootstrap_results/high_risk_issuebench_gemma4sft")
LOW_RISK_DIR = Path("resources/output/bootstrap_results/no_high_risk_issuebench_gemma4sft")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator
    both_zero = numerator.eq(0) & denominator.eq(0)
    return ratio.replace([np.inf, -np.inf], np.nan).mask(both_zero, 1.0)


def _load_point_estimates(high_risk_dir: Path, low_risk_dir: Path) -> pd.DataFrame:
    df_high = pd.read_parquet(high_risk_dir / "issuebench__cis.parquet")
    df_low = pd.read_parquet(low_risk_dir / "issuebench__cis.parquet")

    merged = df_high.merge(
        df_low,
        how="inner",
        on=["dimension", "generating_model_name"],
        suffixes=("_highrisk", "_lowrisk"),
    )
    merged = merged.loc[merged["dimension"].isin(DIMENSION_LABELS)].copy()
    merged["dimension"] = merged["dimension"].map(DIMENSION_LABELS)

    return pd.DataFrame(
        {
            "model": merged["generating_model_name"],
            "dimension": merged["dimension"],
            "diff": merged["judge_positive_rate_highrisk"] - merged["judge_positive_rate_lowrisk"],
            "diff_adj": merged["adjusted_positive_rate_highrisk"] - merged["adjusted_positive_rate_lowrisk"],
            "relative_adj": _safe_ratio(
                merged["adjusted_positive_rate_highrisk"],
                merged["adjusted_positive_rate_lowrisk"],
            ),
        }
    )


def _load_bootstrap_intervals(high_risk_dir: Path, low_risk_dir: Path) -> pd.DataFrame:
    df_high = pd.read_parquet(high_risk_dir / "issuebench__adjusted_bootstrap_distribution.parquet")
    df_low = pd.read_parquet(low_risk_dir / "issuebench__adjusted_bootstrap_distribution.parquet")

    merge_cols = ["dimension", "generating_model_name", "repetition"]
    merged = df_high.merge(df_low, how="inner", on=merge_cols, suffixes=("_highrisk", "_lowrisk"))
    merged = merged.loc[merged["dimension"].isin(DIMENSION_LABELS)].copy()
    merged["dimension"] = merged["dimension"].map(DIMENSION_LABELS)
    merged["diff_adj_boot"] = merged["adjusted_positive_rate_highrisk"] - merged["adjusted_positive_rate_lowrisk"]
    merged["relative_adj_boot"] = _safe_ratio(
        merged["adjusted_positive_rate_highrisk"],
        merged["adjusted_positive_rate_lowrisk"],
    )

    interval_rows = []
    for (model, dimension), group_df in merged.groupby(["generating_model_name", "dimension"]):
        diff_low, diff_high = group_df["diff_adj_boot"].quantile([0.025, 0.975])
        finite_relative = group_df["relative_adj_boot"].replace([np.inf, -np.inf], np.nan).dropna()
        relative_low = relative_high = np.nan
        if len(finite_relative) > 0:
            relative_low, relative_high = finite_relative.quantile([0.025, 0.975])

        interval_rows.append(
            {
                "model": model,
                "dimension": dimension,
                "diff_adj_ci_low": diff_low,
                "diff_adj_ci_high": diff_high,
                "relative_adj_ci_low": relative_low,
                "relative_adj_ci_high": relative_high,
            }
        )

    return pd.DataFrame(interval_rows)


def build_high_risk_impact_table(
    high_risk_dir: Path = HIGH_RISK_DIR,
    low_risk_dir: Path = LOW_RISK_DIR,
) -> pd.DataFrame:
    results = _load_point_estimates(high_risk_dir, low_risk_dir)
    intervals = _load_bootstrap_intervals(high_risk_dir, low_risk_dir)
    results = results.merge(intervals, how="left", on=["model", "dimension"])
    results = results.loc[results["model"].isin(MODEL_ORDER)].copy()
    results["model"] = pd.Categorical(results["model"], categories=MODEL_ORDER, ordered=True)
    results["dimension"] = pd.Categorical(results["dimension"], categories=DIMENSION_ORDER, ordered=True)
    results = results.sort_values(["dimension", "model"]).reset_index(drop=True)
    results["model"] = results["model"].map(MODEL_LABELS)
    results["model"] = pd.Categorical(results["model"], categories=DISPLAY_MODEL_ORDER, ordered=True)
    return results


def _apply_no_change_line(ax: Axes, value: float) -> None:
    # ax.axvline(value, color="#b3261e", linewidth=1.2, zorder=0)
    # ax.annotate(
    #     "No Change",
    #     xy=(value, 1.0),
    #     xycoords=("data", "axes fraction"),
    #     xytext=(4, -8),
    #     textcoords="offset points",
    #     rotation=90,
    #     va="top",
    #     ha="left",
    #     color="#8f1d17",
    #     fontsize=8,
    # )
    pass


def _plot_metric(
    results: pd.DataFrame,
    metric: str,
    ci_low: str,
    ci_high: str,
    no_change_value: float,
    xlabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    present_models = set(results["model"].dropna())
    models = [model for model in DISPLAY_MODEL_ORDER if model in present_models]
    height = 4
    golden_ratio = 2
    # golden_ratio=1.61
    # golden_ratio=1.0/1.61
    fig, axes = plt.subplots(1, 4, figsize=(height * golden_ratio, height), sharey=True)
    # fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.8), sharey=True)
    axes = axes.ravel()
    point_color = "#426a8c"
    ci_color = "#20252b"

    for ax, dimension in zip(axes, DIMENSION_ORDER):
        plot_df = results.loc[results["dimension"].eq(dimension)].set_index("model").reindex(models).reset_index()
        y = np.arange(len(plot_df))
        values = plot_df[metric].astype(float)
        finite = values.notna()
        ax.scatter(
            values[finite],
            y[finite],
            color=point_color,
            alpha=0.95,
            s=9,
            zorder=4,
        )

        error_df = plot_df.loc[finite & plot_df[ci_low].notna() & plot_df[ci_high].notna()]
        if len(error_df) > 0:
            error_y = error_df.index.to_numpy()
            error_values = error_df[metric].astype(float).to_numpy()
            xerr = np.vstack(
                [
                    error_values - error_df[ci_low].astype(float).to_numpy(),
                    error_df[ci_high].astype(float).to_numpy() - error_values,
                ]
            )
            ax.errorbar(
                error_values,
                error_y,
                xerr=xerr,
                fmt="none",
                ecolor=ci_color,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                zorder=3,
            )

        if no_change_value == 1:
            finite_values = values.dropna()
            upper = max(no_change_value * 1.2, finite_values.max() * 1.12) if len(finite_values) else 2
            ax.set_xlim(0, upper)
        else:
            bounds = plot_df[[metric, ci_low, ci_high]].to_numpy(dtype=float).ravel()
            bounds = bounds[np.isfinite(bounds)]
            bound = max(abs(bounds).max() * 1.08, 0.01) if len(bounds) else 0.01
            bound = 0.5
            ax.set_xlim(-bound, bound)

        _apply_no_change_line(ax, no_change_value)
        ax.set_title(dimension, fontsize=8, pad=8)
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#d7dce0", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)

    fig.supxlabel(xlabel, y=0.1, x=0.625, fontsize=8)
    fig.supylabel("Model", x=0.08, fontsize=8)
    fig.tight_layout(rect=(0.05, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def visualize_high_risk_impact(results: pd.DataFrame) -> pd.DataFrame:
    _plot_metric(
        results,
        metric="relative_adj",
        ci_low="relative_adj_ci_low",
        ci_high="relative_adj_ci_high",
        no_change_value=1,
        xlabel="ARR(High-Risk Tasks) / ARR(Non-High-Risk Tasks)",
        output_path=OUTPUT_DIR / "issuebench_high_vs_low_relative.pdf",
    )
    _plot_metric(
        results,
        metric="diff_adj",
        ci_low="diff_adj_ci_low",
        ci_high="diff_adj_ci_high",
        no_change_value=0,
        xlabel="ARR(High-Risk Tasks) - ARR(Non-High-Risk Tasks)",
        output_path=OUTPUT_DIR / "issuebench_high_vs_low_absolute.pdf",
    )

    table = []
    for dim in DIMENSION_ORDER:
        share_increased = (results.loc[results["dimension"].eq(dim), "diff_adj"] > 0).mean()
        table.append({"Factor": dim, "Increased ARR Model Share": share_increased})
    return pd.DataFrame(table)


if __name__ == "__main__":
    input_data = build_high_risk_impact_table()
    visualize_high_risk_impact(input_data)
