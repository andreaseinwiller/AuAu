from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BOOTSTRAP_ROOT = Path("resources/output/bootstrap_results")
DEFAULT_DIR = BOOTSTRAP_ROOT / "default_issuebench_gemma4sft"
AUTHSYS_DIR = BOOTSTRAP_ROOT / "authsys_issuebench_gemma4sft"
FIG_DIR = Path("figs")
DIMENSION = "any"

MODEL_LABELS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen 3",
    "Vikhrmodels/QVikhr-3-8B-Instruction": "QVikhr 3",
    "ai-sage/GigaChat-20B-A3B-instruct": "GigaChat",
    "allenai/Olmo-3-1025-7B": "Olmo 3 Base",
    "allenai/Olmo-3-7B-Instruct": "Olmo 3 RLVR",
    "allenai/Olmo-3-7B-Instruct-DPO": "Olmo 3 DPO",
    "allenai/Olmo-3-7B-Instruct-SFT": "Olmo 3 SFT",
    "allenai/Olmo-3.1-32B-Instruct": "Olmo 3.1",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
    "google/gemini-3-flash-preview": "Gemini 3",
    "mistralai/mistral-large-2512": "Mistral",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "t-tech/T-pro-it-2.0": "T-Pro 2.0",
    "utter-project/EuroLLM-9B-Instruct": "EuroLLM",
    "x-ai/grok-4.1-fast": "Grok 4.1",
    "yandex/YandexGPT-5-Lite-8B-instruct": "YandexGPT",
}


def read_ci(prompt_dir: Path, prompt: str) -> pd.DataFrame:
    df = pd.read_parquet(prompt_dir / "issuebench__cis.parquet")
    df = df.loc[df["dimension"].eq(DIMENSION)].copy()
    df["prompt"] = prompt
    df["model_label"] = df["generating_model_name"].map(MODEL_LABELS).fillna(df["generating_model_name"])
    return df


def read_bootstrap(prompt_dir: Path, prompt: str) -> pd.DataFrame:
    df = pd.read_parquet(prompt_dir / "issuebench__adjusted_bootstrap_distribution.parquet")
    df = df.loc[df["dimension"].eq(DIMENSION)].copy()
    df["prompt"] = prompt
    return df


def load_point_estimates() -> pd.DataFrame:
    default = read_ci(DEFAULT_DIR, "Default")
    authsys = read_ci(AUTHSYS_DIR, "AuthSys")
    return pd.concat([default, authsys], ignore_index=True)


def load_delta_estimates() -> pd.DataFrame:
    default_ci = read_ci(DEFAULT_DIR, "Default").set_index("generating_model_name")
    authsys_ci = read_ci(AUTHSYS_DIR, "AuthSys").set_index("generating_model_name")

    point = pd.DataFrame(
        {
            "default_rate": default_ci["adjusted_positive_rate"],
            "authsys_rate": authsys_ci["adjusted_positive_rate"],
            "model_label": default_ci["model_label"],
        }
    )
    point["delta"] = point["authsys_rate"] - point["default_rate"]

    default_boot = read_bootstrap(DEFAULT_DIR, "Default")
    authsys_boot = read_bootstrap(AUTHSYS_DIR, "AuthSys")
    merged = authsys_boot.merge(
        default_boot,
        on=["generating_model_name", "repetition"],
        suffixes=("_authsys", "_default"),
        validate="one_to_one",
    )
    merged["delta"] = merged["adjusted_positive_rate_authsys"] - merged["adjusted_positive_rate_default"]

    boot_ci = (
        merged.groupby("generating_model_name")["delta"]
        .quantile(np.array([0.025, 0.975]))
        .unstack()
        .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    )

    out = point.join(boot_ci)
    out["ci_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    out = out.sort_values("delta", ascending=True).reset_index(names="generating_model_name")
    return out


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)


def plot_box_scatter(points: pd.DataFrame) -> None:
    wide = points.pivot(index="generating_model_name", columns="prompt", values="adjusted_positive_rate")
    wide = wide.loc[(wide["AuthSys"] - wide["Default"]).sort_values(ascending=False).index]
    labels = points.drop_duplicates("generating_model_name").set_index("generating_model_name")["model_label"]

    rng = np.random.default_rng(7)
    jitter = pd.Series(rng.uniform(-0.055, 0.055, size=len(wide)), index=wide.index)

    fig, ax = plt.subplots(figsize=(3.35, 3.35 / 1.6))
    # fig, ax = plt.subplots(figsize=(3.35, 3.35/1.6))
    box = ax.boxplot(
        [wide["Default"] * 100, wide["AuthSys"] * 100],
        positions=[0, 1],
        widths=0.42,
        vert=False,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], ["#d8dee9", "#e8c4a2"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for model, row in wide.iterrows():
        model = str(model)
        xs = row[["Default", "AuthSys"]].to_numpy(dtype=float) * 100
        ys = np.array([0, 1]) + jitter.loc[model]
        ax.plot(xs, ys, color="#9aa0a6", linewidth=0.6, alpha=0.45, zorder=1)
        ax.scatter(xs[0], ys[0], s=14, color="#4f6f8f", edgecolor="white", linewidth=0.35, zorder=3)
        ax.scatter(xs[1], ys[1], s=14, color="#b45f38", edgecolor="white", linewidth=0.35, zorder=3)

        authsys_rate = xs[1]
        if authsys_rate >= 55 or authsys_rate <= 3:
            name = labels.loc[model]
            # text_x = min(authsys_rate + 2.0, 102.0)
            text_x = authsys_rate
            ha = "left" if name != "Grok 4.1" else "right"
            va = "bottom" if name != "Grok 4.1" else "top"
            # ha = "left" if authsys_rate < 99 else "right"
            # if ha == "right":
            #     text_x = authsys_rate - 2.0
            ax.annotate(
                labels.loc[model],
                xy=(authsys_rate, ys[1]),
                xytext=(text_x, ys[1]),
                ha=ha,
                va=va,
                fontsize=5.3,
                rotation=30,
                color="#333333",
                # arrowprops={
                #     "arrowstyle": "-",
                #     "color": "#777777",
                #     "linewidth": 0.35,
                # "shrinkA": 1,
                # "shrinkB": 1,
                # },
            )

    ax.set_yticks([0, 1], ["Default\nSystem Prompt", "Authoritarian\nSystem Prompt"])
    ax.set_ylim(-0.45, 1.45)
    # ax.set_ylabel("System Prompt")
    ax.set_xlim(-2, 108)
    ax.set_xlabel("Adjusted Authoritarian Response Rate (%)")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    handles = [
        # Patch(facecolor="#d8dee9", edgecolor="#555555", label="Across-model IQR"),
        Line2D([0], [0], color="#9aa0a6", linewidth=0.8, label="Same model"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    save_figure(fig, "issuebench_authsys_box_scatter")
    plt.close(fig)


def plot_delta_forest(delta: pd.DataFrame) -> None:
    y = np.arange(len(delta))
    delta_pp = np.asarray(delta["delta"].to_numpy(), dtype=float) * 100
    low_pp = np.asarray(delta["ci_low"].to_numpy(), dtype=float) * 100
    high_pp = np.asarray(delta["ci_high"].to_numpy(), dtype=float) * 100
    xerr = np.vstack([delta_pp - low_pp, high_pp - delta_pp])
    colors = np.where(delta["ci_excludes_zero"], "#b45f38", "#6f7782")

    fig, ax = plt.subplots(figsize=(3.35, 4.55))
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.75)

    for i, color in enumerate(colors):
        ax.errorbar(
            delta_pp[i],
            y[i],
            xerr=xerr[:, [i]],
            fmt="o",
            markersize=3.2,
            color=color,
            ecolor=color,
            elinewidth=0.85,
            capsize=1.7,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3,
        )

    ax.set_yticks(y, delta["model_label"])
    ax.set_xlabel("Change in adjusted authoritarian response rate (pp)")
    ax.set_xlim(-8, 104)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker="o", color="#b45f38", label="95% CI excludes 0", linestyle=""),
        Line2D([0], [0], marker="o", color="#6f7782", label="95% CI overlaps 0", linestyle=""),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    save_figure(fig, "issuebench_authsys_delta_forest")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    points = load_point_estimates()
    delta = load_delta_estimates()
    plot_box_scatter(points)
    plot_delta_forest(delta)


if __name__ == "__main__":
    main()
