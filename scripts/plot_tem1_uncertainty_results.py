from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "tem1" / "deep_ensemble_uncertainty"
FIGURE_ROOT = RESULT_ROOT / "figures"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["random", "modulo", "contiguous"]

def plot_scatter(split):
    df = pd.read_csv(RESULT_ROOT / f"{split}_oof_predictions.csv")
    x = df["ensemble_std"].to_numpy(float)
    y = df["absolute_error"].to_numpy(float)
    rho, p = spearmanr(x, y)

    trend_df = pd.DataFrame({"uncertainty": x, "absolute_error": y})
    trend_df["bin"] = pd.qcut(
        trend_df["uncertainty"].rank(method="first"),
        q=20,
        duplicates="drop"
    )
    trend = (
        trend_df.groupby("bin", observed=True)
        .agg(
            mean_uncertainty=("uncertainty", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=10, alpha=0.30)
    ax.plot(
        trend["mean_uncertainty"],
        trend["mean_absolute_error"],
        marker="o",
        linewidth=1,
        label="Binned mean trend",
        color='black'
    )
    ax.set_xlabel("Ensemble uncertainty (prediction std)")
    ax.set_ylabel("Absolute prediction error")
    ax.set_title(f"{split.capitalize()}: Uncertainty vs Absolute Error\nSpearman rho = {rho:.3f}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out = FIGURE_ROOT / f"{split}_uncertainty_vs_error.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return split, rho, p, out

def plot_quintiles():
    order = ["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))

    for split in SPLITS:
        df = pd.read_csv(RESULT_ROOT / f"{split}_uncertainty_quintiles.csv")
        sub = df.set_index("uncertainty_quintile").reindex(order)
        ax.plot(
            x,
            sub["mean_absolute_error"].to_numpy(),
            marker="o",
            linewidth=2,
            label=split.capitalize(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nLowest", "Q2", "Q3", "Q4", "Q5\nHighest"])
    ax.set_xlabel("Uncertainty quintile")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Prediction Error by Uncertainty Quintile")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out = FIGURE_ROOT / "uncertainty_quintile_vs_mae.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_risk_coverage():
    fig, ax = plt.subplots(figsize=(8, 5))

    for split in SPLITS:
        df = pd.read_csv(RESULT_ROOT / f"{split}_risk_coverage.csv").sort_values("coverage")
        ax.plot(
            df["coverage"] * 100,
            df["rmse"],
            marker="o",
            linewidth=2,
            label=split.capitalize(),
        )

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("RMSE")
    ax.set_title("Risk-Coverage Curve")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out = FIGURE_ROOT / "risk_coverage_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_quality_summary():
    df = pd.read_csv(RESULT_ROOT / "uncertainty_summary.csv")
    df = df.set_index("split").reindex(SPLITS).reset_index()

    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        df["uncertainty_error_spearman"],
        width,
        label="Uncertainty-error Spearman",
    )
    ax.bar(
        x + width / 2,
        df["high_error_auroc"],
        width,
        label="High-error AUROC",
    )
    ax.axhline(0.5, linestyle="--", linewidth=1, label="AUROC random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(["Random", "Modulo", "Contiguous"])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Metric value")
    ax.set_title("Deep Ensemble Uncertainty Quality")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out = FIGURE_ROOT / "uncertainty_quality_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

if __name__ == "__main__":
    print("=== Uncertainty vs Error ===")
    for split in SPLITS:
        split, rho, p, out = plot_scatter(split)
        print(f"{split:10s} | Spearman={rho:.4f} | p={p:.3e}")
        print(f"  saved: {out}")

    print()
    print("saved:", plot_quintiles())
    print("saved:", plot_risk_coverage())
    print("saved:", plot_quality_summary())
