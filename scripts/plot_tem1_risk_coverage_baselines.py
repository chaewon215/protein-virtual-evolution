from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "tem1"
    / "deep_ensemble_uncertainty"
)

OUTPUT_ROOT = (
    RESULT_ROOT
    / "risk_coverage_baselines"
)

FIGURE_ROOT = (
    RESULT_ROOT
    / "figures"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


SPLITS = [
    "random",
    "modulo",
    "contiguous",
]


COVERAGES = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]


# Number of repeated random subset selections
RANDOM_REPEATS = 1000

SEED = 42


# ============================================================
# Metric helpers
# ============================================================

def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    return float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )


# ============================================================
# Evaluate one selected subset
# ============================================================

def evaluate_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
) -> dict:

    return {
        "rmse":
            rmse(
                y_true[
                    indices
                ],
                y_pred[
                    indices
                ],
            ),

        "mae":
            mae(
                y_true[
                    indices
                ],
                y_pred[
                    indices
                ],
            ),
    }


# ============================================================
# Risk-Coverage evaluation
#
# Three selection strategies:
#
# 1. Uncertainty-based
#    Select samples with the lowest ensemble_std.
#
#    Available at inference time:
#       YES
#
# 2. Random
#    Randomly choose the same number of samples.
#    Repeat RANDOM_REPEATS times.
#
#    Purpose:
#       Baseline for "selection without useful information."
#
# 3. Oracle
#    Select samples with the lowest TRUE absolute error.
#
#    Available at inference time:
#       NO
#
#    Purpose:
#       Retrospective lower bound / ideal ranking reference.
#
# IMPORTANT:
# Oracle uses DMS_score and therefore must NEVER be interpreted
# as a deployable candidate-selection strategy.
# ============================================================

def evaluate_split(
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    path = (
        RESULT_ROOT
        / f"{split_name}_oof_predictions.csv"
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Missing OOF file: {path}"
        )


    df = pd.read_csv(
        path
    )


    required_columns = {
        "DMS_score",
        "ensemble_mean",
        "ensemble_std",
    }


    missing = (
        required_columns
        - set(
            df.columns
        )
    )


    if missing:

        raise ValueError(
            f"{path} missing columns: {missing}"
        )


    y_true = (
        df[
            "DMS_score"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    y_pred = (
        df[
            "ensemble_mean"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    uncertainty = (
        df[
            "ensemble_std"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    absolute_error = np.abs(
        y_pred
        - y_true
    )


    n_total = len(
        df
    )


    # --------------------------------------------
    # Fixed rankings
    # --------------------------------------------

    # Deployable ranking:
    # low uncertainty -> high uncertainty
    uncertainty_order = np.argsort(
        uncertainty
    )


    # Non-deployable retrospective ideal:
    # low true error -> high true error
    oracle_order = np.argsort(
        absolute_error
    )


    rng = np.random.default_rng(
        SEED
    )


    summary_rows = []

    random_repeat_rows = []


    for coverage in COVERAGES:

        n_keep = max(
            1,
            int(
                np.floor(
                    n_total
                    * coverage
                )
            ),
        )


        # ====================================================
        # 1. Uncertainty-based selection
        # ====================================================

        uncertainty_indices = (
            uncertainty_order[
                :n_keep
            ]
        )


        uncertainty_metrics = (
            evaluate_indices(
                y_true,
                y_pred,
                uncertainty_indices,
            )
        )


        # ====================================================
        # 2. Oracle selection
        # ====================================================

        oracle_indices = (
            oracle_order[
                :n_keep
            ]
        )


        oracle_metrics = (
            evaluate_indices(
                y_true,
                y_pred,
                oracle_indices,
            )
        )


        # ====================================================
        # 3. Random selection
        #
        # At coverage = 100%, every repetition is identical.
        # ====================================================

        random_rmses = []
        random_maes = []


        if n_keep == n_total:

            full_indices = np.arange(
                n_total
            )


            full_metrics = evaluate_indices(
                y_true,
                y_pred,
                full_indices,
            )


            random_rmses = np.full(
                RANDOM_REPEATS,
                full_metrics[
                    "rmse"
                ],
                dtype=np.float64,
            )


            random_maes = np.full(
                RANDOM_REPEATS,
                full_metrics[
                    "mae"
                ],
                dtype=np.float64,
            )


        else:

            for repeat in range(
                RANDOM_REPEATS
            ):

                indices = rng.choice(
                    n_total,
                    size=n_keep,
                    replace=False,
                )


                metrics = evaluate_indices(
                    y_true,
                    y_pred,
                    indices,
                )


                random_rmses.append(
                    metrics[
                        "rmse"
                    ]
                )


                random_maes.append(
                    metrics[
                        "mae"
                    ]
                )


        random_rmses = np.asarray(
            random_rmses,
            dtype=np.float64,
        )


        random_maes = np.asarray(
            random_maes,
            dtype=np.float64,
        )


        # Store all random repetitions for auditability.
        for repeat in range(
            RANDOM_REPEATS
        ):

            random_repeat_rows.append(
                {
                    "split":
                        split_name,

                    "coverage":
                        float(
                            coverage
                        ),

                    "n_samples":
                        int(
                            n_keep
                        ),

                    "repeat":
                        int(
                            repeat
                        ),

                    "rmse":
                        float(
                            random_rmses[
                                repeat
                            ]
                        ),

                    "mae":
                        float(
                            random_maes[
                                repeat
                            ]
                        ),
                }
            )


        # ====================================================
        # Random baseline distribution summary
        # ====================================================

        random_rmse_mean = float(
            random_rmses.mean()
        )

        random_rmse_std = float(
            random_rmses.std(
                ddof=1
            )
        )

        random_rmse_ci_low = float(
            np.quantile(
                random_rmses,
                0.025,
            )
        )

        random_rmse_ci_high = float(
            np.quantile(
                random_rmses,
                0.975,
            )
        )


        random_mae_mean = float(
            random_maes.mean()
        )

        random_mae_std = float(
            random_maes.std(
                ddof=1
            )
        )

        random_mae_ci_low = float(
            np.quantile(
                random_maes,
                0.025,
            )
        )

        random_mae_ci_high = float(
            np.quantile(
                random_maes,
                0.975,
            )
        )


        # ====================================================
        # Relative improvement vs random baseline
        # ====================================================

        rmse_reduction_vs_random = (
            (
                random_rmse_mean
                - uncertainty_metrics[
                    "rmse"
                ]
            )
            / random_rmse_mean
        )


        mae_reduction_vs_random = (
            (
                random_mae_mean
                - uncertainty_metrics[
                    "mae"
                ]
            )
            / random_mae_mean
        )


        # Percentile of uncertainty strategy within random
        # distribution.
        #
        # Small percentile is good:
        # very few random selections achieve this low an RMSE.
        uncertainty_random_rmse_percentile = float(
            np.mean(
                random_rmses
                <= uncertainty_metrics[
                    "rmse"
                ]
            )
        )


        summary_rows.append(
            {
                "split":
                    split_name,

                "coverage":
                    float(
                        coverage
                    ),

                "n_samples":
                    int(
                        n_keep
                    ),

                # ----------------------------
                # Uncertainty selection
                # ----------------------------
                "uncertainty_rmse":
                    float(
                        uncertainty_metrics[
                            "rmse"
                        ]
                    ),

                "uncertainty_mae":
                    float(
                        uncertainty_metrics[
                            "mae"
                        ]
                    ),

                # ----------------------------
                # Random selection
                # ----------------------------
                "random_rmse_mean":
                    random_rmse_mean,

                "random_rmse_std":
                    random_rmse_std,

                "random_rmse_ci_low":
                    random_rmse_ci_low,

                "random_rmse_ci_high":
                    random_rmse_ci_high,

                "random_mae_mean":
                    random_mae_mean,

                "random_mae_std":
                    random_mae_std,

                "random_mae_ci_low":
                    random_mae_ci_low,

                "random_mae_ci_high":
                    random_mae_ci_high,

                # ----------------------------
                # Oracle
                # ----------------------------
                "oracle_rmse":
                    float(
                        oracle_metrics[
                            "rmse"
                        ]
                    ),

                "oracle_mae":
                    float(
                        oracle_metrics[
                            "mae"
                        ]
                    ),

                # ----------------------------
                # Utility metrics
                # ----------------------------
                "uncertainty_rmse_reduction_vs_random":
                    float(
                        rmse_reduction_vs_random
                    ),

                "uncertainty_mae_reduction_vs_random":
                    float(
                        mae_reduction_vs_random
                    ),

                "uncertainty_rmse_random_percentile":
                    uncertainty_random_rmse_percentile,
            }
        )


    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.DataFrame(
            random_repeat_rows
        ),
    )


# ============================================================
# Plot one split
#
# Curves:
#   Random       = mean random selection risk
#   Uncertainty  = deployable uncertainty-based selection
#   Oracle       = retrospective lower bound
#
# Random shaded area:
#   empirical 95% interval from repeated random selection
# ============================================================

def plot_split(
    split_name: str,
    summary_df: pd.DataFrame,
):

    sub = (
        summary_df[
            summary_df[
                "split"
            ]
            == split_name
        ]
        .sort_values(
            "coverage"
        )
    )


    x = (
        sub[
            "coverage"
        ]
        .to_numpy()
        * 100.0
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.plot(
        x,
        sub[
            "random_rmse_mean"
        ],
        marker="o",
        linewidth=2,
        label="Random selection",
    )


    ax.fill_between(
        x,
        sub[
            "random_rmse_ci_low"
        ],
        sub[
            "random_rmse_ci_high"
        ],
        alpha=0.15,
        label="Random 95% interval",
    )


    ax.plot(
        x,
        sub[
            "uncertainty_rmse"
        ],
        marker="o",
        linewidth=2,
        label="Uncertainty-based",
    )


    ax.plot(
        x,
        sub[
            "oracle_rmse"
        ],
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Oracle (uses true error)",
    )


    ax.set_xlabel(
        "Coverage (%)"
    )

    ax.set_ylabel(
        "RMSE"
    )

    ax.set_title(
        f"{split_name.capitalize()}: "
        f"Risk-Coverage with Selection Baselines"
    )


    ax.grid(
        alpha=0.2
    )


    ax.legend()


    fig.tight_layout()


    output_path = (
        FIGURE_ROOT
        / (
            f"{split_name}"
            f"_risk_coverage_baselines.png"
        )
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    return output_path


# ============================================================
# Main
# ============================================================

all_summary = []

all_random_repeats = []


for split_name in SPLITS:

    print()
    print(
        "=" * 90
    )

    print(
        f"Split: {split_name}"
    )

    print(
        "=" * 90
    )


    (
        split_summary,
        split_random_repeats,
    ) = evaluate_split(
        split_name
    )


    all_summary.append(
        split_summary
    )


    all_random_repeats.append(
        split_random_repeats
    )


summary_df = pd.concat(
    all_summary,
    ignore_index=True,
)


random_repeats_df = pd.concat(
    all_random_repeats,
    ignore_index=True,
)


# ============================================================
# Save tables
# ============================================================

summary_path = (
    OUTPUT_ROOT
    / "risk_coverage_baseline_summary.csv"
)


random_repeats_path = (
    OUTPUT_ROOT
    / "random_selection_repeats.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
)


random_repeats_df.to_csv(
    random_repeats_path,
    index=False,
)


# ============================================================
# Save JSON summary
# ============================================================

json_records = (
    summary_df
    .to_dict(
        orient="records"
    )
)


json_path = (
    OUTPUT_ROOT
    / "risk_coverage_baseline_summary.json"
)


with open(
    json_path,
    "w",
) as f:

    json.dump(
        json_records,
        f,
        indent=2,
    )


# ============================================================
# Plot each split separately
# ============================================================

figure_paths = []


for split_name in SPLITS:

    figure_paths.append(
        plot_split(
            split_name,
            summary_df,
        )
    )


# ============================================================
# Console summary
# ============================================================

display_columns = [
    "split",
    "coverage",
    "n_samples",
    "uncertainty_rmse",
    "random_rmse_mean",
    "random_rmse_ci_low",
    "random_rmse_ci_high",
    "oracle_rmse",
    "uncertainty_rmse_reduction_vs_random",
    "uncertainty_rmse_random_percentile",
]


print()
print(
    "=" * 120
)

print(
    "Risk-Coverage Baseline Summary"
)

print(
    "=" * 120
)


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print()
print(
    "Saved:"
)

print(
    summary_path
)

print(
    random_repeats_path
)

print(
    json_path
)


for path in figure_paths:

    print(
        path
    )


# ============================================================
# Sanity checks
#
# At 100% coverage:
#   uncertainty == random == oracle
# because all samples are included.
# ============================================================

for split_name in SPLITS:

    row = summary_df[
        (
            summary_df[
                "split"
            ]
            == split_name
        )
        & (
            summary_df[
                "coverage"
            ]
            == 1.0
        )
    ].iloc[0]


    assert np.isclose(
        row[
            "uncertainty_rmse"
        ],
        row[
            "random_rmse_mean"
        ],
        atol=1e-10,
    )


    assert np.isclose(
        row[
            "uncertainty_rmse"
        ],
        row[
            "oracle_rmse"
        ],
        atol=1e-10,
    )


print()
print(
    "100% coverage sanity checks: OK"
)
