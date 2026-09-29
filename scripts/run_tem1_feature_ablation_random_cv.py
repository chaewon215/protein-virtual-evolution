from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tem1_single_mutants.csv"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esmc300m"
)

WT_CACHE_PATH = (
    CACHE_ROOT
    / "wt_embeddings.pt"
)

MUT_CACHE_PATH = (
    CACHE_ROOT
    / "tem1_mutant_embeddings.pt"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "tem1"
    / "feature_ablation_random5"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


ALPHA = 1.0
FOLD_COLUMN = "fold_random_5"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Samples:", len(df))


# ============================================================
# Load embeddings
# ============================================================

wt_cache = torch.load(
    WT_CACHE_PATH,
    map_location="cpu",
    weights_only=False,
)

mut_cache = torch.load(
    MUT_CACHE_PATH,
    map_location="cpu",
    weights_only=False,
)


# ============================================================
# Alignment validation
# ============================================================

assert (
    df["mutant"].tolist()
    == mut_cache["mutants"]
)


positions = (
    df["position"]
    .to_numpy(dtype=np.int64)
)


cache_positions = (
    mut_cache["positions"]
    .numpy()
)


assert np.array_equal(
    positions,
    cache_positions,
)


print("Dataset/cache alignment: OK")


# ============================================================
# Convert tensors
# ============================================================

wt_residue = (
    wt_cache["residue_embeddings"]
    .numpy()
    .astype(np.float64)
)

wt_global = (
    wt_cache["global_embedding"]
    .numpy()
    .astype(np.float64)
)

mutant_local = (
    mut_cache["mutant_local"]
    .numpy()
    .astype(np.float64)
)

mutant_global = (
    mut_cache["mutant_global"]
    .numpy()
    .astype(np.float64)
)


# ============================================================
# Construct basic features
# ============================================================

# position is biological 1-based
# numpy index is 0-based
wt_local = wt_residue[
    positions - 1
]


delta_local = (
    mutant_local
    - wt_local
)


delta_global = (
    mutant_global
    - wt_global[None, :]
)


print()
print("=== Base feature shapes ===")

print(
    "WT local     :",
    wt_local.shape,
)

print(
    "Mutant local :",
    mutant_local.shape,
)

print(
    "Delta local  :",
    delta_local.shape,
)

print(
    "Delta global :",
    delta_global.shape,
)


# ============================================================
# Define feature sets
# ============================================================

FEATURE_SETS = {

    "wt_local":
        wt_local,

    "mut_local":
        mutant_local,

    "delta_local":
        delta_local,

    "wt_local_delta_local":
        np.concatenate(
            [
                wt_local,
                delta_local,
            ],
            axis=1,
        ),

    "full":
        np.concatenate(
            [
                wt_local,
                delta_local,
                delta_global,
            ],
            axis=1,
        ),
}


# ============================================================
# Target / folds
# ============================================================

y = (
    df["DMS_score"]
    .to_numpy(dtype=np.float64)
)


folds = (
    df[FOLD_COLUMN]
    .to_numpy(dtype=np.int64)
)


unique_folds = sorted(
    np.unique(folds).tolist()
)


assert unique_folds == [
    0,
    1,
    2,
    3,
    4,
]


# ============================================================
# Evaluation function
# ============================================================

def evaluate_feature_set(
    name,
    X,
):

    print()
    print("=" * 70)
    print(
        f"Feature set: {name}"
    )
    print(
        f"Input shape: {X.shape}"
    )
    print("=" * 70)


    assert len(X) == len(df)
    assert np.isfinite(X).all()


    oof_predictions = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )


    fold_results = []


    for fold in unique_folds:

        train_mask = (
            folds != fold
        )

        test_mask = (
            folds == fold
        )


        X_train = X[
            train_mask
        ]

        X_test = X[
            test_mask
        ]

        y_train = y[
            train_mask
        ]

        y_test = y[
            test_mask
        ]


        model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=ALPHA,
                        solver="lsqr",
                        tol=1e-6,
                    ),
                ),
            ]
        )


        model.fit(
            X_train,
            y_train,
        )


        predictions = model.predict(
            X_test
        )


        oof_predictions[
            test_mask
        ] = predictions


        spearman, p_value = (
            spearmanr(
                y_test,
                predictions,
            )
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )


        r2 = r2_score(
            y_test,
            predictions,
        )


        fold_results.append(
            {
                "feature_set": name,
                "fold": int(fold),

                "n_train": int(
                    len(y_train)
                ),

                "n_test": int(
                    len(y_test)
                ),

                "spearman": float(
                    spearman
                ),

                "spearman_p": float(
                    p_value
                ),

                "rmse": float(
                    rmse
                ),

                "r2": float(
                    r2
                ),
            }
        )


        print(
            f"Fold {fold} | "
            f"Spearman={spearman:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R²={r2:.4f}"
        )


    # --------------------------------------------------------
    # Validate OOF
    # --------------------------------------------------------

    assert np.isfinite(
        oof_predictions
    ).all()


    fold_df = pd.DataFrame(
        fold_results
    )


    mean_metrics = (
        fold_df[
            [
                "spearman",
                "rmse",
                "r2",
            ]
        ]
        .mean()
    )


    std_metrics = (
        fold_df[
            [
                "spearman",
                "rmse",
                "r2",
            ]
        ]
        .std(ddof=1)
    )


    oof_spearman, oof_p = (
        spearmanr(
            y,
            oof_predictions,
        )
    )


    oof_rmse = np.sqrt(
        mean_squared_error(
            y,
            oof_predictions,
        )
    )


    oof_r2 = r2_score(
        y,
        oof_predictions,
    )


    summary = {

        "feature_set": name,

        "feature_dim": int(
            X.shape[1]
        ),

        "fold_mean_spearman": float(
            mean_metrics["spearman"]
        ),

        "fold_std_spearman": float(
            std_metrics["spearman"]
        ),

        "fold_mean_rmse": float(
            mean_metrics["rmse"]
        ),

        "fold_std_rmse": float(
            std_metrics["rmse"]
        ),

        "fold_mean_r2": float(
            mean_metrics["r2"]
        ),

        "fold_std_r2": float(
            std_metrics["r2"]
        ),

        "global_oof_spearman": float(
            oof_spearman
        ),

        "global_oof_spearman_p": float(
            oof_p
        ),

        "global_oof_rmse": float(
            oof_rmse
        ),

        "global_oof_r2": float(
            oof_r2
        ),
    }


    print()
    print(
        f"Summary | "
        f"Spearman="
        f"{summary['fold_mean_spearman']:.4f} "
        f"± "
        f"{summary['fold_std_spearman']:.4f} | "
        f"RMSE="
        f"{summary['fold_mean_rmse']:.4f} "
        f"± "
        f"{summary['fold_std_rmse']:.4f} | "
        f"R²="
        f"{summary['fold_mean_r2']:.4f} "
        f"± "
        f"{summary['fold_std_r2']:.4f}"
    )


    return (
        fold_df,
        summary,
        oof_predictions,
    )


# ============================================================
# Run all feature sets
# ============================================================

all_fold_results = []

all_summaries = []


for name, X in FEATURE_SETS.items():

    (
        fold_df,
        summary,
        oof_predictions,
    ) = evaluate_feature_set(
        name,
        X,
    )


    all_fold_results.append(
        fold_df
    )

    all_summaries.append(
        summary
    )


    # --------------------------------------------------------
    # Save OOF predictions for each feature set
    # --------------------------------------------------------

    oof_df = df[
        [
            "mutant",
            "position",
            "wt_aa",
            "mut_aa",
            "DMS_score",
            FOLD_COLUMN,
        ]
    ].copy()


    oof_df["prediction"] = (
        oof_predictions
    )


    oof_df["error"] = (
        oof_df["prediction"]
        - oof_df["DMS_score"]
    )


    oof_df[
        "absolute_error"
    ] = np.abs(
        oof_df["error"]
    )


    oof_df.to_csv(
        RESULT_ROOT
        / f"oof_{name}.csv",
        index=False,
    )


# ============================================================
# Save combined results
# ============================================================

all_fold_df = pd.concat(
    all_fold_results,
    ignore_index=True,
)


summary_df = pd.DataFrame(
    all_summaries
)


all_fold_df.to_csv(
    RESULT_ROOT
    / "fold_metrics.csv",
    index=False,
)


summary_df.to_csv(
    RESULT_ROOT
    / "summary.csv",
    index=False,
)


with open(
    RESULT_ROOT
    / "summary.json",
    "w",
) as f:

    json.dump(
        all_summaries,
        f,
        indent=2,
    )


# ============================================================
# Final comparison
# ============================================================

print()
print("=" * 90)
print("Feature Ablation Summary")
print("=" * 90)


display_columns = [
    "feature_set",
    "feature_dim",
    "fold_mean_spearman",
    "fold_std_spearman",
    "fold_mean_rmse",
    "fold_mean_r2",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print()
print(
    "Saved to:",
    RESULT_ROOT,
)
