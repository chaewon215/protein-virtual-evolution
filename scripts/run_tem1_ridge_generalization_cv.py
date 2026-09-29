from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
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
    / "ridge_generalization"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


SPLITS = {
    "random": "fold_random_5",
    "modulo": "fold_modulo_5",
    "contiguous": "fold_contiguous_5",
}


ALPHA = 1.0


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print(
    "Samples:",
    len(df),
)


# ============================================================
# Load ESM-C cache
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
# Validate alignment
# ============================================================

assert (
    df["mutant"].tolist()
    == mut_cache["mutants"]
), "Dataset/cache mutant order mismatch."


positions = (
    df["position"]
    .to_numpy(
        dtype=np.int64
    )
)


cache_positions = (
    mut_cache["positions"]
    .numpy()
)


assert np.array_equal(
    positions,
    cache_positions,
), "Dataset/cache position mismatch."


print(
    "Dataset/cache alignment: OK"
)


# ============================================================
# Construct ESM-C feature
#
# Raw biological input:
#   WT sequence      : 286 aa
#   Mutant sequence  : 286 aa
#
# ESM-C:
#   WT residue embeddings    : [286, 960]
#   Mutant local embeddings  : [4996, 960]
#
# Feature construction:
#   WT local                 : [960]
#   Delta local              : [960]
#
#   concat(WT local, Delta local)
#       -> [1920]
#
# Entire supervised dataset:
#   X : [4996, 1920]
#   y : [4996]
#
# Ridge:
#   StandardScaler
#       [B, 1920] -> [B, 1920]
#
#   Ridge Regression
#       [B, 1920] -> [B]
#
# Final output:
#   predicted DMS_score
# ============================================================

wt_residue = (
    wt_cache[
        "residue_embeddings"
    ]
    .numpy()
    .astype(
        np.float64
    )
)


mutant_local = (
    mut_cache[
        "mutant_local"
    ]
    .numpy()
    .astype(
        np.float64
    )
)


wt_local = wt_residue[
    positions - 1
]


delta_local = (
    mutant_local
    - wt_local
)


X = np.concatenate(
    [
        wt_local,
        delta_local,
    ],
    axis=1,
)


y = (
    df["DMS_score"]
    .to_numpy(
        dtype=np.float64
    )
)


print()
print(
    "=== Feature Matrix ==="
)

print(
    "WT local    :",
    wt_local.shape,
)

print(
    "Delta local :",
    delta_local.shape,
)

print(
    "X           :",
    X.shape,
)

print(
    "y           :",
    y.shape,
)


assert X.shape == (
    len(df),
    1920,
)

assert np.isfinite(
    X
).all()

assert np.isfinite(
    y
).all()


# ============================================================
# Ridge model summary
# ============================================================

print()
print(
    "=== Ridge Architecture ==="
)

print(
    "Input            : [B, 1920]"
)

print(
    "StandardScaler   : [B, 1920] -> [B, 1920]"
)

print(
    "Ridge Regression : [B, 1920] -> [B]"
)

print(
    "Weight shape     : [1920]"
)

print(
    "Bias             : scalar"
)

print(
    "Output           : predicted DMS_score"
)

print(
    f"alpha            : {ALPHA}"
)

print(
    "solver           : lsqr"
)


# ============================================================
# Evaluate one split
# ============================================================

def evaluate_split(
    split_name,
    fold_column,
):

    print()
    print(
        "=" * 90
    )

    print(
        f"Split: {split_name}"
    )

    print(
        f"Fold column: {fold_column}"
    )

    print(
        "=" * 90
    )


    folds = (
        df[fold_column]
        .to_numpy(
            dtype=np.int64
        )
    )


    unique_folds = sorted(
        np.unique(
            folds
        ).tolist()
    )


    assert unique_folds == [
        0,
        1,
        2,
        3,
        4,
    ]


    oof_predictions = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )


    fold_results = []


    for test_fold in unique_folds:

        train_mask = (
            folds != test_fold
        )

        test_mask = (
            folds == test_fold
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


        # ====================================================
        # Pipeline prevents scaler leakage:
        #
        # scaler.fit() uses training fold only.
        # test fold is transformed using train statistics.
        # ====================================================

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
                "split":
                    split_name,

                "fold_column":
                    fold_column,

                "test_fold":
                    int(
                        test_fold
                    ),

                "n_train":
                    int(
                        train_mask.sum()
                    ),

                "n_test":
                    int(
                        test_mask.sum()
                    ),

                "spearman":
                    float(
                        spearman
                    ),

                "spearman_p":
                    float(
                        p_value
                    ),

                "rmse":
                    float(
                        rmse
                    ),

                "r2":
                    float(
                        r2
                    ),
            }
        )


        print()
        print(
            f"Test fold : {test_fold}"
        )

        print(
            f"Train     : {train_mask.sum()}"
        )

        print(
            f"Test      : {test_mask.sum()}"
        )

        print(
            f"Spearman  : {spearman:.4f}"
        )

        print(
            f"RMSE      : {rmse:.4f}"
        )

        print(
            f"R²        : {r2:.4f}"
        )


    # ========================================================
    # Validate OOF coverage
    # ========================================================

    assert np.isfinite(
        oof_predictions
    ).all()


    # ========================================================
    # Aggregate fold metrics
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )


    mean_spearman = (
        fold_df[
            "spearman"
        ]
        .mean()
    )

    std_spearman = (
        fold_df[
            "spearman"
        ]
        .std(
            ddof=1
        )
    )


    mean_rmse = (
        fold_df[
            "rmse"
        ]
        .mean()
    )

    std_rmse = (
        fold_df[
            "rmse"
        ]
        .std(
            ddof=1
        )
    )


    mean_r2 = (
        fold_df[
            "r2"
        ]
        .mean()
    )

    std_r2 = (
        fold_df[
            "r2"
        ]
        .std(
            ddof=1
        )
    )


    global_spearman, global_p = (
        spearmanr(
            y,
            oof_predictions,
        )
    )


    global_rmse = np.sqrt(
        mean_squared_error(
            y,
            oof_predictions,
        )
    )


    global_r2 = r2_score(
        y,
        oof_predictions,
    )


    print()
    print(
        "=" * 70
    )

    print(
        f"{split_name.upper()} Summary"
    )

    print(
        "=" * 70
    )


    print(
        "Spearman : "
        f"{mean_spearman:.4f} "
        f"± {std_spearman:.4f}"
    )


    print(
        "RMSE     : "
        f"{mean_rmse:.4f} "
        f"± {std_rmse:.4f}"
    )


    print(
        "R²       : "
        f"{mean_r2:.4f} "
        f"± {std_r2:.4f}"
    )


    print()
    print(
        "=== Global OOF ==="
    )


    print(
        "Spearman : "
        f"{global_spearman:.4f}"
    )


    print(
        "RMSE     : "
        f"{global_rmse:.4f}"
    )


    print(
        "R²       : "
        f"{global_r2:.4f}"
    )


    # ========================================================
    # Save fold results
    # ========================================================

    fold_df.to_csv(
        RESULT_ROOT
        / (
            f"{split_name}"
            f"_fold_metrics.csv"
        ),
        index=False,
    )


    # ========================================================
    # Save OOF predictions
    # ========================================================

    oof_df = df[
        [
            "mutant",
            "position",
            "wt_aa",
            "mut_aa",
            "DMS_score",
            fold_column,
        ]
    ].copy()


    oof_df[
        "prediction"
    ] = oof_predictions


    oof_df[
        "error"
    ] = (
        oof_predictions
        - y
    )


    oof_df[
        "absolute_error"
    ] = np.abs(
        oof_df[
            "error"
        ]
    )


    oof_df.to_csv(
        RESULT_ROOT
        / (
            f"{split_name}"
            f"_oof_predictions.csv"
        ),
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    return {

        "split":
            split_name,

        "fold_column":
            fold_column,

        "model":
            "Ridge",

        "alpha":
            ALPHA,

        "solver":
            "lsqr",

        "input_feature":
            "wt_local + delta_local",

        "input_dim":
            int(
                X.shape[1]
            ),

        "fold_mean_spearman":
            float(
                mean_spearman
            ),

        "fold_std_spearman":
            float(
                std_spearman
            ),

        "fold_mean_rmse":
            float(
                mean_rmse
            ),

        "fold_std_rmse":
            float(
                std_rmse
            ),

        "fold_mean_r2":
            float(
                mean_r2
            ),

        "fold_std_r2":
            float(
                std_r2
            ),

        "global_oof_spearman":
            float(
                global_spearman
            ),

        "global_oof_spearman_p":
            float(
                global_p
            ),

        "global_oof_rmse":
            float(
                global_rmse
            ),

        "global_oof_r2":
            float(
                global_r2
            ),
    }


# ============================================================
# Run all splits
# ============================================================

all_results = []


for split_name, fold_column in SPLITS.items():

    result = evaluate_split(
        split_name,
        fold_column,
    )

    all_results.append(
        result
    )


# ============================================================
# Save summary
# ============================================================

summary_df = pd.DataFrame(
    all_results
)


summary_df.to_csv(
    RESULT_ROOT
    / "generalization_summary.csv",
    index=False,
)


with open(
    RESULT_ROOT
    / "generalization_summary.json",
    "w",
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
    )


# ============================================================
# Final comparison
# ============================================================

print()
print(
    "=" * 100
)

print(
    "Ridge Generalization Summary"
)

print(
    "=" * 100
)


display_columns = [
    "split",
    "fold_mean_spearman",
    "fold_std_spearman",
    "fold_mean_rmse",
    "fold_std_rmse",
    "fold_mean_r2",
    "fold_std_r2",
    "global_oof_spearman",
    "global_oof_rmse",
    "global_oof_r2",
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
