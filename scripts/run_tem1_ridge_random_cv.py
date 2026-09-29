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
    / "ridge_random5"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

ALPHA = 1.0

FOLD_COLUMN = "fold_random_5"


# ============================================================
# Load dataset
# ============================================================

print("=== Loading data ===")

df = pd.read_csv(DATA_PATH)

print("Samples:", len(df))


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
# Validate sample alignment
# ============================================================

dataset_mutants = df["mutant"].tolist()
cache_mutants = mut_cache["mutants"]

assert dataset_mutants == cache_mutants, (
    "Dataset/cache mutant order mismatch."
)


dataset_positions = (
    df["position"]
    .to_numpy(dtype=np.int64)
)

cache_positions = (
    mut_cache["positions"]
    .numpy()
)

assert np.array_equal(
    dataset_positions,
    cache_positions,
), "Dataset/cache position mismatch."


print("Dataset/cache alignment: OK")


# ============================================================
# Convert cached tensors to NumPy
# ============================================================

wt_residue = (
    wt_cache["residue_embeddings"]
    .numpy()
)

wt_global = (
    wt_cache["global_embedding"]
    .numpy()
)

mutant_local = (
    mut_cache["mutant_local"]
    .numpy()
)

mutant_global = (
    mut_cache["mutant_global"]
    .numpy()
)


print()
print("WT residue    :", wt_residue.shape)
print("WT global     :", wt_global.shape)
print("Mutant local  :", mutant_local.shape)
print("Mutant global :", mutant_global.shape)


# ============================================================
# Build features
# ============================================================

# Protein positions are 1-based.
# NumPy indexes are 0-based.
wt_local = wt_residue[
    dataset_positions - 1
]


delta_local = (
    mutant_local
    - wt_local
)


delta_global = (
    mutant_global
    - wt_global[None, :]
)


# Main feature:
#
# [ WT local | Delta local | Delta global ]
#
X = np.concatenate(
    [
        wt_local,
        delta_local,
        delta_global,
    ],
    axis=1,
).astype(np.float64)


y = (
    df["DMS_score"]
    .to_numpy(dtype=np.float64)
)


print()
print("=== Feature Matrix ===")
print("WT local     :", wt_local.shape)
print("Delta local  :", delta_local.shape)
print("Delta global :", delta_global.shape)

print()
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# Feature validation
# ============================================================

assert X.shape == (
    len(df),
    2880,
)

assert y.shape == (
    len(df),
)

assert np.isfinite(X).all()
assert np.isfinite(y).all()


print("Feature validation: OK")


# ============================================================
# Fold validation
# ============================================================

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


print()
print("=== Fold Distribution ===")

for fold in unique_folds:

    n = np.sum(folds == fold)

    print(
        f"Fold {fold}: {n} samples"
    )


# ============================================================
# 5-fold CV
# ============================================================

oof_predictions = np.full(
    shape=len(df),
    fill_value=np.nan,
    dtype=np.float64,
)

fold_results = []


print()
print("=== Random 5-Fold Ridge CV ===")


for fold in unique_folds:

    train_mask = folds != fold
    test_mask = folds == fold


    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]


    print()
    print(
        f"Fold {fold}"
    )

    print(
        f"Train: {len(y_train)} | "
        f"Test: {len(y_test)}"
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # StandardScaler is inside Pipeline.
    #
    # Therefore scaler.fit() sees TRAIN data only.
    #
    # Test samples are transformed using parameters learned
    # exclusively from the training fold.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    spearman, p_value = spearmanr(
        y_test,
        predictions,
    )


    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)


    r2 = r2_score(
        y_test,
        predictions,
    )


    fold_result = {
        "fold": int(fold),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "spearman": float(spearman),
        "spearman_p": float(p_value),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


    fold_results.append(
        fold_result
    )


    print(
        f"Spearman : {spearman:.4f}"
    )

    print(
        f"RMSE     : {rmse:.4f}"
    )

    print(
        f"R²       : {r2:.4f}"
    )


# ============================================================
# Validate OOF predictions
# ============================================================

assert np.isfinite(
    oof_predictions
).all()

assert not np.isnan(
    oof_predictions
).any()


print()
print("OOF prediction coverage: OK")


# ============================================================
# Fold summary
# ============================================================

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


# ============================================================
# Global OOF metrics
# ============================================================

oof_spearman, oof_p = spearmanr(
    y,
    oof_predictions,
)

oof_mse = mean_squared_error(
    y,
    oof_predictions,
)

oof_rmse = np.sqrt(
    oof_mse
)

oof_r2 = r2_score(
    y,
    oof_predictions,
)


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 60)
print("Random 5-Fold Summary")
print("=" * 60)

print(
    "Spearman : "
    f"{mean_metrics['spearman']:.4f} "
    f"± {std_metrics['spearman']:.4f}"
)

print(
    "RMSE     : "
    f"{mean_metrics['rmse']:.4f} "
    f"± {std_metrics['rmse']:.4f}"
)

print(
    "R²       : "
    f"{mean_metrics['r2']:.4f} "
    f"± {std_metrics['r2']:.4f}"
)


print()
print("=== Global OOF ===")

print(
    f"Spearman : {oof_spearman:.4f}"
)

print(
    f"RMSE     : {oof_rmse:.4f}"
)

print(
    f"R²       : {oof_r2:.4f}"
)


# ============================================================
# Save fold metrics
# ============================================================

fold_metrics_path = (
    RESULT_ROOT
    / "fold_metrics.csv"
)

fold_df.to_csv(
    fold_metrics_path,
    index=False,
)


# ============================================================
# Save OOF predictions
# ============================================================

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

oof_df["absolute_error"] = (
    np.abs(
        oof_df["error"]
    )
)


oof_path = (
    RESULT_ROOT
    / "oof_predictions.csv"
)

oof_df.to_csv(
    oof_path,
    index=False,
)


# ============================================================
# Save summary
# ============================================================

summary = {
    "model": "Ridge",
    "alpha": ALPHA,
    "feature_set": [
        "wt_local",
        "delta_local",
        "delta_global",
    ],
    "feature_dim": int(X.shape[1]),
    "split": FOLD_COLUMN,

    "fold_mean": {
        "spearman": float(
            mean_metrics["spearman"]
        ),
        "rmse": float(
            mean_metrics["rmse"]
        ),
        "r2": float(
            mean_metrics["r2"]
        ),
    },

    "fold_std": {
        "spearman": float(
            std_metrics["spearman"]
        ),
        "rmse": float(
            std_metrics["rmse"]
        ),
        "r2": float(
            std_metrics["r2"]
        ),
    },

    "global_oof": {
        "spearman": float(
            oof_spearman
        ),
        "spearman_p": float(
            oof_p
        ),
        "rmse": float(
            oof_rmse
        ),
        "r2": float(
            oof_r2
        ),
    },
}


summary_path = (
    RESULT_ROOT
    / "summary.json"
)


with open(
    summary_path,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=== Saved ===")
print(fold_metrics_path)
print(oof_path)
print(summary_path) 
