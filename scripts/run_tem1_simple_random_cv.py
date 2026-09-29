from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


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

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "tem1"
    / "simple_random5"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_COLUMN = "fold_random_5"

ALPHA = 1.0


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Samples:", len(df))


# ============================================================
# Build categorical mutation features
# ============================================================

feature_df = df[
    [
        "position",
        "wt_aa",
        "mut_aa",
    ]
].copy()


# Treat position as categorical, not continuous.
feature_df["position"] = (
    feature_df["position"]
    .astype(str)
)


encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
    dtype=np.float64,
)


X = encoder.fit_transform(
    feature_df
)


y = (
    df["DMS_score"]
    .to_numpy(dtype=np.float64)
)


print()
print("=== Simple Feature Matrix ===")
print("X:", X.shape)
print("y:", y.shape)


assert len(X) == len(df)
assert np.isfinite(X).all()
assert np.isfinite(y).all()


# ============================================================
# Fold information
# ============================================================

folds = (
    df[FOLD_COLUMN]
    .to_numpy(dtype=np.int64)
)

unique_folds = sorted(
    np.unique(folds).tolist()
)

assert unique_folds == [0, 1, 2, 3, 4]


# ============================================================
# Cross validation
# ============================================================

oof_predictions = np.full(
    len(df),
    np.nan,
    dtype=np.float64,
)

fold_results = []


print()
print("=== Simple Mutation Random 5-Fold CV ===")


for fold in unique_folds:

    train_mask = folds != fold
    test_mask = folds == fold

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]


    model = Ridge(
        alpha=ALPHA,
        solver="lsqr",
        tol=1e-6,
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
        "rmse": float(rmse),
        "r2": float(r2),
    }


    fold_results.append(
        fold_result
    )


    print()
    print(f"Fold {fold}")
    print(
        f"Train: {len(y_train)} | "
        f"Test: {len(y_test)}"
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
# Validate OOF
# ============================================================

assert np.isfinite(
    oof_predictions
).all()


# ============================================================
# Summary
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


oof_spearman, oof_p = spearmanr(
    y,
    oof_predictions,
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


print()
print("=" * 60)
print("Simple Mutation Random 5-Fold Summary")
print("=" * 60)

print(
    f"Spearman : "
    f"{mean_metrics['spearman']:.4f} "
    f"± {std_metrics['spearman']:.4f}"
)

print(
    f"RMSE     : "
    f"{mean_metrics['rmse']:.4f} "
    f"± {std_metrics['rmse']:.4f}"
)

print(
    f"R²       : "
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
# Save results
# ============================================================

fold_df.to_csv(
    RESULT_ROOT / "fold_metrics.csv",
    index=False,
)


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


oof_df.to_csv(
    RESULT_ROOT
    / "oof_predictions.csv",
    index=False,
)


summary = {
    "model": "Ridge",
    "alpha": ALPHA,

    "features": [
        "position_onehot",
        "wt_aa_onehot",
        "mut_aa_onehot",
    ],

    "feature_dim": int(
        X.shape[1]
    ),

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


with open(
    RESULT_ROOT
    / "summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("Saved to:")
print(RESULT_ROOT)
