from pathlib import Path
import json
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SINGLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_single_mutants.csv"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esmc300m_gfp"
    / "gfp_variant_features.pt"
)

FEATURE_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esmc300m_gfp"
    / "gfp_variant_feature_metadata.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "single_mutant_cv"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_METRICS_PATH = (
    OUTPUT_ROOT
    / "fold_metrics.csv"
)

OOF_PATH = (
    OUTPUT_ROOT
    / "oof_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


SEED = 42
N_SPLITS = 5

RIDGE_ALPHA = 1.0

MLP_HIDDEN_1 = 512
MLP_HIDDEN_2 = 128
DROPOUT = 0.2

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 300
PATIENCE = 20
INNER_VAL_FRACTION = 0.15

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):

    rho = spearmanr(
        y_true,
        y_pred,
    ).statistic

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "spearman": float(rho),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


# ============================================================
# Model
#
# Input:
#   GFP feature [B, 1920]
#
# Architecture:
#   1920 -> 512 -> 128 -> 1
#
# Output:
#   predicted continuous DMS_score [B, 1]
# ============================================================

class FitnessMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                1920,
                MLP_HIDDEN_1,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                MLP_HIDDEN_1,
                MLP_HIDDEN_2,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                MLP_HIDDEN_2,
                1,
            ),
        )


    def forward(
        self,
        x,
    ):

        return (
            self.net(
                x
            )
            .squeeze(
                -1
            )
        )


# ============================================================
# Training utilities
# ============================================================

def make_loader(
    X,
    y,
    batch_size,
    shuffle,
):

    dataset = TensorDataset(
        torch.from_numpy(
            X.astype(
                np.float32
            )
        ),
        torch.from_numpy(
            y.astype(
                np.float32
            )
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:

        xb = xb.to(
            DEVICE,
            non_blocking=True,
        )

        yb = yb.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            xb
        )

        loss = criterion(
            pred,
            yb,
        )

        loss.backward()

        optimizer.step()

        batch_n = len(
            xb
        )

        total_loss += (
            loss.item()
            * batch_n
        )

        total_n += batch_n

    return (
        total_loss
        / total_n
    )


@torch.inference_mode()
def evaluate_loss(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:

        xb = xb.to(
            DEVICE,
            non_blocking=True,
        )

        yb = yb.to(
            DEVICE,
            non_blocking=True,
        )

        pred = model(
            xb
        )

        loss = criterion(
            pred,
            yb,
        )

        batch_n = len(
            xb
        )

        total_loss += (
            loss.item()
            * batch_n
        )

        total_n += batch_n

    return (
        total_loss
        / total_n
    )


@torch.inference_mode()
def predict_mlp(
    model,
    X,
):

    model.eval()

    X_tensor = torch.from_numpy(
        X.astype(
            np.float32
        )
    )

    loader = DataLoader(
        TensorDataset(
            X_tensor
        ),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    preds = []

    for (
        xb,
    ) in loader:

        xb = xb.to(
            DEVICE,
            non_blocking=True,
        )

        pred = model(
            xb
        )

        preds.append(
            pred.cpu().numpy()
        )

    return np.concatenate(
        preds
    )


# ============================================================
# Inner validation:
# select training epoch WITHOUT using outer test labels
# ============================================================

def select_best_epoch(
    X_outer_train,
    y_outer_train,
    seed,
):

    indices = np.arange(
        len(
            X_outer_train
        )
    )

    inner_train_idx, inner_val_idx = (
        train_test_split(
            indices,
            test_size=INNER_VAL_FRACTION,
            random_state=seed,
            shuffle=True,
        )
    )


    scaler = StandardScaler()

    X_inner_train = scaler.fit_transform(
        X_outer_train[
            inner_train_idx
        ]
    )

    X_inner_val = scaler.transform(
        X_outer_train[
            inner_val_idx
        ]
    )


    y_inner_train = (
        y_outer_train[
            inner_train_idx
        ]
    )

    y_inner_val = (
        y_outer_train[
            inner_val_idx
        ]
    )


    set_seed(
        seed
    )


    model = (
        FitnessMLP()
        .to(
            DEVICE
        )
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    criterion = nn.MSELoss()


    train_loader = make_loader(
        X_inner_train,
        y_inner_train,
        BATCH_SIZE,
        True,
    )


    val_loader = make_loader(
        X_inner_val,
        y_inner_val,
        BATCH_SIZE,
        False,
    )


    best_epoch = 1
    best_val_loss = float(
        "inf"
    )

    patience_counter = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )


        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
        )


        if (
            val_loss
            < best_val_loss
            - 1e-8
        ):

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

        else:

            patience_counter += 1


        if (
            epoch == 1
            or epoch % 25 == 0
            or patience_counter
            >= PATIENCE
        ):

            print(
                f"    epoch={epoch:3d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"best_epoch={best_epoch}"
            )


        if (
            patience_counter
            >= PATIENCE
        ):

            break


    return (
        best_epoch,
        best_val_loss,
    )


# ============================================================
# Refit MLP on full outer-train for selected epoch
# ============================================================

def fit_full_mlp(
    X_train,
    y_train,
    selected_epoch,
    seed,
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )


    set_seed(
        seed
    )


    model = (
        FitnessMLP()
        .to(
            DEVICE
        )
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    criterion = nn.MSELoss()


    loader = make_loader(
        X_train_scaled,
        y_train,
        BATCH_SIZE,
        True,
    )


    for _ in range(
        selected_epoch
    ):

        train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
        )


    return (
        scaler,
        model,
    )


# ============================================================
# Load SINGLE-MUTANT labels only
#
# Multi-mutant DMS_score is never loaded in this script.
# ============================================================

print(
    "=" * 90
)

print(
    "GFP Single-Mutant Fitness Prediction Benchmark"
)

print(
    "=" * 90
)

print(
    f"Device: {DEVICE}"
)


single_df = pd.read_csv(
    SINGLE_PATH,
    usecols=[
        "mutant",
        "DMS_score",
        "mutation_count",
    ],
)


if not (
    single_df[
        "mutation_count"
    ]
    == 1
).all():

    raise ValueError(
        "gfp_single_mutants.csv contains "
        "non-single mutants."
    )


feature_metadata = pd.read_csv(
    FEATURE_METADATA_PATH
)


if (
    feature_metadata[
        "mutant"
    ].duplicated().any()
):

    raise ValueError(
        "Duplicate mutant IDs in "
        "feature metadata."
    )


single_df = single_df.merge(
    feature_metadata[
        [
            "mutant",
            "feature_index",
            "mutation_count",
        ]
    ],
    on="mutant",
    how="left",
    validate="one_to_one",
    suffixes=(
        "_label",
        "_feature",
    ),
)


if single_df[
    "feature_index"
].isna().any():

    raise ValueError(
        "Some single mutants could not be "
        "matched to feature cache."
    )


feature_cache = torch.load(
    FEATURE_PATH,
    map_location="cpu",
)


features = (
    feature_cache[
        "features"
    ]
    .float()
    .numpy()
)


if features.shape != (
    51714,
    1920,
):

    print(
        "WARNING: feature cache shape is "
        f"{features.shape}, not the previously "
        "observed (51714, 1920)."
    )


feature_indices = (
    single_df[
        "feature_index"
    ]
    .astype(
        int
    )
    .to_numpy()
)


X = features[
    feature_indices
]


y = (
    single_df[
        "DMS_score"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


mutants = (
    single_df[
        "mutant"
    ]
    .astype(
        str
    )
    .to_numpy()
)


print()
print(
    "Input / output shapes"
)

print(
    f"All cached GFP features : "
    f"{features.shape}"
)

print(
    f"Single-mutant input X   : "
    f"{X.shape}"
)

print(
    f"Target y                : "
    f"{y.shape}"
)

print(
    f"Target meaning          : "
    f"continuous GFP DMS_score"
)


# ============================================================
# 5-fold CV
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)


ridge_oof = np.full(
    len(
        y
    ),
    np.nan,
    dtype=np.float64,
)


mlp_oof = np.full(
    len(
        y
    ),
    np.nan,
    dtype=np.float64,
)


fold_rows = []


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    kf.split(
        X
    )
):

    print()
    print(
        "=" * 90
    )

    print(
        f"Fold {fold}"
    )

    print(
        "=" * 90
    )

    print(
        f"Train: {len(train_idx)}"
    )

    print(
        f"Test : {len(test_idx)}"
    )


    X_train = X[
        train_idx
    ]

    y_train = y[
        train_idx
    ]


    X_test = X[
        test_idx
    ]

    y_test = y[
        test_idx
    ]


    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    ridge = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA,
                    solver="lsqr",
                    tol=1e-6,
                ),
            ),
        ]
    )


    ridge.fit(
        X_train.astype(
            np.float64
        ),
        y_train,
    )


    ridge_pred = ridge.predict(
        X_test.astype(
            np.float64
        )
    )


    ridge_oof[
        test_idx
    ] = ridge_pred


    ridge_metrics = compute_metrics(
        y_test,
        ridge_pred,
    )


    print()
    print(
        "Ridge:"
    )

    print(
        f"  Spearman = "
        f"{ridge_metrics['spearman']:.4f}"
    )

    print(
        f"  RMSE     = "
        f"{ridge_metrics['rmse']:.4f}"
    )

    print(
        f"  MAE      = "
        f"{ridge_metrics['mae']:.4f}"
    )

    print(
        f"  R2       = "
        f"{ridge_metrics['r2']:.4f}"
    )


    # --------------------------------------------------------
    # MLP epoch selection using inner validation
    # --------------------------------------------------------

    print()
    print(
        "MLP inner validation:"
    )


    fold_seed = (
        SEED
        + fold
    )


    selected_epoch, best_val_loss = (
        select_best_epoch(
            X_train,
            y_train,
            fold_seed,
        )
    )


    print(
        f"  selected_epoch = "
        f"{selected_epoch}"
    )


    # --------------------------------------------------------
    # Refit fresh MLP on entire outer train
    # --------------------------------------------------------

    mlp_scaler, mlp_model = (
        fit_full_mlp(
            X_train,
            y_train,
            selected_epoch,
            fold_seed,
        )
    )


    X_test_scaled = (
        mlp_scaler.transform(
            X_test
        )
    )


    mlp_pred = predict_mlp(
        mlp_model,
        X_test_scaled,
    )


    mlp_oof[
        test_idx
    ] = mlp_pred


    mlp_metrics = compute_metrics(
        y_test,
        mlp_pred,
    )


    print()
    print(
        "MLP:"
    )

    print(
        f"  Spearman = "
        f"{mlp_metrics['spearman']:.4f}"
    )

    print(
        f"  RMSE     = "
        f"{mlp_metrics['rmse']:.4f}"
    )

    print(
        f"  MAE      = "
        f"{mlp_metrics['mae']:.4f}"
    )

    print(
        f"  R2       = "
        f"{mlp_metrics['r2']:.4f}"
    )


    fold_rows.append(
        {
            "fold":
                fold,

            "n_train":
                int(
                    len(
                        train_idx
                    )
                ),

            "n_test":
                int(
                    len(
                        test_idx
                    )
                ),

            "mlp_selected_epoch":
                int(
                    selected_epoch
                ),

            "mlp_inner_best_val_mse":
                float(
                    best_val_loss
                ),

            "ridge_spearman":
                ridge_metrics[
                    "spearman"
                ],

            "ridge_rmse":
                ridge_metrics[
                    "rmse"
                ],

            "ridge_mae":
                ridge_metrics[
                    "mae"
                ],

            "ridge_r2":
                ridge_metrics[
                    "r2"
                ],

            "mlp_spearman":
                mlp_metrics[
                    "spearman"
                ],

            "mlp_rmse":
                mlp_metrics[
                    "rmse"
                ],

            "mlp_mae":
                mlp_metrics[
                    "mae"
                ],

            "mlp_r2":
                mlp_metrics[
                    "r2"
                ],
        }
    )


# ============================================================
# Validate OOF completeness
# ============================================================

if not np.isfinite(
    ridge_oof
).all():

    raise ValueError(
        "Missing Ridge OOF predictions."
    )


if not np.isfinite(
    mlp_oof
).all():

    raise ValueError(
        "Missing MLP OOF predictions."
    )


# ============================================================
# Global OOF metrics
# ============================================================

ridge_global = compute_metrics(
    y,
    ridge_oof,
)


mlp_global = compute_metrics(
    y,
    mlp_oof,
)


fold_df = pd.DataFrame(
    fold_rows
)


# ============================================================
# Fold mean ± SD
# ============================================================

def fold_summary(
    prefix,
):

    result = {}

    for metric in [
        "spearman",
        "rmse",
        "mae",
        "r2",
    ]:

        values = (
            fold_df[
                f"{prefix}_{metric}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        result[
            metric
        ] = {
            "mean":
                float(
                    values.mean()
                ),

            "std":
                float(
                    values.std(
                        ddof=1
                    )
                ),
        }

    return result


summary = {
    "protocol": {
        "dataset":
            "GFP_AEQVI_Sarkisyan_2016",

        "training_population":
            (
                "1084 single mutants only"
            ),

        "multi_mutant_label_usage":
            (
                "None. Multi-mutant DMS_score "
                "is never loaded."
            ),

        "input_shape":
            [
                int(
                    len(
                        X
                    )
                ),
                1920,
            ],

        "feature":
            (
                "[WT context mean, "
                "local perturbation sum]"
            ),

        "outer_cv":
            "5-fold shuffled KFold, seed 42",

        "mlp_epoch_selection":
            (
                "15% validation split inside "
                "outer-train only; fresh model "
                "refit on full outer-train for "
                "selected epoch"
            ),
    },

    "ridge": {
        "alpha":
            RIDGE_ALPHA,

        "fold":
            fold_summary(
                "ridge"
            ),

        "global_oof":
            ridge_global,
    },

    "mlp": {
        "architecture":
            "1920 -> 512 -> 128 -> 1",

        "dropout":
            DROPOUT,

        "lr":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "fold":
            fold_summary(
                "mlp"
            ),

        "global_oof":
            mlp_global,

        "selected_epochs":
            fold_df[
                "mlp_selected_epoch"
            ]
            .astype(
                int
            )
            .tolist(),
    },
}


# ============================================================
# Save outputs
# ============================================================

fold_df.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)


oof_df = pd.DataFrame(
    {
        "mutant":
            mutants,

        "DMS_score":
            y,

        "ridge_prediction":
            ridge_oof,

        "mlp_prediction":
            mlp_oof,

        "ridge_abs_error":
            np.abs(
                ridge_oof
                - y
            ),

        "mlp_abs_error":
            np.abs(
                mlp_oof
                - y
            ),
    }
)


oof_df.to_csv(
    OOF_PATH,
    index=False,
)


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Final report
# ============================================================

print()
print(
    "=" * 90
)

print(
    "Final GFP Single-Mutant CV Summary"
)

print(
    "=" * 90
)


for name, result in [
    (
        "Ridge",
        summary[
            "ridge"
        ],
    ),
    (
        "MLP",
        summary[
            "mlp"
        ],
    ),
]:

    print()
    print(
        name
    )


    print(
        "Fold mean ± SD"
    )


    for metric in [
        "spearman",
        "rmse",
        "mae",
        "r2",
    ]:

        mean = (
            result[
                "fold"
            ][
                metric
            ][
                "mean"
            ]
        )

        std = (
            result[
                "fold"
            ][
                metric
            ][
                "std"
            ]
        )


        print(
            f"  {metric:9s}: "
            f"{mean:.4f} ± "
            f"{std:.4f}"
        )


    print(
        "Global OOF"
    )


    for metric, value in (
        result[
            "global_oof"
        ]
        .items()
    ):

        print(
            f"  {metric:9s}: "
            f"{value:.4f}"
        )


print()
print(
    "MLP selected epochs:",
    summary[
        "mlp"
    ][
        "selected_epochs"
    ],
)


print()
print(
    "Saved:"
)

print(
    FOLD_METRICS_PATH
)

print(
    OOF_PATH
)

print(
    SUMMARY_PATH
)


print()
print(
    "IMPORTANT:"
)

print(
    "This benchmark used labels from "
    "GFP single mutants only."
)

print(
    "No multi-mutant DMS_score was loaded."
)
