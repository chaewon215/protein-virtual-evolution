from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Step 8-4
# GFP Branch-Fusion Deep Ensemble + OOF Uncertainty Validation
# ============================================================


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
    / "deep_ensemble_uncertainty"
)

FIGURE_ROOT = (
    OUTPUT_ROOT
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


FOLD_METRICS_PATH = (
    OUTPUT_ROOT
    / "fold_metrics.csv"
)

MEMBER_METRICS_PATH = (
    OUTPUT_ROOT
    / "member_fold_metrics.csv"
)

INNER_EPOCHS_PATH = (
    OUTPUT_ROOT
    / "inner_cv_epochs.csv"
)

OOF_PATH = (
    OUTPUT_ROOT
    / "oof_predictions.csv"
)

QUINTILE_PATH = (
    OUTPUT_ROOT
    / "uncertainty_quintiles.csv"
)

RISK_COVERAGE_PATH = (
    OUTPUT_ROOT
    / "risk_coverage_baselines.csv"
)

RANDOM_REPEATS_PATH = (
    OUTPUT_ROOT
    / "random_selection_repeats.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


# ============================================================
# Reproducibility / training settings
# ============================================================

SEED = 42

OUTER_SPLITS = 5
INNER_SPLITS = 3

ENSEMBLE_SIZE = 5

MEMBER_BASE_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

BATCH_SIZE = 64

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

MAX_EPOCHS = 300
PATIENCE = 20

COVERAGES = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]

RANDOM_REPEATS = 1000

HIGH_ERROR_FRACTION = 0.20

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Input layout
#
# WT context mean       [960]
# Delta local sum       [960]
# Delta global          [960]
#
# Concatenate
#   ↓
# x                     [2880]
#
# Branch Fusion:
#
# WT branch:
#   [B, 960] -> [B, 64]
#
# Local-delta branch:
#   [B, 960] -> [B, 64]
#
# Global-delta branch:
#   [B, 960] -> [B, 64]
#
# concat:
#   [B, 192]
#
# head:
#   [B, 192] -> [B, 64] -> [B, 1]
#
# Output:
#   standardized predicted DMS score during training
#
# After inverse transform:
#   predicted GFP DMS_score [B]
# ============================================================

WT_START = 0
WT_END = 960

LOCAL_START = 960
LOCAL_END = 1920

GLOBAL_START = 1920
GLOBAL_END = 2880


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    return {
        "spearman":
            float(
                spearmanr(
                    y_true,
                    y_pred,
                ).statistic
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

        "mae":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "r2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),
    }


def compute_rmse(
    y_true,
    y_pred,
):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def compute_mae(
    y_true,
    y_pred,
):
    return float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )


# ============================================================
# Branch-Fusion model
#
# Trainable parameter count:
# 196,929
# ============================================================

class BranchFusionMLP(
    nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

        self.wt_branch = nn.Sequential(
            nn.Linear(
                960,
                64,
            ),
            nn.ReLU(),
        )

        self.local_delta_branch = nn.Sequential(
            nn.Linear(
                960,
                64,
            ),
            nn.ReLU(),
        )

        self.global_delta_branch = nn.Sequential(
            nn.Linear(
                960,
                64,
            ),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                192,
                64,
            ),
            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                1,
            ),
        )


    def forward(
        self,
        x,
    ):
        x_wt = x[
            :,
            WT_START:WT_END,
        ]


        x_local_delta = x[
            :,
            LOCAL_START:LOCAL_END,
        ]


        x_global_delta = x[
            :,
            GLOBAL_START:GLOBAL_END,
        ]


        z_wt = self.wt_branch(
            x_wt
        )


        z_local = (
            self.local_delta_branch(
                x_local_delta
            )
        )


        z_global = (
            self.global_delta_branch(
                x_global_delta
            )
        )


        z = torch.cat(
            [
                z_wt,
                z_local,
                z_global,
            ],
            dim=1,
        )


        return (
            self.head(
                z
            )
            .squeeze(
                -1
            )
        )


def count_parameters(
    model,
):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# DataLoader / training helpers
# ============================================================

def make_loader(
    X,
    y,
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
        batch_size=BATCH_SIZE,
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


        total_loss += (
            loss.item()
            * len(
                xb
            )
        )


        total_n += len(
            xb
        )


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


        total_loss += (
            loss.item()
            * len(
                xb
            )
        )


        total_n += len(
            xb
        )


    return (
        total_loss
        / total_n
    )


@torch.inference_mode()
def predict_scaled(
    model,
    X,
):
    # IMPORTANT:
    # model.eval() disables Dropout.
    # Therefore this is a Deep Ensemble,
    # NOT MC Dropout.
    model.eval()


    dataset = TensorDataset(
        torch.from_numpy(
            X.astype(
                np.float32
            )
        )
    )


    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )


    predictions = []


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


        predictions.append(
            pred.cpu().numpy()
        )


    return np.concatenate(
        predictions
    )


# ============================================================
# Robust epoch selection
#
# Done ONCE per outer fold.
#
# outer train
#      ↓
# 3-fold inner CV
#
# inner fold 0 -> best epoch e0
# inner fold 1 -> best epoch e1
# inner fold 2 -> best epoch e2
#
# selected epoch
# = rounded median(e0, e1, e2)
#
# The selected epoch is then shared by all 5 ensemble members
# in that outer fold.
#
# This isolates ensemble diversity to random initialization,
# minibatch order, and training stochasticity rather than
# different hyperparameter choices.
# ============================================================

def select_epoch_inner_cv(
    X_outer_train,
    y_outer_train,
    outer_fold,
):
    inner_cv = KFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=(
            SEED
            + 1000
            + outer_fold
        ),
    )


    best_epochs = []

    records = []


    for inner_fold, (
        inner_train_idx,
        inner_val_idx,
    ) in enumerate(
        inner_cv.split(
            X_outer_train
        )
    ):

        # ----------------------------------------------------
        # Input scaler
        # fit on inner train only
        # ----------------------------------------------------

        x_scaler = StandardScaler()


        X_inner_train = (
            x_scaler.fit_transform(
                X_outer_train[
                    inner_train_idx
                ]
            )
        )


        X_inner_val = (
            x_scaler.transform(
                X_outer_train[
                    inner_val_idx
                ]
            )
        )


        # ----------------------------------------------------
        # Target scaler
        # fit on inner train labels only
        # ----------------------------------------------------

        y_scaler = StandardScaler()


        y_inner_train = (
            y_scaler
            .fit_transform(
                y_outer_train[
                    inner_train_idx
                ].reshape(
                    -1,
                    1,
                )
            )
            .ravel()
        )


        y_inner_val = (
            y_scaler
            .transform(
                y_outer_train[
                    inner_val_idx
                ].reshape(
                    -1,
                    1,
                )
            )
            .ravel()
        )


        inner_seed = (
            SEED
            + outer_fold * 100
            + inner_fold
        )


        set_seed(
            inner_seed
        )


        model = (
            BranchFusionMLP()
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
            True,
        )


        val_loader = make_loader(
            X_inner_val,
            y_inner_val,
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

            train_one_epoch(
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

                best_val_loss = (
                    val_loss
                )

                best_epoch = (
                    epoch
                )

                patience_counter = 0

            else:

                patience_counter += 1


            if (
                patience_counter
                >= PATIENCE
            ):

                break


        best_epochs.append(
            best_epoch
        )


        records.append(
            {
                "outer_fold":
                    int(
                        outer_fold
                    ),

                "inner_fold":
                    int(
                        inner_fold
                    ),

                "seed":
                    int(
                        inner_seed
                    ),

                "best_epoch":
                    int(
                        best_epoch
                    ),

                "best_val_mse_scaled":
                    float(
                        best_val_loss
                    ),
            }
        )


    selected_epoch = max(
        1,

        int(
            np.rint(
                np.median(
                    best_epochs
                )
            )
        ),
    )


    return (
        selected_epoch,
        best_epochs,
        records,
    )


# ============================================================
# Train one final ensemble member
#
# All ensemble members:
# - use identical outer-training data
# - use identical train-fitted x scaler
# - use identical train-fitted y scaler
# - train for identical selected epoch
#
# Diversity comes from:
# - initialization
# - minibatch shuffling
# - Dropout during training
# ============================================================

def train_ensemble_member(
    X_train_scaled,
    y_train_scaled,
    selected_epoch,
    member_seed,
):
    set_seed(
        member_seed
    )


    model = (
        BranchFusionMLP()
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
        y_train_scaled,
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


    return model


# ============================================================
# Load GFP single-mutant labels ONLY
#
# No multi-mutant label file is opened.
# ============================================================

print(
    "=" * 100
)

print(
    "GFP Branch-Fusion Deep Ensemble"
)

print(
    "=" * 100
)

print(
    f"Device          : {DEVICE}"
)

print(
    f"Ensemble size   : {ENSEMBLE_SIZE}"
)

print(
    f"Base seeds      : {MEMBER_BASE_SEEDS}"
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
        "Single-mutant file contains "
        "non-single variants."
    )


feature_metadata = pd.read_csv(
    FEATURE_METADATA_PATH
)


single_df = single_df.merge(
    feature_metadata[
        [
            "mutant",
            "feature_index",
        ]
    ],
    on="mutant",
    how="left",
    validate="one_to_one",
)


if (
    single_df[
        "feature_index"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "Feature alignment failed."
    )


feature_cache = torch.load(
    FEATURE_PATH,
    map_location="cpu",
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


# ============================================================
# Construct [1084, 2880] input
# ============================================================

wt_context = (
    feature_cache[
        "wt_context_mean"
    ][
        feature_indices
    ]
    .float()
    .numpy()
)


delta_local = (
    feature_cache[
        "delta_local_sum"
    ][
        feature_indices
    ]
    .float()
    .numpy()
)


delta_global = (
    feature_cache[
        "delta_global"
    ][
        feature_indices
    ]
    .float()
    .numpy()
)


X = np.concatenate(
    [
        wt_context,
        delta_local,
        delta_global,
    ],
    axis=1,
)


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


if X.shape != (
    len(
        y
    ),
    2880,
):

    raise ValueError(
        f"Unexpected X shape: {X.shape}"
    )


probe = BranchFusionMLP()


print()
print(
    "Input / output shapes"
)

print(
    f"WT context      : "
    f"{wt_context.shape}"
)

print(
    f"Delta local     : "
    f"{delta_local.shape}"
)

print(
    f"Delta global    : "
    f"{delta_global.shape}"
)

print(
    f"Final X         : "
    f"{X.shape}"
)

print(
    f"Target y        : "
    f"{y.shape}"
)

print(
    "Per-member model:"
)

print(
    "  [960] -> [64]"
)

print(
    "  [960] -> [64]"
)

print(
    "  [960] -> [64]"
)

print(
    "  concat [192]"
)

print(
    "  [192] -> [64] -> [1]"
)

print(
    f"Trainable params: "
    f"{count_parameters(probe):,}"
)


del probe


# ============================================================
# Shared outer CV
# ============================================================

outer_cv = KFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=SEED,
)


outer_folds = list(
    outer_cv.split(
        X
    )
)


# ============================================================
# OOF storage
# ============================================================

oof_member_predictions = np.full(
    (
        len(
            y
        ),
        ENSEMBLE_SIZE,
    ),
    np.nan,
    dtype=np.float64,
)


oof_fold = np.full(
    len(
        y
    ),
    -1,
    dtype=np.int64,
)


fold_rows = []

member_rows = []

inner_epoch_rows = []


# ============================================================
# Outer CV
# ============================================================

for outer_fold, (
    train_idx,
    test_idx,
) in enumerate(
    outer_folds
):

    print()
    print(
        "=" * 100
    )

    print(
        f"Outer Fold {outer_fold}"
    )

    print(
        "=" * 100
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
    # Robust epoch selection ONCE for this outer fold
    # --------------------------------------------------------

    (
        selected_epoch,
        inner_best_epochs,
        inner_records,
    ) = select_epoch_inner_cv(
        X_train,
        y_train,
        outer_fold,
    )


    inner_epoch_rows.extend(
        inner_records
    )


    print(
        f"Inner best epochs : "
        f"{inner_best_epochs}"
    )

    print(
        f"Selected epoch    : "
        f"{selected_epoch}"
    )


    # --------------------------------------------------------
    # Fit scalers on entire outer train
    # --------------------------------------------------------

    x_scaler = StandardScaler()


    X_train_scaled = (
        x_scaler.fit_transform(
            X_train
        )
    )


    X_test_scaled = (
        x_scaler.transform(
            X_test
        )
    )


    y_scaler = StandardScaler()


    y_train_scaled = (
        y_scaler
        .fit_transform(
            y_train.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )


    # --------------------------------------------------------
    # Train 5 independent members
    # --------------------------------------------------------

    fold_member_predictions = []


    for member_idx, base_seed in enumerate(
        MEMBER_BASE_SEEDS
    ):

        member_seed = (
            base_seed
            + outer_fold * 1000
        )


        print(
            f"  Member "
            f"{member_idx + 1}/{ENSEMBLE_SIZE} "
            f"| seed={member_seed}"
        )


        model = train_ensemble_member(
            X_train_scaled,
            y_train_scaled,
            selected_epoch,
            member_seed,
        )


        pred_scaled = predict_scaled(
            model,
            X_test_scaled,
        )


        prediction = (
            y_scaler
            .inverse_transform(
                pred_scaled.reshape(
                    -1,
                    1,
                )
            )
            .ravel()
        )


        fold_member_predictions.append(
            prediction
        )


        member_metrics = compute_metrics(
            y_test,
            prediction,
        )


        member_rows.append(
            {
                "outer_fold":
                    int(
                        outer_fold
                    ),

                "member":
                    int(
                        member_idx
                    ),

                "seed":
                    int(
                        member_seed
                    ),

                "selected_epoch":
                    int(
                        selected_epoch
                    ),

                **member_metrics,
            }
        )


        del model


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # --------------------------------------------------------
    # [B, 5]
    # --------------------------------------------------------

    fold_member_predictions = (
        np.column_stack(
            fold_member_predictions
        )
    )


    if (
        fold_member_predictions.shape
        != (
            len(
                test_idx
            ),
            ENSEMBLE_SIZE,
        )
    ):

        raise ValueError(
            "Unexpected member prediction shape: "
            f"{fold_member_predictions.shape}"
        )


    oof_member_predictions[
        test_idx,
        :
    ] = fold_member_predictions


    oof_fold[
        test_idx
    ] = outer_fold


    # --------------------------------------------------------
    # Ensemble outputs
    #
    # mean [B]
    # std  [B], sample SD across members
    # --------------------------------------------------------

    ensemble_mean = (
        fold_member_predictions.mean(
            axis=1
        )
    )


    ensemble_std = (
        fold_member_predictions.std(
            axis=1,
            ddof=1,
        )
    )


    fold_metrics = compute_metrics(
        y_test,
        ensemble_mean,
    )


    absolute_error = np.abs(
        ensemble_mean
        - y_test
    )


    uncertainty_error_rho = float(
        spearmanr(
            ensemble_std,
            absolute_error,
        ).statistic
    )


    # --------------------------------------------------------
    # Exact top-20% error classification
    # --------------------------------------------------------

    n_high_error = max(
        1,
        int(
            np.ceil(
                HIGH_ERROR_FRACTION
                * len(
                    y_test
                )
            )
        ),
    )


    high_error_labels = np.zeros(
        len(
            y_test
        ),
        dtype=np.int64,
    )


    high_error_indices = (
        np.argsort(
            absolute_error
        )[
            -n_high_error:
        ]
    )


    high_error_labels[
        high_error_indices
    ] = 1


    high_error_auroc = float(
        roc_auc_score(
            high_error_labels,
            ensemble_std,
        )
    )


    fold_rows.append(
        {
            "outer_fold":
                int(
                    outer_fold
                ),

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

            "selected_epoch":
                int(
                    selected_epoch
                ),

            "inner_epoch_min":
                int(
                    np.min(
                        inner_best_epochs
                    )
                ),

            "inner_epoch_median":
                float(
                    np.median(
                        inner_best_epochs
                    )
                ),

            "inner_epoch_max":
                int(
                    np.max(
                        inner_best_epochs
                    )
                ),

            "mean_uncertainty":
                float(
                    ensemble_std.mean()
                ),

            "uncertainty_error_spearman":
                uncertainty_error_rho,

            "high_error_auroc":
                high_error_auroc,

            **fold_metrics,
        }
    )


    print()
    print(
        "Ensemble:"
    )

    print(
        f"  Spearman = "
        f"{fold_metrics['spearman']:.4f}"
    )

    print(
        f"  RMSE     = "
        f"{fold_metrics['rmse']:.4f}"
    )

    print(
        f"  MAE      = "
        f"{fold_metrics['mae']:.4f}"
    )

    print(
        f"  R2       = "
        f"{fold_metrics['r2']:.4f}"
    )

    print(
        f"  Mean unc = "
        f"{ensemble_std.mean():.4f}"
    )

    print(
        f"  unc-error rho = "
        f"{uncertainty_error_rho:.4f}"
    )

    print(
        f"  high-error AUROC = "
        f"{high_error_auroc:.4f}"
    )


# ============================================================
# OOF validation
# ============================================================

if not np.isfinite(
    oof_member_predictions
).all():

    raise ValueError(
        "Missing OOF member predictions."
    )


if (
    oof_fold
    < 0
).any():

    raise ValueError(
        "Missing OOF fold assignments."
    )


# ============================================================
# Global OOF ensemble
# ============================================================

oof_mean = (
    oof_member_predictions.mean(
        axis=1
    )
)


oof_std = (
    oof_member_predictions.std(
        axis=1,
        ddof=1,
    )
)


oof_abs_error = np.abs(
    oof_mean
    - y
)


global_metrics = compute_metrics(
    y,
    oof_mean,
)


uncertainty_error_rho = float(
    spearmanr(
        oof_std,
        oof_abs_error,
    ).statistic
)


# ============================================================
# High-error AUROC
#
# Positive class:
# exact top 20% of OOF samples by absolute error.
# ============================================================

n_high_error = max(
    1,
    int(
        np.ceil(
            HIGH_ERROR_FRACTION
            * len(
                y
            )
        )
    ),
)


high_error_labels = np.zeros(
    len(
        y
    ),
    dtype=np.int64,
)


high_error_indices = (
    np.argsort(
        oof_abs_error
    )[
        -n_high_error:
    ]
)


high_error_labels[
    high_error_indices
] = 1


high_error_auroc = float(
    roc_auc_score(
        high_error_labels,
        oof_std,
    )
)


# ============================================================
# Save OOF predictions
# ============================================================

oof_df = pd.DataFrame(
    {
        "mutant":
            mutants,

        "outer_fold":
            oof_fold,

        "DMS_score":
            y,

        "ensemble_mean":
            oof_mean,

        "ensemble_std":
            oof_std,

        "absolute_error":
            oof_abs_error,

        "high_error_top20":
            high_error_labels,
    }
)


for member_idx in range(
    ENSEMBLE_SIZE
):

    oof_df[
        f"member_{member_idx}_prediction"
    ] = oof_member_predictions[
        :,
        member_idx
    ]


oof_df.to_csv(
    OOF_PATH,
    index=False,
)


# ============================================================
# Uncertainty quintiles
#
# We sort by uncertainty and split into 5 nearly equal groups.
# This avoids qcut issues if uncertainty values tie.
#
# Q1 = lowest uncertainty
# Q5 = highest uncertainty
# ============================================================

uncertainty_order = np.argsort(
    oof_std
)


quintile_groups = np.array_split(
    uncertainty_order,
    5,
)


quintile_rows = []


for quintile_idx, indices in enumerate(
    quintile_groups,
    start=1,
):

    quintile_rows.append(
        {
            "quintile":
                int(
                    quintile_idx
                ),

            "n_samples":
                int(
                    len(
                        indices
                    )
                ),

            "uncertainty_min":
                float(
                    oof_std[
                        indices
                    ].min()
                ),

            "uncertainty_mean":
                float(
                    oof_std[
                        indices
                    ].mean()
                ),

            "uncertainty_max":
                float(
                    oof_std[
                        indices
                    ].max()
                ),

            "mae":
                compute_mae(
                    y[
                        indices
                    ],
                    oof_mean[
                        indices
                    ],
                ),

            "rmse":
                compute_rmse(
                    y[
                        indices
                    ],
                    oof_mean[
                        indices
                    ],
                ),
        }
    )


quintile_df = pd.DataFrame(
    quintile_rows
)


quintile_df.to_csv(
    QUINTILE_PATH,
    index=False,
)


# ============================================================
# Risk-Coverage Baselines
#
# 1. Uncertainty-based:
#    lowest ensemble_std first
#
# 2. Random:
#    same number of samples, repeated 1000 times
#
# 3. Oracle:
#    lowest TRUE absolute error first
#
# Oracle is retrospective only.
# It is not a usable selection strategy.
# ============================================================

oracle_order = np.argsort(
    oof_abs_error
)


rng = np.random.default_rng(
    SEED
)


risk_rows = []

random_repeat_rows = []


n_total = len(
    y
)


for coverage in COVERAGES:

    n_keep = max(
        1,

        int(
            np.floor(
                coverage
                * n_total
            )
        ),
    )


    # --------------------------------------------------------
    # Uncertainty selection
    # --------------------------------------------------------

    uncertainty_indices = (
        uncertainty_order[
            :n_keep
        ]
    )


    uncertainty_rmse = compute_rmse(
        y[
            uncertainty_indices
        ],
        oof_mean[
            uncertainty_indices
        ],
    )


    uncertainty_mae = compute_mae(
        y[
            uncertainty_indices
        ],
        oof_mean[
            uncertainty_indices
        ],
    )


    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    oracle_indices = (
        oracle_order[
            :n_keep
        ]
    )


    oracle_rmse = compute_rmse(
        y[
            oracle_indices
        ],
        oof_mean[
            oracle_indices
        ],
    )


    oracle_mae = compute_mae(
        y[
            oracle_indices
        ],
        oof_mean[
            oracle_indices
        ],
    )


    # --------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------

    random_rmses = []

    random_maes = []


    if n_keep == n_total:

        full_rmse = compute_rmse(
            y,
            oof_mean,
        )


        full_mae = compute_mae(
            y,
            oof_mean,
        )


        random_rmses = np.full(
            RANDOM_REPEATS,
            full_rmse,
            dtype=np.float64,
        )


        random_maes = np.full(
            RANDOM_REPEATS,
            full_mae,
            dtype=np.float64,
        )


    else:

        for repeat_idx in range(
            RANDOM_REPEATS
        ):

            random_indices = rng.choice(
                n_total,
                size=n_keep,
                replace=False,
            )


            repeat_rmse = compute_rmse(
                y[
                    random_indices
                ],
                oof_mean[
                    random_indices
                ],
            )


            repeat_mae = compute_mae(
                y[
                    random_indices
                ],
                oof_mean[
                    random_indices
                ],
            )


            random_rmses.append(
                repeat_rmse
            )


            random_maes.append(
                repeat_mae
            )


    random_rmses = np.asarray(
        random_rmses,
        dtype=np.float64,
    )


    random_maes = np.asarray(
        random_maes,
        dtype=np.float64,
    )


    for repeat_idx in range(
        RANDOM_REPEATS
    ):

        random_repeat_rows.append(
            {
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
                        repeat_idx
                    ),

                "rmse":
                    float(
                        random_rmses[
                            repeat_idx
                        ]
                    ),

                "mae":
                    float(
                        random_maes[
                            repeat_idx
                        ]
                    ),
            }
        )


    random_rmse_mean = float(
        random_rmses.mean()
    )


    random_mae_mean = float(
        random_maes.mean()
    )


    if n_keep == n_total:

        rmse_reduction_vs_random = 0.0

        random_percentile = np.nan

    else:

        rmse_reduction_vs_random = float(
            (
                random_rmse_mean
                - uncertainty_rmse
            )
            / random_rmse_mean
        )


        random_percentile = float(
            np.mean(
                random_rmses
                <= uncertainty_rmse
            )
        )


    risk_rows.append(
        {
            "coverage":
                float(
                    coverage
                ),

            "n_samples":
                int(
                    n_keep
                ),

            "uncertainty_rmse":
                float(
                    uncertainty_rmse
                ),

            "uncertainty_mae":
                float(
                    uncertainty_mae
                ),

            "random_rmse_mean":
                random_rmse_mean,

            "random_rmse_std":
                float(
                    random_rmses.std(
                        ddof=1
                    )
                ),

            "random_rmse_ci_low":
                float(
                    np.quantile(
                        random_rmses,
                        0.025,
                    )
                ),

            "random_rmse_ci_high":
                float(
                    np.quantile(
                        random_rmses,
                        0.975,
                    )
                ),

            "random_mae_mean":
                random_mae_mean,

            "random_mae_ci_low":
                float(
                    np.quantile(
                        random_maes,
                        0.025,
                    )
                ),

            "random_mae_ci_high":
                float(
                    np.quantile(
                        random_maes,
                        0.975,
                    )
                ),

            "oracle_rmse":
                float(
                    oracle_rmse
                ),

            "oracle_mae":
                float(
                    oracle_mae
                ),

            "uncertainty_rmse_reduction_vs_random":
                rmse_reduction_vs_random,

            "uncertainty_rmse_random_percentile":
                random_percentile,
        }
    )


risk_df = pd.DataFrame(
    risk_rows
)


risk_df.to_csv(
    RISK_COVERAGE_PATH,
    index=False,
)


pd.DataFrame(
    random_repeat_rows
).to_csv(
    RANDOM_REPEATS_PATH,
    index=False,
)


# ============================================================
# Fold / member tables
# ============================================================

fold_df = pd.DataFrame(
    fold_rows
)


member_df = pd.DataFrame(
    member_rows
)


inner_epoch_df = pd.DataFrame(
    inner_epoch_rows
)


fold_df.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)


member_df.to_csv(
    MEMBER_METRICS_PATH,
    index=False,
)


inner_epoch_df.to_csv(
    INNER_EPOCHS_PATH,
    index=False,
)


# ============================================================
# Summary
# ============================================================

def metric_mean_std(
    frame,
    column,
):
    values = frame[
        column
    ].to_numpy(
        dtype=np.float64
    )

    return {
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


summary = {
    "protocol": {
        "dataset":
            "GFP_AEQVI_Sarkisyan_2016",

        "population":
            "1084 single mutants only",

        "multi_mutant_label_usage":
            "None",

        "representation":
            {
                "wt_context_mean":
                    960,

                "delta_local_sum":
                    960,

                "delta_global":
                    960,

                "total_input_dim":
                    2880,
            },

        "model":
            {
                "name":
                    "BranchFusionMLP",

                "architecture":
                    (
                        "WT[960]->64; "
                        "DeltaLocal[960]->64; "
                        "DeltaGlobal[960]->64; "
                        "concat[192]->64->1"
                    ),

                "parameter_count":
                    196929,
            },

        "ensemble_size":
            ENSEMBLE_SIZE,

        "member_base_seeds":
            MEMBER_BASE_SEEDS,

        "outer_cv":
            (
                "5-fold shuffled KFold, "
                "random_state=42"
            ),

        "epoch_selection":
            (
                "3-fold inner CV on each "
                "outer-train; rounded median "
                "of inner best epochs"
            ),

        "uncertainty":
            (
                "sample standard deviation "
                "across 5 ensemble-member "
                "predictions in original "
                "DMS-score units"
            ),

        "high_error_definition":
            (
                "exact top 20% of samples "
                "by absolute OOF prediction error"
            ),

        "random_baseline_repeats":
            RANDOM_REPEATS,
    },

    "global_oof_prediction": {
        **global_metrics,

        "mean_uncertainty":
            float(
                oof_std.mean()
            ),
    },

    "global_uncertainty_quality": {
        "uncertainty_error_spearman":
            uncertainty_error_rho,

        "high_error_auroc":
            high_error_auroc,
    },

    "fold_prediction": {
        metric_name:
            metric_mean_std(
                fold_df,
                metric_name,
            )

        for metric_name in [
            "spearman",
            "rmse",
            "mae",
            "r2",
        ]
    },

    "fold_uncertainty": {
        "mean_uncertainty":
            metric_mean_std(
                fold_df,
                "mean_uncertainty",
            ),

        "uncertainty_error_spearman":
            metric_mean_std(
                fold_df,
                "uncertainty_error_spearman",
            ),

        "high_error_auroc":
            metric_mean_std(
                fold_df,
                "high_error_auroc",
            ),
    },

    "selected_epochs":
        fold_df[
            "selected_epoch"
        ]
        .astype(
            int
        )
        .tolist(),

    "uncertainty_quintiles":
        quintile_df.to_dict(
            orient="records"
        ),

    "risk_coverage":
        risk_df.to_dict(
            orient="records"
        ),
}


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
# Figures
#
# Each chart is a separate figure.
# ============================================================

# ------------------------------------------------------------
# Figure 1: Uncertainty vs absolute error
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        7,
        5,
    )
)


ax.scatter(
    oof_std,
    oof_abs_error,
    alpha=0.35,
    s=16,
)


sorted_indices = np.argsort(
    oof_std
)


binned_indices = np.array_split(
    sorted_indices,
    10,
)


bin_x = [
    float(
        oof_std[
            idx
        ].mean()
    )
    for idx in binned_indices
]


bin_y = [
    float(
        oof_abs_error[
            idx
        ].mean()
    )
    for idx in binned_indices
]


ax.plot(
    bin_x,
    bin_y,
    marker="o",
    linewidth=2,
    label=(
        "Equal-frequency bin mean"
    ),
)


ax.set_xlabel(
    "Ensemble uncertainty (std)"
)

ax.set_ylabel(
    "Absolute prediction error"
)

ax.set_title(
    (
        "GFP Deep Ensemble: "
        "Uncertainty vs Prediction Error\n"
        f"Spearman ρ = "
        f"{uncertainty_error_rho:.3f}"
    )
)

ax.legend()

fig.tight_layout()


fig.savefig(
    FIGURE_ROOT
    / "uncertainty_vs_error.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------
# Figure 2: Uncertainty quintile vs MAE
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        7,
        5,
    )
)


ax.plot(
    quintile_df[
        "quintile"
    ],
    quintile_df[
        "mae"
    ],
    marker="o",
    linewidth=2,
)


ax.set_xticks(
    quintile_df[
        "quintile"
    ]
)


ax.set_xlabel(
    (
        "Uncertainty quintile "
        "(Q1 = lowest uncertainty)"
    )
)

ax.set_ylabel(
    "MAE"
)

ax.set_title(
    (
        "GFP Prediction Error "
        "by Uncertainty Quintile"
    )
)

fig.tight_layout()


fig.savefig(
    FIGURE_ROOT
    / "uncertainty_quintile_vs_mae.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------
# Figure 3: Risk-Coverage + Random + Oracle
# ------------------------------------------------------------

coverage_percent = (
    risk_df[
        "coverage"
    ]
    .to_numpy()
    * 100.0
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


ax.plot(
    coverage_percent,
    risk_df[
        "random_rmse_mean"
    ],
    marker="o",
    linewidth=2,
    label="Random selection",
)


ax.fill_between(
    coverage_percent,
    risk_df[
        "random_rmse_ci_low"
    ],
    risk_df[
        "random_rmse_ci_high"
    ],
    alpha=0.15,
    label="Random 95% interval",
)


ax.plot(
    coverage_percent,
    risk_df[
        "uncertainty_rmse"
    ],
    marker="o",
    linewidth=2,
    label="Uncertainty-based",
)


ax.plot(
    coverage_percent,
    risk_df[
        "oracle_rmse"
    ],
    marker="o",
    linewidth=2,
    linestyle="--",
    label="Oracle (true error)",
)


ax.set_xlabel(
    "Coverage (%)"
)

ax.set_ylabel(
    "RMSE"
)

ax.set_title(
    (
        "GFP Risk-Coverage "
        "with Selection Baselines"
    )
)

ax.legend()

fig.tight_layout()


fig.savefig(
    FIGURE_ROOT
    / "risk_coverage_baselines.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# Console report
# ============================================================

print()
print(
    "=" * 110
)

print(
    "GFP Deep Ensemble OOF Summary"
)

print(
    "=" * 110
)


print()
print(
    "Prediction performance"
)

print(
    f"Spearman : "
    f"{global_metrics['spearman']:.6f}"
)

print(
    f"RMSE     : "
    f"{global_metrics['rmse']:.6f}"
)

print(
    f"MAE      : "
    f"{global_metrics['mae']:.6f}"
)

print(
    f"R2       : "
    f"{global_metrics['r2']:.6f}"
)


print()
print(
    "Uncertainty quality"
)

print(
    f"Mean uncertainty       : "
    f"{oof_std.mean():.6f}"
)

print(
    f"Uncertainty-error rho  : "
    f"{uncertainty_error_rho:.6f}"
)

print(
    f"High-error AUROC       : "
    f"{high_error_auroc:.6f}"
)


print()
print(
    "Selected epochs"
)

print(
    fold_df[
        "selected_epoch"
    ]
    .astype(
        int
    )
    .tolist()
)


print()
print(
    "Uncertainty quintiles"
)

print(
    quintile_df.to_string(
        index=False
    )
)


print()
print(
    "Risk-Coverage baselines"
)

display_risk_columns = [
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


print(
    risk_df[
        display_risk_columns
    ].to_string(
        index=False
    )
)


print()
print(
    "Saved:"
)

for path in [
    FOLD_METRICS_PATH,
    MEMBER_METRICS_PATH,
    INNER_EPOCHS_PATH,
    OOF_PATH,
    QUINTILE_PATH,
    RISK_COVERAGE_PATH,
    RANDOM_REPEATS_PATH,
    SUMMARY_PATH,
    FIGURE_ROOT
    / "uncertainty_vs_error.png",
    FIGURE_ROOT
    / "uncertainty_quintile_vs_mae.png",
    FIGURE_ROOT
    / "risk_coverage_baselines.png",
]:

    print(
        path
    )


print()
print(
    "IMPORTANT:"
)

print(
    "This script used GFP single-mutant "
    "DMS_score labels only."
)

print(
    "No GFP multi-mutant DMS_score "
    "was loaded or evaluated."
)

print(
    "ensemble_std is a relative "
    "epistemic disagreement signal, "
    "not a calibrated confidence interval."
)