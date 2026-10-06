from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
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
    / "architecture_ablation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_METRICS_PATH = (
    OUTPUT_ROOT
    / "fold_metrics.csv"
)

INNER_EPOCHS_PATH = (
    OUTPUT_ROOT
    / "inner_cv_epochs.csv"
)

SUMMARY_CSV_PATH = (
    OUTPUT_ROOT
    / "summary.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

OOF_PATH = (
    OUTPUT_ROOT
    / "oof_predictions.csv"
)


# ============================================================
# Experiment settings
# ============================================================

SEED = 42

OUTER_SPLITS = 5
INNER_SPLITS = 3

BATCH_SIZE = 64

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DROPOUT = 0.2

MAX_EPOCHS = 300
PATIENCE = 20

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Input layout
#
# Final GFP representation:
#
# WT context mean    [960]
# Delta local        [960]
# Delta global       [960]
#
# concatenate
#      ↓
# x                  [2880]
#
# Slices:
#
# x[:,    0: 960] = WT context
# x[:,  960:1920] = Delta local
# x[:, 1920:2880] = Delta global
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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


# ============================================================
# Model 1: Compact MLP
#
# Input:
#   [B, 2880]
#
# 2880 -> 128 -> 32 -> 1
#
# Output:
#   [B]
#   standardized predicted DMS_score
# ============================================================

class CompactMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                2880,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(self, x):
        return (
            self.net(
                x
            )
            .squeeze(
                -1
            )
        )


# ============================================================
# Model 2: Residual Linear + MLP
#
# Input:
#   x [B, 2880]
#
# Linear path:
#   2880 -> 1
#
# Nonlinear path:
#   2880 -> 128 -> 32 -> 1
#
# Final:
#   y_hat = y_linear + y_nonlinear
#
# Output:
#   [B]
# ============================================================

class ResidualMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_path = nn.Linear(
            2880,
            1,
        )

        self.nonlinear_path = nn.Sequential(
            nn.Linear(
                2880,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                32,
                1,
            ),
        )

        # Start the nonlinear residual close to zero so that
        # optimization can initially behave like a linear model.
        nn.init.zeros_(
            self.nonlinear_path[-1].weight
        )

        nn.init.zeros_(
            self.nonlinear_path[-1].bias
        )

    def forward(self, x):
        y_linear = (
            self.linear_path(
                x
            )
            .squeeze(
                -1
            )
        )

        y_nonlinear = (
            self.nonlinear_path(
                x
            )
            .squeeze(
                -1
            )
        )

        return (
            y_linear
            + y_nonlinear
        )


# ============================================================
# Model 3: Local-Global Gated Fusion
#
# Local input:
#   [WT, Delta local]
#   [B, 1920]
#
# Local branch:
#   1920 -> 128 -> 64
#
# Global input:
#   Delta global
#   [B, 960]
#
# Global branch:
#   960 -> 64
#
# Gate:
#   concat(local, global)
#   [B, 128]
#       ↓
#   128 -> 64 -> sigmoid
#       ↓
#   g [B, 64]
#
# Fusion:
#   z = z_local + g * z_global
#   [B, 64]
#
# Head:
#   64 -> 32 -> 1
#
# Output:
#   [B]
# ============================================================

class GatedFusionMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.local_branch = nn.Sequential(
            nn.Linear(
                1920,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
        )

        self.global_branch = nn.Sequential(
            nn.Linear(
                960,
                64,
            ),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(
                128,
                64,
            ),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(self, x):

        x_local = x[
            :,
            WT_START:LOCAL_END,
        ]  # [B, 1920]

        x_global = x[
            :,
            GLOBAL_START:GLOBAL_END,
        ]  # [B, 960]

        z_local = self.local_branch(
            x_local
        )  # [B, 64]

        z_global = self.global_branch(
            x_global
        )  # [B, 64]

        gate_input = torch.cat(
            [
                z_local,
                z_global,
            ],
            dim=1,
        )  # [B, 128]

        g = self.gate(
            gate_input
        )  # [B, 64]

        z = (
            z_local
            + g * z_global
        )  # [B, 64]

        return (
            self.head(
                z
            )
            .squeeze(
                -1
            )
        )


# ============================================================
# Model 4: Branch-wise Fusion
#
# WT context:
#   [B, 960]
#       ↓
#   960 -> 64
#
# Delta local:
#   [B, 960]
#       ↓
#   960 -> 64
#
# Delta global:
#   [B, 960]
#       ↓
#   960 -> 64
#
# concatenate:
#   [B, 192]
#
# Fusion head:
#   192 -> 64 -> 1
#
# Output:
#   [B]
# ============================================================

class BranchFusionMLP(nn.Module):

    def __init__(self):
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

    def forward(self, x):

        x_wt = x[
            :,
            WT_START:WT_END,
        ]  # [B, 960]

        x_local_delta = x[
            :,
            LOCAL_START:LOCAL_END,
        ]  # [B, 960]

        x_global_delta = x[
            :,
            GLOBAL_START:GLOBAL_END,
        ]  # [B, 960]

        z_wt = self.wt_branch(
            x_wt
        )  # [B, 64]

        z_local = self.local_delta_branch(
            x_local_delta
        )  # [B, 64]

        z_global = self.global_delta_branch(
            x_global_delta
        )  # [B, 64]

        z = torch.cat(
            [
                z_wt,
                z_local,
                z_global,
            ],
            dim=1,
        )  # [B, 192]

        return (
            self.head(
                z
            )
            .squeeze(
                -1
            )
        )


# ============================================================
# Model registry
# ============================================================

MODEL_BUILDERS = {
    "compact_mlp":
        CompactMLP,

    "residual_mlp":
        ResidualMLP,

    "gated_fusion":
        GatedFusionMLP,

    "branch_fusion":
        BranchFusionMLP,
}


# ============================================================
# Utility
# ============================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


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
# Inner CV epoch selection
#
# For each outer train:
#
# outer train
#    ↓
# 3-fold inner CV
#
# inner fold 0 -> best epoch e0
# inner fold 1 -> best epoch e1
# inner fold 2 -> best epoch e2
#
# selected epoch
# = rounded median(e0, e1, e2)
#
# This is more robust than using one random validation split.
# ============================================================

def select_epoch_inner_cv(
    model_name,
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

    inner_records = []

    for inner_fold, (
        inner_train_idx,
        inner_val_idx,
    ) in enumerate(
        inner_cv.split(
            X_outer_train
        )
    ):

        # ----------------------------------------------------
        # Input scaling
        # fitted ONLY on inner train
        # ----------------------------------------------------

        x_scaler = (
            StandardScaler()
        )

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
        # Target scaling
        # fitted ONLY on inner train labels
        # ----------------------------------------------------

        y_scaler = (
            StandardScaler()
        )

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
            MODEL_BUILDERS[
                model_name
            ]()
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

        inner_records.append(
            {
                "model":
                    model_name,

                "outer_fold":
                    int(
                        outer_fold
                    ),

                "inner_fold":
                    int(
                        inner_fold
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
        inner_records,
    )


# ============================================================
# Final outer-train refit
#
# x scaler:
# fitted on ALL outer train
#
# y scaler:
# fitted on ALL outer train labels
#
# fresh model:
# trained selected_epoch times
#
# outer test:
# evaluated exactly once
# ============================================================

def fit_predict_outer(
    model_name,
    X_train,
    y_train,
    X_test,
    selected_epoch,
    seed,
):
    x_scaler = (
        StandardScaler()
    )

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


    y_scaler = (
        StandardScaler()
    )

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


    set_seed(
        seed
    )

    model = (
        MODEL_BUILDERS[
            model_name
        ]()
        .to(
            DEVICE
        )
    )

    parameter_count = (
        count_parameters(
            model
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


    pred_scaled = predict_scaled(
        model,
        X_test_scaled,
    )


    pred = (
        y_scaler
        .inverse_transform(
            pred_scaled.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )


    return (
        pred,
        parameter_count,
    )


# ============================================================
# Load GFP single-mutant labels ONLY
#
# IMPORTANT:
# No gfp_multi_mutants.csv is opened anywhere.
# ============================================================

print(
    "=" * 100
)

print(
    "GFP Architecture Ablation"
)

print(
    "=" * 100
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
# Construct final [2880] input
#
# Single mutant:
#
# wt_context_mean
# [N, 960]
#
# delta_local_sum
# [N, 960]
#
# delta_global
# [N, 960]
#
# concatenate
#      ↓
# X
# [N, 2880]
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
        single_df
    ),
    2880,
):
    raise ValueError(
        f"Unexpected input shape: {X.shape}"
    )


print()
print(
    "Input / output"
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
    "Output meaning  : "
    "continuous GFP DMS_score"
)


print()
print(
    "Model parameter counts"
)

for (
    model_name,
    builder,
) in MODEL_BUILDERS.items():

    probe = builder()

    print(
        f"{model_name:20s}: "
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
# Run architecture ablation
# ============================================================

fold_rows = []

inner_epoch_rows = []

oof_data = {
    "mutant":
        mutants,

    "DMS_score":
        y,
}


for model_idx, model_name in enumerate(
    MODEL_BUILDERS.keys()
):

    print()
    print(
        "#" * 100
    )

    print(
        f"Model: {model_name}"
    )

    print(
        "#" * 100
    )


    oof_pred = np.full(
        len(
            y
        ),
        np.nan,
        dtype=np.float64,
    )


    for outer_fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_folds
    ):

        print()
        print(
            f"Outer fold {outer_fold}"
        )

        print(
            f"  train = "
            f"{len(train_idx)}"
        )

        print(
            f"  test  = "
            f"{len(test_idx)}"
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


        # ----------------------------------------------------
        # Robust inner-CV epoch selection
        # ----------------------------------------------------

        (
            selected_epoch,
            inner_best_epochs,
            inner_records,
        ) = select_epoch_inner_cv(
            model_name,
            X_train,
            y_train,
            outer_fold,
        )


        inner_epoch_rows.extend(
            inner_records
        )


        print(
            f"  inner best epochs = "
            f"{inner_best_epochs}"
        )

        print(
            f"  selected epoch    = "
            f"{selected_epoch}"
        )


        # ----------------------------------------------------
        # Fresh full outer-train fit
        # ----------------------------------------------------

        final_seed = (
            SEED
            + model_idx * 10000
            + outer_fold
        )


        (
            prediction,
            parameter_count,
        ) = fit_predict_outer(
            model_name,
            X_train,
            y_train,
            X_test,
            selected_epoch,
            final_seed,
        )


        oof_pred[
            test_idx
        ] = prediction


        metrics = compute_metrics(
            y_test,
            prediction,
        )


        print(
            f"  Spearman = "
            f"{metrics['spearman']:.4f}"
        )

        print(
            f"  RMSE     = "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"  MAE      = "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"  R2       = "
            f"{metrics['r2']:.4f}"
        )


        fold_rows.append(
            {
                "model":
                    model_name,

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

                "parameter_count":
                    int(
                        parameter_count
                    ),

                **metrics,
            }
        )


    if not np.isfinite(
        oof_pred
    ).all():

        raise ValueError(
            f"Missing OOF predictions "
            f"for {model_name}."
        )


    oof_data[
        model_name
    ] = oof_pred


# ============================================================
# Aggregate
# ============================================================

fold_df = pd.DataFrame(
    fold_rows
)


summary_rows = []


for (
    model_name,
    group,
) in fold_df.groupby(
    "model",
    sort=False,
):

    global_oof_metrics = (
        compute_metrics(
            y,
            oof_data[
                model_name
            ],
        )
    )


    row = {
        "model":
            model_name,

        "parameter_count":
            int(
                group[
                    "parameter_count"
                ].iloc[
                    0
                ]
            ),

        "selected_epoch_mean":
            float(
                group[
                    "selected_epoch"
                ].mean()
            ),

        "selected_epoch_std":
            float(
                group[
                    "selected_epoch"
                ].std(
                    ddof=1
                )
            ),

        "selected_epoch_min":
            int(
                group[
                    "selected_epoch"
                ].min()
            ),

        "selected_epoch_max":
            int(
                group[
                    "selected_epoch"
                ].max()
            ),
    }


    for metric_name in [
        "spearman",
        "rmse",
        "mae",
        "r2",
    ]:

        values = (
            group[
                metric_name
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        row[
            f"{metric_name}_mean"
        ] = float(
            values.mean()
        )


        row[
            f"{metric_name}_std"
        ] = float(
            values.std(
                ddof=1
            )
        )


        row[
            f"{metric_name}_global_oof"
        ] = global_oof_metrics[
            metric_name
        ]


    summary_rows.append(
        row
    )


summary_df = (
    pd.DataFrame(
        summary_rows
    )
    .sort_values(
        by=[
            "spearman_global_oof",
            "rmse_global_oof",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# Save
# ============================================================

fold_df.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)


pd.DataFrame(
    inner_epoch_rows
).to_csv(
    INNER_EPOCHS_PATH,
    index=False,
)


summary_df.to_csv(
    SUMMARY_CSV_PATH,
    index=False,
)


pd.DataFrame(
    oof_data
).to_csv(
    OOF_PATH,
    index=False,
)


summary_json = {
    "protocol": {
        "dataset":
            "GFP_AEQVI_Sarkisyan_2016",

        "population":
            "1084 single mutants only",

        "multi_mutant_label_usage":
            "None",

        "input":
            {
                "wt_context":
                    960,

                "delta_local":
                    960,

                "delta_global":
                    960,

                "total":
                    2880,
            },

        "target":
            (
                "continuous GFP DMS_score"
            ),

        "outer_cv":
            (
                "5-fold shuffled KFold "
                "with random_state=42"
            ),

        "epoch_selection":
            (
                "3-fold inner CV inside "
                "each outer train; final "
                "epoch is rounded median "
                "of three inner best epochs"
            ),

        "input_scaling":
            (
                "StandardScaler fitted "
                "within each training split"
            ),

        "target_scaling":
            (
                "StandardScaler fitted "
                "within each training split"
            ),

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "dropout":
            DROPOUT,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,
    },

    "models": {
        "compact_mlp":
            (
                "2880 -> 128 -> 32 -> 1"
            ),

        "residual_mlp":
            (
                "linear(2880->1) + "
                "MLP(2880->128->32->1)"
            ),

        "gated_fusion":
            (
                "local[1920]->128->64; "
                "global[960]->64; "
                "sigmoid gate; "
                "fusion 64->32->1"
            ),

        "branch_fusion":
            (
                "WT[960]->64; "
                "DeltaLocal[960]->64; "
                "DeltaGlobal[960]->64; "
                "concat[192]->64->1"
            ),
    },

    "results":
        summary_df.to_dict(
            orient="records"
        ),
}


with open(
    SUMMARY_JSON_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_json,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Final report
# ============================================================

print()
print(
    "=" * 140
)

print(
    "GFP Architecture Ablation Summary"
)

print(
    "=" * 140
)


display_columns = [
    "model",
    "parameter_count",
    "spearman_global_oof",
    "rmse_global_oof",
    "mae_global_oof",
    "r2_global_oof",
    "spearman_mean",
    "spearman_std",
    "selected_epoch_mean",
    "selected_epoch_std",
    "selected_epoch_min",
    "selected_epoch_max",
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
    "Saved:"
)

print(
    FOLD_METRICS_PATH
)

print(
    INNER_EPOCHS_PATH
)

print(
    SUMMARY_CSV_PATH
)

print(
    SUMMARY_JSON_PATH
)

print(
    OOF_PATH
)


print()
print(
    "IMPORTANT:"
)

print(
    "All architecture selection "
    "used GFP single-mutant labels only."
)

print(
    "No GFP multi-mutant DMS_score "
    "was loaded or evaluated."
)
