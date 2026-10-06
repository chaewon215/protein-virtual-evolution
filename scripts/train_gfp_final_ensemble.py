from pathlib import Path
import json
import random
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Step 8-5
# Final GFP Deep Ensemble Training
# + Label-Free Multi-Mutant Prediction
# ============================================================


# ============================================================
# Paths
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
    / "final_ensemble"
)

CHECKPOINT_ROOT = (
    OUTPUT_ROOT
    / "checkpoints"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


PREDICTIONS_PATH = (
    OUTPUT_ROOT
    / "multi_mutant_predictions.csv"
)

MEMBER_PREDICTIONS_PATH = (
    OUTPUT_ROOT
    / "multi_mutant_member_predictions.npy"
)

SCALER_PATH = (
    OUTPUT_ROOT
    / "scalers.pkl"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


# ============================================================
# Frozen protocol
# ============================================================

SEED = 42

ENSEMBLE_SIZE = 5

MEMBER_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

# Median of Step 8-4 selected epochs:
# [24, 9, 7, 48, 34]
# median = 24
FINAL_EPOCHS = 24

BATCH_SIZE = 64

PRED_BATCH_SIZE = 512

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

DROPOUT = 0.2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Feature layout
#
# WT context mean    [960]
# Delta local sum    [960]
# Delta global       [960]
#
# concat:
# [2880]
#
# Multi-mutant example with K mutations:
#
# WT embeddings at K positions
# [K, 960]
#       ↓ mean
# [960]
#
# Local embedding differences
# [K, 960]
#       ↓ sum
# [960]
#
# Whole-protein difference
# [960]
#
#       ↓
# [2880]
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
# Frozen final model
#
# Input:
#   x [B, 2880]
#
# Branch 1:
#   WT context
#   [B,960] -> [B,64]
#
# Branch 2:
#   Delta local
#   [B,960] -> [B,64]
#
# Branch 3:
#   Delta global
#   [B,960] -> [B,64]
#
# Concatenate:
#   [B,192]
#
# Head:
#   [B,192] -> [B,64] -> [B,1]
#
# During training:
#   output is standardized GFP DMS_score
#
# During inference:
#   inverse transform -> GFP DMS_score scale
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


        x_local = x[
            :,
            LOCAL_START:LOCAL_END,
        ]


        x_global = x[
            :,
            GLOBAL_START:GLOBAL_END,
        ]


        z_wt = self.wt_branch(
            x_wt
        )


        z_local = (
            self.local_delta_branch(
                x_local
            )
        )


        z_global = (
            self.global_delta_branch(
                x_global
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
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Training helpers
# ============================================================

def make_train_loader(
    X,
    y,
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
        shuffle=True,
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


        prediction = model(
            xb
        )


        loss = criterion(
            prediction,
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
def predict_scaled(
    model,
    X,
):
    # Dropout OFF at inference.
    # Ensemble uncertainty comes from
    # independent trained models.
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
        batch_size=PRED_BATCH_SIZE,
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
# Load SINGLE-MUTANT labels
#
# These are the ONLY labels used in this script.
# ============================================================

print(
    "=" * 110
)

print(
    "Step 8-5: Final GFP Deep Ensemble"
)

print(
    "=" * 110
)

print(
    f"Device          : {DEVICE}"
)

print(
    f"Ensemble size   : {ENSEMBLE_SIZE}"
)

print(
    f"Member seeds    : {MEMBER_SEEDS}"
)

print(
    f"Final epochs    : {FINAL_EPOCHS}"
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
        "non-single variants."
    )


if (
    single_df[
        "mutant"
    ]
    .duplicated()
    .any()
):

    raise ValueError(
        "Duplicate single-mutant IDs found."
    )


# ============================================================
# Load LABEL-FREE feature metadata
#
# This file must provide only indexing / mutation metadata.
# DMS_score is never requested or used.
# ============================================================

feature_metadata = pd.read_csv(
    FEATURE_METADATA_PATH
)


required_metadata_columns = {
    "mutant",
    "feature_index",
    "mutation_count",
}


missing_columns = (
    required_metadata_columns
    - set(
        feature_metadata.columns
    )
)


if missing_columns:

    raise ValueError(
        "Feature metadata is missing columns: "
        f"{sorted(missing_columns)}"
    )


if (
    feature_metadata[
        "mutant"
    ]
    .duplicated()
    .any()
):

    raise ValueError(
        "Duplicate mutant IDs found in "
        "feature metadata."
    )


# ============================================================
# Align single-mutant rows to feature cache
# ============================================================

single_index_df = single_df[
    [
        "mutant",
        "DMS_score",
        "mutation_count",
    ]
].merge(
    feature_metadata[
        [
            "mutant",
            "feature_index",
            "mutation_count",
        ]
    ].rename(
        columns={
            "mutation_count":
                "feature_mutation_count",
        }
    ),

    on="mutant",

    how="left",

    validate="one_to_one",
)


if (
    single_index_df[
        "feature_index"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "Some single mutants could not be "
        "aligned to cached features."
    )


if not (
    single_index_df[
        "feature_mutation_count"
    ]
    == 1
).all():

    raise ValueError(
        "Feature metadata reports unexpected "
        "mutation counts for single mutants."
    )


# ============================================================
# Identify hidden multi-mutant candidate pool
#
# IMPORTANT:
#
# We use ONLY feature metadata.
# No multi-mutant DMS_score file is opened.
#
# Candidate pool:
# mutation_count >= 2
# ============================================================

multi_meta = (
    feature_metadata.loc[
        feature_metadata[
            "mutation_count"
        ]
        >= 2
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if len(
    multi_meta
) == 0:

    raise ValueError(
        "No multi-mutant candidates found "
        "in feature metadata."
    )


# ============================================================
# Load feature cache
# ============================================================

feature_cache = torch.load(
    FEATURE_PATH,
    map_location="cpu",
)


required_feature_keys = {
    "wt_context_mean",
    "delta_local_sum",
    "delta_global",
}


missing_feature_keys = (
    required_feature_keys
    - set(
        feature_cache.keys()
    )
)


if missing_feature_keys:

    raise ValueError(
        "Feature cache is missing keys: "
        f"{sorted(missing_feature_keys)}"
    )


n_cached = (
    feature_cache[
        "wt_context_mean"
    ].shape[
        0
    ]
)


for key in [
    "wt_context_mean",
    "delta_local_sum",
    "delta_global",
]:

    tensor = feature_cache[
        key
    ]

    if tensor.shape != (
        n_cached,
        960,
    ):

        raise ValueError(
            f"Unexpected shape for {key}: "
            f"{tuple(tensor.shape)}"
        )


# ============================================================
# Build training matrix
#
# Single mutants:
#
# WT context      [1084,960]
# Delta local     [1084,960]
# Delta global    [1084,960]
#
# concat:
# X_train_raw     [1084,2880]
#
# y_train_raw     [1084]
# ============================================================

single_feature_indices = (
    single_index_df[
        "feature_index"
    ]
    .astype(
        int
    )
    .to_numpy()
)


single_wt = (
    feature_cache[
        "wt_context_mean"
    ][
        single_feature_indices
    ]
    .float()
    .numpy()
)


single_local = (
    feature_cache[
        "delta_local_sum"
    ][
        single_feature_indices
    ]
    .float()
    .numpy()
)


single_global = (
    feature_cache[
        "delta_global"
    ][
        single_feature_indices
    ]
    .float()
    .numpy()
)


X_train_raw = np.concatenate(
    [
        single_wt,
        single_local,
        single_global,
    ],
    axis=1,
)


y_train_raw = (
    single_index_df[
        "DMS_score"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


# ============================================================
# Build hidden candidate feature matrix
#
# Multi-mutants:
#
# WT context mean      [50630,960]
# Delta local sum      [50630,960]
# Delta global         [50630,960]
#
# concat:
# X_multi_raw          [50630,2880]
#
# NO LABEL VECTOR IS CREATED.
# ============================================================

multi_feature_indices = (
    multi_meta[
        "feature_index"
    ]
    .astype(
        int
    )
    .to_numpy()
)


multi_wt = (
    feature_cache[
        "wt_context_mean"
    ][
        multi_feature_indices
    ]
    .float()
    .numpy()
)


multi_local = (
    feature_cache[
        "delta_local_sum"
    ][
        multi_feature_indices
    ]
    .float()
    .numpy()
)


multi_global = (
    feature_cache[
        "delta_global"
    ][
        multi_feature_indices
    ]
    .float()
    .numpy()
)


X_multi_raw = np.concatenate(
    [
        multi_wt,
        multi_local,
        multi_global,
    ],
    axis=1,
)


# ============================================================
# Shape validation
# ============================================================

if X_train_raw.shape[
    1
] != 2880:

    raise ValueError(
        f"Unexpected train input shape: "
        f"{X_train_raw.shape}"
    )


if X_multi_raw.shape[
    1
] != 2880:

    raise ValueError(
        f"Unexpected multi input shape: "
        f"{X_multi_raw.shape}"
    )


print()
print(
    "Input / output shapes"
)

print(
    f"Single-mutant train X : "
    f"{X_train_raw.shape}"
)

print(
    f"Single-mutant train y : "
    f"{y_train_raw.shape}"
)

print(
    f"Hidden multi X        : "
    f"{X_multi_raw.shape}"
)

print(
    f"Hidden multi labels   : "
    f"NOT LOADED"
)


# ============================================================
# Fit final scalers on ALL labeled single mutants
# ============================================================

x_scaler = StandardScaler()


X_train = (
    x_scaler.fit_transform(
        X_train_raw
    )
)


X_multi = (
    x_scaler.transform(
        X_multi_raw
    )
)


y_scaler = StandardScaler()


y_train = (
    y_scaler.fit_transform(
        y_train_raw.reshape(
            -1,
            1,
        )
    )
    .ravel()
)


# ============================================================
# Save scalers for Step 8-6 retraining / inference
# ============================================================

with open(
    SCALER_PATH,
    "wb",
) as f:

    pickle.dump(
        {
            "x_scaler":
                x_scaler,

            "y_scaler":
                y_scaler,

            "fitted_on":
                (
                    "1084 GFP single-mutant "
                    "labeled samples only"
                ),
        },
        f,
    )


# ============================================================
# Train final 5-member ensemble
# ============================================================

probe = BranchFusionMLP()


print()
print(
    "Final Branch Fusion model"
)

print(
    "WT context    [B,960] -> [B,64]"
)

print(
    "Delta local   [B,960] -> [B,64]"
)

print(
    "Delta global  [B,960] -> [B,64]"
)

print(
    "concat                    [B,192]"
)

print(
    "head          [B,192] -> [B,64] -> [B,1]"
)

print(
    f"Trainable params/member : "
    f"{count_parameters(probe):,}"
)


del probe


member_predictions = []

training_rows = []


for member_idx, member_seed in enumerate(
    MEMBER_SEEDS
):

    print()
    print(
        "=" * 90
    )

    print(
        f"Training member "
        f"{member_idx + 1}/{ENSEMBLE_SIZE}"
    )

    print(
        f"Seed   : {member_seed}"
    )

    print(
        f"Epochs : {FINAL_EPOCHS}"
    )

    print(
        "=" * 90
    )


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


    train_loader = make_train_loader(
        X_train,
        y_train,
    )


    epoch_losses = []


    for epoch in range(
        1,
        FINAL_EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )


        epoch_losses.append(
            train_loss
        )


        if (
            epoch == 1
            or epoch == FINAL_EPOCHS
            or epoch % 5 == 0
        ):

            print(
                f"  epoch "
                f"{epoch:3d}/{FINAL_EPOCHS} "
                f"| train MSE(scaled) = "
                f"{train_loss:.6f}"
            )


    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_ROOT
        / (
            f"branch_fusion_member_"
            f"{member_idx}.pt"
        )
    )


    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "member_index":
                int(
                    member_idx
                ),

            "seed":
                int(
                    member_seed
                ),

            "epochs":
                int(
                    FINAL_EPOCHS
                ),

            "input_dim":
                2880,

            "architecture":
                (
                    "WT[960]->64; "
                    "DeltaLocal[960]->64; "
                    "DeltaGlobal[960]->64; "
                    "concat[192]->64->1"
                ),
        },
        checkpoint_path,
    )


    # --------------------------------------------------------
    # Predict ALL hidden multi-mutants
    #
    # scaled prediction:
    # [50630]
    #
    # inverse transform:
    # original GFP DMS_score units
    # --------------------------------------------------------

    prediction_scaled = predict_scaled(
        model,
        X_multi,
    )


    prediction = (
        y_scaler
        .inverse_transform(
            prediction_scaled.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )


    member_predictions.append(
        prediction
    )


    training_rows.append(
        {
            "member":
                int(
                    member_idx
                ),

            "seed":
                int(
                    member_seed
                ),

            "epochs":
                int(
                    FINAL_EPOCHS
                ),

            "final_train_mse_scaled":
                float(
                    epoch_losses[
                        -1
                    ]
                ),

            "checkpoint":
                str(
                    checkpoint_path
                ),
        }
    )


    del model


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Ensemble predictions
#
# member matrix:
# [N_multi, 5]
#
# mean:
# mu [N_multi]
#
# sample std:
# sigma [N_multi]
#
# sigma is RELATIVE epistemic disagreement.
# It is NOT a calibrated confidence interval.
# ============================================================

member_predictions = np.column_stack(
    member_predictions
)


expected_shape = (
    len(
        multi_meta
    ),
    ENSEMBLE_SIZE,
)


if (
    member_predictions.shape
    != expected_shape
):

    raise ValueError(
        "Unexpected ensemble prediction shape: "
        f"{member_predictions.shape}"
    )


mu = member_predictions.mean(
    axis=1
)


sigma = member_predictions.std(
    axis=1,
    ddof=1,
)


# ============================================================
# Save raw member matrix
# ============================================================

np.save(
    MEMBER_PREDICTIONS_PATH,
    member_predictions,
)


# ============================================================
# Build LABEL-FREE candidate table
#
# Intentionally excludes:
# - DMS_score
# - true fitness
# - absolute error
#
# Safe for acquisition policy input.
# ============================================================

safe_columns = [
    "mutant",
    "mutation_count",
    "feature_index",
]


optional_columns = [
    "mutation_positions",
]


for column in optional_columns:

    if column in multi_meta.columns:

        safe_columns.append(
            column
        )


candidate_df = (
    multi_meta[
        safe_columns
    ]
    .copy()
)


candidate_df[
    "predicted_fitness_mu"
] = mu


candidate_df[
    "uncertainty_sigma"
] = sigma


# Add individual member predictions for auditability.
for member_idx in range(
    ENSEMBLE_SIZE
):

    candidate_df[
        f"member_{member_idx}_prediction"
    ] = member_predictions[
        :,
        member_idx
    ]


candidate_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# Summary
# ============================================================

training_df = pd.DataFrame(
    training_rows
)


summary = {
    "stage":
        "Step 8-5",

    "dataset":
        "GFP_AEQVI_Sarkisyan_2016",

    "label_policy": {
        "training_labels":
            (
                "GFP single-mutant DMS_score "
                "only"
            ),

        "multi_mutant_labels_loaded":
            False,

        "candidate_prediction_table_contains_dms_score":
            False,
    },

    "training": {
        "n_labeled_single_mutants":
            int(
                len(
                    single_df
                )
            ),

        "input_shape":
            list(
                X_train_raw.shape
            ),

        "target_shape":
            list(
                y_train_raw.shape
            ),

        "epochs":
            int(
                FINAL_EPOCHS
            ),

        "epoch_source":
            (
                "median of Step 8-4 outer-fold "
                "selected epochs [24, 9, 7, 48, 34]"
            ),

        "ensemble_size":
            int(
                ENSEMBLE_SIZE
            ),

        "member_seeds":
            MEMBER_SEEDS,

        "model":
            (
                "BranchFusionMLP"
            ),

        "parameter_count_per_member":
            196929,

        "representation":
            {
                "wt_context_mean":
                    960,

                "delta_local_sum":
                    960,

                "delta_global":
                    960,

                "total":
                    2880,
            },
    },

    "hidden_candidate_pool": {
        "n_multi_mutants":
            int(
                len(
                    multi_meta
                )
            ),

        "input_shape":
            list(
                X_multi_raw.shape
            ),

        "member_prediction_shape":
            list(
                member_predictions.shape
            ),

        "mu_shape":
            list(
                mu.shape
            ),

        "sigma_shape":
            list(
                sigma.shape
            ),
    },

    "prediction_distribution": {
        "mu_min":
            float(
                mu.min()
            ),

        "mu_mean":
            float(
                mu.mean()
            ),

        "mu_median":
            float(
                np.median(
                    mu
                )
            ),

        "mu_max":
            float(
                mu.max()
            ),

        "sigma_min":
            float(
                sigma.min()
            ),

        "sigma_mean":
            float(
                sigma.mean()
            ),

        "sigma_median":
            float(
                np.median(
                    sigma
                )
            ),

        "sigma_max":
            float(
                sigma.max()
            ),
    },

    "uncertainty_definition":
        (
            "Sample standard deviation "
            "(ddof=1) across five independently "
            "trained BranchFusionMLP predictions "
            "in original GFP DMS-score units. "
            "This is a relative epistemic "
            "disagreement signal, not a "
            "calibrated confidence interval."
        ),

    "artifacts": {
        "candidate_predictions":
            str(
                PREDICTIONS_PATH
            ),

        "member_predictions":
            str(
                MEMBER_PREDICTIONS_PATH
            ),

        "scalers":
            str(
                SCALER_PATH
            ),

        "checkpoint_directory":
            str(
                CHECKPOINT_ROOT
            ),
    },
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
# Console report
# ============================================================

print()
print(
    "=" * 110
)

print(
    "Final GFP Multi-Mutant Prediction Summary"
)

print(
    "=" * 110
)


print()
print(
    "Training"
)

print(
    f"Labeled single mutants : "
    f"{len(single_df)}"
)

print(
    f"Train X                : "
    f"{X_train_raw.shape}"
)

print(
    f"Train y                : "
    f"{y_train_raw.shape}"
)

print(
    f"Ensemble members       : "
    f"{ENSEMBLE_SIZE}"
)

print(
    f"Epochs / member        : "
    f"{FINAL_EPOCHS}"
)


print()
print(
    "Hidden candidate inference"
)

print(
    f"Multi-mutants          : "
    f"{len(multi_meta)}"
)

print(
    f"Candidate X            : "
    f"{X_multi_raw.shape}"
)

print(
    f"Member predictions     : "
    f"{member_predictions.shape}"
)

print(
    f"mu                     : "
    f"{mu.shape}"
)

print(
    f"sigma                  : "
    f"{sigma.shape}"
)


print()
print(
    "Prediction distribution"
)

print(
    f"mu    min/median/mean/max = "
    f"{mu.min():.6f} / "
    f"{np.median(mu):.6f} / "
    f"{mu.mean():.6f} / "
    f"{mu.max():.6f}"
)

print(
    f"sigma min/median/mean/max = "
    f"{sigma.min():.6f} / "
    f"{np.median(sigma):.6f} / "
    f"{sigma.mean():.6f} / "
    f"{sigma.max():.6f}"
)


print()
print(
    "Top 10 candidates by predicted fitness (mu)"
)

print(
    candidate_df.sort_values(
        "predicted_fitness_mu",
        ascending=False,
    )[
        [
            "mutant",
            "mutation_count",
            "predicted_fitness_mu",
            "uncertainty_sigma",
        ]
    ]
    .head(
        10
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Saved:"
)

print(
    PREDICTIONS_PATH
)

print(
    MEMBER_PREDICTIONS_PATH
)

print(
    SCALER_PATH
)

print(
    SUMMARY_PATH
)

print(
    CHECKPOINT_ROOT
)


print()
print(
    "IMPORTANT:"
)

print(
    "No GFP multi-mutant DMS_score "
    "was loaded anywhere in this script."
)

print(
    "The candidate prediction table is "
    "label-free and is safe to use as "
    "input to Step 8-6 acquisition policies."
)

print(
    "predicted_fitness_mu is the "
    "ensemble-mean predicted GFP fitness."
)

print(
    "uncertainty_sigma is ensemble "
    "disagreement, not a calibrated "
    "probability or confidence interval."
)
