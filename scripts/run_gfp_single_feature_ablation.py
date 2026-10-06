from pathlib import Path
import json
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
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

WT_FASTA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_wt.fasta"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "single_feature_ablation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_METRICS_PATH = (
    OUTPUT_ROOT
    / "fold_metrics.csv"
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


SEED = 42
N_SPLITS = 5

RIDGE_ALPHAS = [
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

MLP_HIDDEN_1 = 128
MLP_HIDDEN_2 = 32
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

AA_LIST = list(
    "ACDEFGHIKLMNPQRSTVWY"
)

AA_TO_INDEX = {
    aa: idx
    for idx, aa
    in enumerate(
        AA_LIST
    )
}

MUTATION_RE = re.compile(
    r"^([A-Z])(\d+)([A-Z])$"
)


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
    y_true,
    y_pred,
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
# Compact MLP
#
# Input
#   [B, D]
#
# Hidden
#   [B, 128]
#   [B, 32]
#
# Output
#   [B, 1]
#
# During training:
#   output = standardized DMS_score
#
# During final prediction:
#   inverse-transform back to original DMS_score scale
# ============================================================

class CompactMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
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


def count_parameters(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
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
# Compact MLP fit/predict
#
# Outer test is NEVER used for epoch selection.
#
# outer train
#    ↓
# inner train / validation
#    ↓
# best epoch
#    ↓
# fresh preprocessing on full outer train
#    ↓
# fresh MLP trained for selected epoch
#    ↓
# outer test prediction
# ============================================================

def fit_predict_compact_mlp(
    X_train,
    y_train,
    X_test,
    seed,
):
    all_indices = np.arange(
        len(
            X_train
        )
    )

    (
        inner_train_idx,
        inner_val_idx,
    ) = train_test_split(
        all_indices,
        test_size=INNER_VAL_FRACTION,
        random_state=seed,
        shuffle=True,
    )


    # --------------------------------------------------------
    # Inner feature scaling
    # --------------------------------------------------------

    x_scaler_inner = (
        StandardScaler()
    )

    X_inner_train = (
        x_scaler_inner.fit_transform(
            X_train[
                inner_train_idx
            ]
        )
    )

    X_inner_val = (
        x_scaler_inner.transform(
            X_train[
                inner_val_idx
            ]
        )
    )


    # --------------------------------------------------------
    # Inner target scaling
    #
    # Important:
    # fit ONLY on inner train labels.
    # --------------------------------------------------------

    y_scaler_inner = (
        StandardScaler()
    )

    y_inner_train = (
        y_scaler_inner
        .fit_transform(
            y_train[
                inner_train_idx
            ].reshape(
                -1,
                1,
            )
        )
        .ravel()
    )

    y_inner_val = (
        y_scaler_inner
        .transform(
            y_train[
                inner_val_idx
            ].reshape(
                -1,
                1,
            )
        )
        .ravel()
    )


    # --------------------------------------------------------
    # Inner model
    # --------------------------------------------------------

    set_seed(
        seed
    )

    model = CompactMLP(
        X_train.shape[
            1
        ]
    ).to(
        DEVICE
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


    # --------------------------------------------------------
    # Refit preprocessing on full outer train
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Fresh model on full outer train
    # --------------------------------------------------------

    set_seed(
        seed
    )

    final_model = CompactMLP(
        X_train.shape[
            1
        ]
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    full_train_loader = (
        make_loader(
            X_train_scaled,
            y_train_scaled,
            True,
        )
    )

    for _ in range(
        best_epoch
    ):
        train_one_epoch(
            final_model,
            full_train_loader,
            optimizer,
            criterion,
        )


    # --------------------------------------------------------
    # Predict standardized target
    # --------------------------------------------------------

    pred_scaled = predict_scaled(
        final_model,
        X_test_scaled,
    )


    # --------------------------------------------------------
    # Back to original GFP DMS_score scale
    # --------------------------------------------------------

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

    return {
        "prediction":
            pred,

        "selected_epoch":
            int(
                best_epoch
            ),

        "best_val_mse_scaled":
            float(
                best_val_loss
            ),

        "parameter_count":
            int(
                parameter_count
            ),
    }


# ============================================================
# Load GFP single-mutant labels ONLY
# ============================================================

print(
    "=" * 100
)

print(
    "GFP Single-Mutant Representation + Predictor Ablation"
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
        "gfp_single_mutants.csv "
        "contains non-single mutants."
    )


# ============================================================
# Feature alignment
# ============================================================

feature_metadata = (
    pd.read_csv(
        FEATURE_METADATA_PATH
    )
)


if (
    feature_metadata[
        "mutant"
    ]
    .duplicated()
    .any()
):

    raise ValueError(
        "Duplicate mutant IDs "
        "in feature metadata."
    )


single_df = (
    single_df.merge(
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
)


if (
    single_df[
        "feature_index"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "Some single mutants could "
        "not be aligned to ESM-C features."
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
# Cached single-mutant representations
#
# Because K = 1:
#
# wt_context_mean
# = WT local
# [1084, 960]
#
# delta_local_sum
# = Delta local
# [1084, 960]
# ============================================================

wt_local = (
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


# Mutant local:
#
# h_mut
# = h_WT + (h_mut - h_WT)
#
# shape [1084, 960]
mutant_local = (
    wt_local
    + delta_local
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


# ============================================================
# Simple mutation one-hot baseline
#
# GFP sequence length = 238
#
# position one-hot    [238]
# WT AA one-hot       [20]
# mutant AA one-hot   [20]
#
# total               [278]
# ============================================================

wt_sequence = "".join(
    line.strip()
    for line in (
        WT_FASTA_PATH
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )
    if (
        line.strip()
        and not line.startswith(
            ">"
        )
    )
)


sequence_length = len(
    wt_sequence
)


simple_onehot = np.zeros(
    (
        len(
            single_df
        ),
        sequence_length
        + 20
        + 20,
    ),
    dtype=np.float32,
)


for row_idx, mutant in enumerate(
    single_df[
        "mutant"
    ].astype(
        str
    )
):

    match = (
        MUTATION_RE.fullmatch(
            mutant
        )
    )


    if match is None:
        raise ValueError(
            f"Invalid mutation: {mutant}"
        )


    (
        wt_aa,
        position,
        mut_aa,
    ) = match.groups()


    position = int(
        position
    )


    if (
        wt_sequence[
            position - 1
        ]
        != wt_aa
    ):

        raise ValueError(
            f"WT sequence mismatch: {mutant}"
        )


    # position one-hot
    simple_onehot[
        row_idx,
        position - 1,
    ] = 1.0


    # WT amino acid one-hot
    simple_onehot[
        row_idx,
        sequence_length
        + AA_TO_INDEX[
            wt_aa
        ],
    ] = 1.0


    # mutant amino acid one-hot
    simple_onehot[
        row_idx,
        sequence_length
        + 20
        + AA_TO_INDEX[
            mut_aa
        ],
    ] = 1.0


# ============================================================
# Feature sets
#
# Each has fixed [N, D] input.
# ============================================================

feature_sets = {
    # --------------------------------
    # Non-ESM baseline
    # --------------------------------
    "simple_onehot":
        simple_onehot,

    # --------------------------------
    # Local ESM-C
    # --------------------------------
    "wt_local":
        wt_local,

    "delta_local":
        delta_local,

    "mutant_local":
        mutant_local,

    "wt_local_plus_delta_local":
        np.concatenate(
            [
                wt_local,
                delta_local,
            ],
            axis=1,
        ),

    # --------------------------------
    # Global perturbation
    # --------------------------------
    "delta_global":
        delta_global,

    # --------------------------------
    # Local + global combinations
    # --------------------------------
    "mutant_local_plus_delta_global":
        np.concatenate(
            [
                mutant_local,
                delta_global,
            ],
            axis=1,
        ),

    "wt_delta_plus_delta_global":
        np.concatenate(
            [
                wt_local,
                delta_local,
                delta_global,
            ],
            axis=1,
        ),
}


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
    f"Samples / target       : "
    f"{len(y)} / {y.shape}"
)

print(
    "Target meaning         : "
    "continuous GFP DMS_score"
)

print()

for (
    feature_name,
    X,
) in feature_sets.items():

    print(
        f"{feature_name:34s}"
        f" -> {X.shape}"
    )


# ============================================================
# Shared outer 5-fold splits
#
# Every representation/model sees EXACTLY the same test folds.
# ============================================================

outer_cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)


outer_folds = list(
    outer_cv.split(
        np.arange(
            len(
                y
            )
        )
    )
)


# ============================================================
# Experiment
# ============================================================

fold_rows = []


oof_data = {
    "mutant":
        mutants,

    "DMS_score":
        y,
}


for (
    feature_name,
    X,
) in feature_sets.items():

    print()
    print(
        "#" * 100
    )

    print(
        f"Feature: {feature_name}"
    )

    print(
        f"Input shape: {X.shape}"
    )

    compact_probe = (
        CompactMLP(
            X.shape[
                1
            ]
        )
    )

    print(
        "Compact MLP:"
    )

    print(
        f"  [{X.shape[1]}] "
        f"-> [{MLP_HIDDEN_1}] "
        f"-> [{MLP_HIDDEN_2}] "
        f"-> [1]"
    )

    print(
        f"  parameters = "
        f"{count_parameters(compact_probe):,}"
    )

    del compact_probe

    print(
        "#" * 100
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


    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_folds
    ):

        print()
        print(
            f"Fold {fold}"
        )

        print(
            f"  train = {len(train_idx)}"
        )

        print(
            f"  test  = {len(test_idx)}"
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


        # ====================================================
        # Tuned Ridge
        #
        # scaler is INSIDE Pipeline,
        # so each inner-CV training fold fits its own scaler.
        # ====================================================

        ridge_pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "ridge",
                    Ridge(
                        solver="lsqr",
                        tol=1e-6,
                    ),
                ),
            ]
        )


        ridge_inner_cv = KFold(
            n_splits=3,
            shuffle=True,
            random_state=(
                SEED
                + fold
            ),
        )


        ridge_search = GridSearchCV(
            estimator=ridge_pipeline,

            param_grid={
                "ridge__alpha":
                    RIDGE_ALPHAS
            },

            scoring=(
                "neg_mean_squared_error"
            ),

            cv=ridge_inner_cv,

            refit=True,

            n_jobs=-1,
        )


        ridge_search.fit(
            X_train.astype(
                np.float64
            ),
            y_train,
        )


        ridge_prediction = (
            ridge_search.predict(
                X_test.astype(
                    np.float64
                )
            )
        )


        ridge_oof[
            test_idx
        ] = ridge_prediction


        ridge_metrics = compute_metrics(
            y_test,
            ridge_prediction,
        )


        ridge_best_alpha = float(
            ridge_search.best_params_[
                "ridge__alpha"
            ]
        )


        # ====================================================
        # Compact MLP
        # ====================================================

        mlp_result = (
            fit_predict_compact_mlp(
                X_train,
                y_train,
                X_test,
                seed=(
                    SEED
                    + fold
                ),
            )
        )


        mlp_prediction = (
            mlp_result[
                "prediction"
            ]
        )


        mlp_oof[
            test_idx
        ] = mlp_prediction


        mlp_metrics = compute_metrics(
            y_test,
            mlp_prediction,
        )


        # ====================================================
        # Console
        # ====================================================

        print(
            f"  Ridge       "
            f"alpha={ridge_best_alpha:g} | "
            f"rho={ridge_metrics['spearman']:.4f} | "
            f"RMSE={ridge_metrics['rmse']:.4f} | "
            f"R2={ridge_metrics['r2']:.4f}"
        )


        print(
            f"  Compact MLP "
            f"epoch={mlp_result['selected_epoch']:3d} | "
            f"rho={mlp_metrics['spearman']:.4f} | "
            f"RMSE={mlp_metrics['rmse']:.4f} | "
            f"R2={mlp_metrics['r2']:.4f}"
        )


        # ====================================================
        # Fold records
        # ====================================================

        fold_rows.append(
            {
                "feature":
                    feature_name,

                "input_dim":
                    int(
                        X.shape[
                            1
                        ]
                    ),

                "model":
                    "ridge",

                "fold":
                    int(
                        fold
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

                "selected_alpha":
                    ridge_best_alpha,

                "selected_epoch":
                    np.nan,

                "parameter_count":
                    np.nan,

                **ridge_metrics,
            }
        )


        fold_rows.append(
            {
                "feature":
                    feature_name,

                "input_dim":
                    int(
                        X.shape[
                            1
                        ]
                    ),

                "model":
                    "compact_mlp",

                "fold":
                    int(
                        fold
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

                "selected_alpha":
                    np.nan,

                "selected_epoch":
                    int(
                        mlp_result[
                            "selected_epoch"
                        ]
                    ),

                "parameter_count":
                    int(
                        mlp_result[
                            "parameter_count"
                        ]
                    ),

                "inner_best_val_mse_scaled":
                    float(
                        mlp_result[
                            "best_val_mse_scaled"
                        ]
                    ),

                **mlp_metrics,
            }
        )


    # --------------------------------------------------------
    # Validate OOF
    # --------------------------------------------------------

    if not np.isfinite(
        ridge_oof
    ).all():

        raise ValueError(
            f"Missing Ridge OOF "
            f"for {feature_name}."
        )


    if not np.isfinite(
        mlp_oof
    ).all():

        raise ValueError(
            f"Missing MLP OOF "
            f"for {feature_name}."
        )


    oof_data[
        f"{feature_name}__ridge"
    ] = ridge_oof


    oof_data[
        f"{feature_name}__compact_mlp"
    ] = mlp_oof


# ============================================================
# Aggregate results
# ============================================================

fold_df = pd.DataFrame(
    fold_rows
)


summary_rows = []


for (
    feature_name,
    model_name,
), group in fold_df.groupby(
    [
        "feature",
        "model",
    ],
    sort=False,
):

    prediction_column = (
        f"{feature_name}"
        f"__{model_name}"
    )


    global_oof_metrics = (
        compute_metrics(
            y,
            oof_data[
                prediction_column
            ],
        )
    )


    row = {
        "feature":
            feature_name,

        "model":
            model_name,

        "input_dim":
            int(
                group[
                    "input_dim"
                ].iloc[
                    0
                ]
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
# Save outputs
# ============================================================

fold_df.to_csv(
    FOLD_METRICS_PATH,
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
            "1084 GFP single mutants only",

        "multi_mutant_label_usage":
            "None",

        "outer_cv":
            (
                "5-fold shuffled KFold, "
                "random_state=42"
            ),

        "shared_outer_folds":
            True,

        "ridge": {
            "alpha_grid":
                RIDGE_ALPHAS,

            "alpha_selection":
                (
                    "3-fold inner CV "
                    "inside outer train only"
                ),
        },

        "compact_mlp": {
            "architecture":
                "input -> 128 -> 32 -> 1",

            "dropout":
                DROPOUT,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "target_scaling":
                (
                    "StandardScaler fit on "
                    "training labels only"
                ),

            "epoch_selection":
                (
                    "15% validation split "
                    "inside outer train only"
                ),
        },
    },

    "feature_shapes": {
        feature_name:
            list(
                X.shape
            )
        for (
            feature_name,
            X,
        ) in feature_sets.items()
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
# Final console summary
# ============================================================

print()
print(
    "=" * 130
)

print(
    "GFP Single-Mutant Feature Ablation Summary"
)

print(
    "=" * 130
)


display_columns = [
    "feature",
    "model",
    "input_dim",
    "spearman_global_oof",
    "rmse_global_oof",
    "mae_global_oof",
    "r2_global_oof",
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
    "This experiment uses GFP "
    "single-mutant labels only."
)

print(
    "No multi-mutant DMS_score "
    "was loaded or used for "
    "feature/model selection."
)
