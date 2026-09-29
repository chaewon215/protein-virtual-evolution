from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


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
    / "mlp_random5"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


FOLD_COLUMN = "fold_random_5"

SEED = 42

VALIDATION_RATIO = 0.10

BATCH_SIZE = 128

MAX_EPOCHS = 300

PATIENCE = 20

MIN_DELTA = 1e-5

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

DROPOUT = 0.2


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Model
# ============================================================

class MLPRegressor(nn.Module):

    def __init__(
        self,
        input_dim,
        dropout=0.2,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                512,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                512,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                1,
            ),
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    X,
    y,
    batch_size,
    shuffle,
    seed,
):

    X_tensor = torch.from_numpy(
        X.astype(
            np.float32
        )
    )

    y_tensor = torch.from_numpy(
        y.astype(
            np.float32
        )
    ).unsqueeze(1)


    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0

    total_samples = 0


    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        predictions = model(
            X_batch
        )


        loss = criterion(
            predictions,
            y_batch,
        )


        loss.backward()

        optimizer.step()


        batch_size = (
            X_batch.shape[0]
        )


        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )


    return (
        total_loss
        / total_samples
    )


# ============================================================
# Prediction
# ============================================================

def predict(
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
        X_tensor,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    predictions = []


    with torch.no_grad():

        for X_batch in loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True,
            )


            output = model(
                X_batch
            )


            predictions.append(
                output
                .squeeze(1)
                .cpu()
                .numpy()
            )


    return np.concatenate(
        predictions
    )


# ============================================================
# Early stopping
# ============================================================

def select_best_epoch(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    seed,
):

    set_seed(
        seed
    )


    model = MLPRegressor(
        input_dim=input_dim,
        dropout=DROPOUT,
    ).to(
        DEVICE
    )


    criterion = nn.MSELoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )


    best_val_loss = np.inf

    best_epoch = 1

    epochs_without_improvement = 0

    history = []


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


        val_predictions = predict(
            model,
            X_val,
        )


        val_loss = mean_squared_error(
            y_val,
            val_predictions,
        )


        history.append(
            {
                "epoch": epoch,
                "train_mse": float(
                    train_loss
                ),
                "val_mse": float(
                    val_loss
                ),
            }
        )


        if (
            val_loss
            < best_val_loss
            - MIN_DELTA
        ):

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            break


    return (
        best_epoch,
        best_val_loss,
        pd.DataFrame(
            history
        ),
    )


# ============================================================
# Train final model for fixed epochs
# ============================================================

def train_fixed_epochs(
    X_train,
    y_train,
    input_dim,
    num_epochs,
    seed,
):

    set_seed(
        seed
    )


    model = MLPRegressor(
        input_dim=input_dim,
        dropout=DROPOUT,
    ).to(
        DEVICE
    )


    criterion = nn.MSELoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )


    for epoch in range(
        1,
        num_epochs + 1,
    ):

        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )


    return model


# ============================================================
# Load dataset
# ============================================================

print(
    "Device:",
    DEVICE,
)

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
)


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
)


print(
    "Dataset/cache alignment: OK"
)


# ============================================================
# Construct feature
# ============================================================

wt_residue = (
    wt_cache[
        "residue_embeddings"
    ]
    .numpy()
)


mutant_local = (
    mut_cache[
        "mutant_local"
    ]
    .numpy()
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
# Model summary
# ============================================================

model_summary = MLPRegressor(
    input_dim=X.shape[1],
    dropout=DROPOUT,
)


num_parameters = sum(
    p.numel()
    for p in model_summary.parameters()
    if p.requires_grad
)


print()
print(
    "=== MLP Architecture ==="
)

print(
    model_summary
)

print(
    "Trainable parameters:",
    f"{num_parameters:,}",
)


# ============================================================
# Outer CV
# ============================================================

folds = (
    df[FOLD_COLUMN]
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


print()
print(
    "=== MLP Random 5-Fold CV ==="
)


for fold in unique_folds:

    fold_seed = (
        SEED + fold
    )


    outer_train_mask = (
        folds != fold
    )

    outer_test_mask = (
        folds == fold
    )


    X_outer_train = X[
        outer_train_mask
    ]

    X_test = X[
        outer_test_mask
    ]

    y_outer_train = y[
        outer_train_mask
    ]

    y_test = y[
        outer_test_mask
    ]


    # --------------------------------------------------------
    # Inner train / validation split
    #
    # Used only to decide the number of training epochs.
    # Outer test is never used here.
    # --------------------------------------------------------

    (
        X_inner_train_raw,
        X_val_raw,
        y_inner_train,
        y_val,
    ) = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=VALIDATION_RATIO,
        random_state=fold_seed,
    )


    # --------------------------------------------------------
    # Inner scaler
    #
    # Fit ONLY on inner train.
    # --------------------------------------------------------

    inner_scaler = StandardScaler()


    X_inner_train = (
        inner_scaler
        .fit_transform(
            X_inner_train_raw
        )
        .astype(
            np.float32
        )
    )


    X_val = (
        inner_scaler
        .transform(
            X_val_raw
        )
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # Select best epoch
    # --------------------------------------------------------

    (
        best_epoch,
        best_val_loss,
        history_df,
    ) = select_best_epoch(
        X_inner_train,
        y_inner_train,
        X_val,
        y_val,
        input_dim=X.shape[1],
        seed=fold_seed,
    )


    history_df.to_csv(
        RESULT_ROOT
        / f"history_fold_{fold}.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Final scaler
    #
    # Now fit on ALL outer train samples.
    # --------------------------------------------------------

    final_scaler = StandardScaler()


    X_outer_train_scaled = (
        final_scaler
        .fit_transform(
            X_outer_train
        )
        .astype(
            np.float32
        )
    )


    X_test_scaled = (
        final_scaler
        .transform(
            X_test
        )
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # Final model
    #
    # Retrain from scratch on all outer train samples
    # for the selected number of epochs.
    # --------------------------------------------------------

    model = train_fixed_epochs(
        X_outer_train_scaled,
        y_outer_train,
        input_dim=X.shape[1],
        num_epochs=best_epoch,
        seed=fold_seed,
    )


    predictions = predict(
        model,
        X_test_scaled,
    )


    oof_predictions[
        outer_test_mask
    ] = predictions


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    spearman, p_value = spearmanr(
        y_test,
        predictions,
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
            "fold": int(
                fold
            ),

            "n_train": int(
                len(
                    y_outer_train
                )
            ),

            "n_test": int(
                len(
                    y_test
                )
            ),

            "best_epoch": int(
                best_epoch
            ),

            "best_val_mse": float(
                best_val_loss
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


    print()
    print(
        f"Fold {fold}"
    )

    print(
        f"Train: "
        f"{len(y_outer_train)} | "
        f"Test: {len(y_test)}"
    )

    print(
        f"Best epoch : "
        f"{best_epoch}"
    )

    print(
        f"Val MSE    : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Spearman   : "
        f"{spearman:.4f}"
    )

    print(
        f"RMSE       : "
        f"{rmse:.4f}"
    )

    print(
        f"R²         : "
        f"{r2:.4f}"
    )


# ============================================================
# Validate OOF predictions
# ============================================================

assert np.isfinite(
    oof_predictions
).all()


print()
print(
    "OOF prediction coverage: OK"
)


# ============================================================
# Summary
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)


metric_columns = [
    "spearman",
    "rmse",
    "r2",
]


mean_metrics = (
    fold_df[
        metric_columns
    ]
    .mean()
)


std_metrics = (
    fold_df[
        metric_columns
    ]
    .std(
        ddof=1
    )
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
print(
    "=" * 60
)

print(
    "MLP Random 5-Fold Summary"
)

print(
    "=" * 60
)


print(
    "Spearman : "
    f"{mean_metrics['spearman']:.4f} "
    f"± "
    f"{std_metrics['spearman']:.4f}"
)


print(
    "RMSE     : "
    f"{mean_metrics['rmse']:.4f} "
    f"± "
    f"{std_metrics['rmse']:.4f}"
)


print(
    "R²       : "
    f"{mean_metrics['r2']:.4f} "
    f"± "
    f"{std_metrics['r2']:.4f}"
)


print()
print(
    "=== Global OOF ==="
)


print(
    f"Spearman : "
    f"{oof_spearman:.4f}"
)


print(
    f"RMSE     : "
    f"{oof_rmse:.4f}"
)


print(
    f"R²       : "
    f"{oof_r2:.4f}"
)


# ============================================================
# Save fold metrics
# ============================================================

fold_df.to_csv(
    RESULT_ROOT
    / "fold_metrics.csv",
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
    / "oof_predictions.csv",
    index=False,
)


# ============================================================
# Save summary
# ============================================================

summary = {

    "model": "MLP",

    "input_feature": (
        "wt_local + delta_local"
    ),

    "input_dim": 1920,

    "hidden_dims": [
        512,
        128,
    ],

    "output_dim": 1,

    "activation": "ReLU",

    "dropout": DROPOUT,

    "optimizer": "AdamW",

    "learning_rate": (
        LEARNING_RATE
    ),

    "weight_decay": (
        WEIGHT_DECAY
    ),

    "batch_size": (
        BATCH_SIZE
    ),

    "max_epochs": (
        MAX_EPOCHS
    ),

    "patience": (
        PATIENCE
    ),

    "seed": SEED,

    "trainable_parameters": int(
        num_parameters
    ),

    "split": (
        FOLD_COLUMN
    ),

    "fold_mean": {

        "spearman": float(
            mean_metrics[
                "spearman"
            ]
        ),

        "rmse": float(
            mean_metrics[
                "rmse"
            ]
        ),

        "r2": float(
            mean_metrics[
                "r2"
            ]
        ),
    },

    "fold_std": {

        "spearman": float(
            std_metrics[
                "spearman"
            ]
        ),

        "rmse": float(
            std_metrics[
                "rmse"
            ]
        ),

        "r2": float(
            std_metrics[
                "r2"
            ]
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
print(
    "Saved to:",
    RESULT_ROOT,
)
