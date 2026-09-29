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
    roc_auc_score,
)
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
    / "deep_ensemble_uncertainty"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# Split -> ProteinGym fold column
SPLITS = {
    "random": "fold_random_5",
    "modulo": "fold_modulo_5",
    "contiguous": "fold_contiguous_5",
}


# We reuse the leakage-free epoch choices already obtained
# in Step 5 / Step 6 rather than selecting epochs again.
EPOCH_RESULT_PATHS = {
    "random": (
        PROJECT_ROOT
        / "results"
        / "tem1"
        / "mlp_random5"
        / "fold_metrics.csv"
    ),

    "modulo": (
        PROJECT_ROOT
        / "results"
        / "tem1"
        / "mlp_generalization_robust"
        / "modulo_fold_metrics.csv"
    ),

    "contiguous": (
        PROJECT_ROOT
        / "results"
        / "tem1"
        / "mlp_generalization_robust"
        / "contiguous_fold_metrics.csv"
    ),
}


# Five independently initialized MLPs per outer fold.
ENSEMBLE_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

DROPOUT = 0.2


# High-error detection metric:
# errors in the top 20% are labeled as "high error".
HIGH_ERROR_QUANTILE = 0.80


# Coverage values for selective-risk analysis.
# Example: coverage=0.2 means keep only the 20% lowest-uncertainty samples.
RISK_COVERAGES = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]


SAVE_MODELS = False


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# MLP
#
# Input:
#   [B, 1920]
#
# Hidden:
#   [B, 512]
#   [B, 128]
#
# Output:
#   [B, 1]
# ============================================================

class MLPRegressor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.2,
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
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:

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
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:

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


        current_batch_size = (
            X_batch.shape[0]
        )


        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_samples += (
            current_batch_size
        )


    return (
        total_loss
        / total_samples
    )


# ============================================================
# Fixed-epoch training
#
# Epoch count comes from the previous leakage-free
# model-selection experiment.
# ============================================================

def train_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    num_epochs: int,
    seed: int,
) -> nn.Module:

    set_seed(seed)


    model = MLPRegressor(
        input_dim=input_dim,
        dropout=DROPOUT,
    ).to(DEVICE)


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


    for _ in range(
        num_epochs
    ):

        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )


    return model


# ============================================================
# Prediction
# ============================================================

def predict(
    model: nn.Module,
    X: np.ndarray,
) -> np.ndarray:

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
# Load leakage-free epoch mapping
# ============================================================

def load_epoch_mapping(
    split_name: str,
) -> dict:

    path = EPOCH_RESULT_PATHS[
        split_name
    ]


    if not path.exists():

        raise FileNotFoundError(
            f"Epoch result file not found: {path}"
        )


    epoch_df = pd.read_csv(
        path
    )


    if split_name == "random":

        fold_column = "fold"
        epoch_column = "best_epoch"

    else:

        fold_column = "test_fold"
        epoch_column = "final_epoch"


    required = {
        fold_column,
        epoch_column,
    }


    missing = (
        required
        - set(
            epoch_df.columns
        )
    )


    if missing:

        raise ValueError(
            f"{path} is missing columns: {missing}"
        )


    mapping = {
        int(row[fold_column]):
        int(row[epoch_column])

        for _, row
        in epoch_df.iterrows()
    }


    expected_folds = {
        0,
        1,
        2,
        3,
        4,
    }


    if set(mapping) != expected_folds:

        raise ValueError(
            f"Unexpected fold mapping in {path}: {mapping}"
        )


    return mapping


# ============================================================
# Regression metrics
# ============================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    spearman, spearman_p = (
        spearmanr(
            y_true,
            y_pred,
        )
    )


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
        "spearman":
            float(
                spearman
            ),

        "spearman_p":
            float(
                spearman_p
            ),

        "rmse":
            float(
                rmse
            ),

        "mae":
            float(
                mae
            ),

        "r2":
            float(
                r2
            ),
    }


# ============================================================
# Uncertainty quality metrics
#
# Deep Ensemble:
#
# member predictions:
#   [B, M]
#
# ensemble mean:
#   [B]
#
# ensemble std:
#   [B]
#
# Here M = 5.
#
# The std captures model disagreement / epistemic uncertainty.
# It is NOT automatically a calibrated total predictive
# standard deviation.
# ============================================================

def uncertainty_metrics(
    y_true: np.ndarray,
    mean_prediction: np.ndarray,
    uncertainty_std: np.ndarray,
) -> dict:

    absolute_error = np.abs(
        mean_prediction
        - y_true
    )


    uncertainty_error_spearman, p_value = (
        spearmanr(
            uncertainty_std,
            absolute_error,
        )
    )


    # --------------------------------------------
    # High-error detection
    #
    # Label top 20% absolute errors as positive.
    # If uncertainty is meaningful, it should rank
    # these samples higher.
    # --------------------------------------------

    threshold = np.quantile(
        absolute_error,
        HIGH_ERROR_QUANTILE,
    )


    high_error_label = (
        absolute_error
        >= threshold
    ).astype(
        np.int64
    )


    if (
        np.unique(
            high_error_label
        ).size
        == 2
    ):

        high_error_auroc = (
            roc_auc_score(
                high_error_label,
                uncertainty_std,
            )
        )

    else:

        high_error_auroc = np.nan


    return {
        "uncertainty_error_spearman":
            float(
                uncertainty_error_spearman
            ),

        "uncertainty_error_spearman_p":
            float(
                p_value
            ),

        "high_error_threshold":
            float(
                threshold
            ),

        "high_error_auroc":
            float(
                high_error_auroc
            ),
    }


# ============================================================
# Uncertainty quintiles
#
# Q1 = lowest uncertainty
# Q5 = highest uncertainty
#
# If uncertainty is informative:
#   Q5 error > Q1 error
# ============================================================

def make_uncertainty_quintile_table(
    y_true: np.ndarray,
    mean_prediction: np.ndarray,
    uncertainty_std: np.ndarray,
) -> pd.DataFrame:

    df_q = pd.DataFrame(
        {
            "y_true":
                y_true,

            "prediction":
                mean_prediction,

            "uncertainty":
                uncertainty_std,
        }
    )


    df_q[
        "absolute_error"
    ] = np.abs(
        df_q[
            "prediction"
        ]
        - df_q[
            "y_true"
        ]
    )


    # rank(method="first") avoids qcut failures
    # when uncertainty contains repeated values.
    df_q[
        "uncertainty_quintile"
    ] = pd.qcut(
        df_q[
            "uncertainty"
        ].rank(
            method="first"
        ),
        q=5,
        labels=[
            "Q1_lowest",
            "Q2",
            "Q3",
            "Q4",
            "Q5_highest",
        ],
    )


    table = (
        df_q
        .groupby(
            "uncertainty_quintile",
            observed=True,
        )
        .agg(
            n=(
                "absolute_error",
                "size",
            ),

            mean_uncertainty=(
                "uncertainty",
                "mean",
            ),

            mean_absolute_error=(
                "absolute_error",
                "mean",
            ),

            median_absolute_error=(
                "absolute_error",
                "median",
            ),
        )
        .reset_index()
    )


    return table


# ============================================================
# Selective risk
#
# Sort samples from low uncertainty -> high uncertainty.
#
# coverage = 0.20:
#   keep the 20% most certain samples
#
# If uncertainty is useful:
#   error should increase as coverage increases.
# ============================================================

def make_risk_coverage_table(
    y_true: np.ndarray,
    mean_prediction: np.ndarray,
    uncertainty_std: np.ndarray,
) -> pd.DataFrame:

    order = np.argsort(
        uncertainty_std
    )


    y_sorted = y_true[
        order
    ]

    pred_sorted = mean_prediction[
        order
    ]

    unc_sorted = uncertainty_std[
        order
    ]


    rows = []


    n_total = len(
        y_true
    )


    for coverage in RISK_COVERAGES:

        n_keep = max(
            1,
            int(
                np.floor(
                    n_total
                    * coverage
                )
            ),
        )


        y_keep = y_sorted[
            :n_keep
        ]

        pred_keep = pred_sorted[
            :n_keep
        ]

        unc_keep = unc_sorted[
            :n_keep
        ]


        rows.append(
            {
                "coverage":
                    float(
                        coverage
                    ),

                "n_samples":
                    int(
                        n_keep
                    ),

                "max_uncertainty_kept":
                    float(
                        np.max(
                            unc_keep
                        )
                    ),

                "mae":
                    float(
                        mean_absolute_error(
                            y_keep,
                            pred_keep,
                        )
                    ),

                "rmse":
                    float(
                        np.sqrt(
                            mean_squared_error(
                                y_keep,
                                pred_keep,
                            )
                        )
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Load data
# ============================================================

print(
    "Device:",
    DEVICE,
)

print(
    "Ensemble size:",
    len(
        ENSEMBLE_SEEDS
    ),
)

print(
    "Ensemble seeds:",
    ENSEMBLE_SEEDS,
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
# Alignment validation
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
# Feature construction
#
# Raw protein input:
#   WT sequence      : 286 aa
#   Mutant sequence  : 286 aa
#
# ESM-C representation:
#   WT residue embeddings : [286, 960]
#   Mutant local          : [4996, 960]
#
# For each mutant:
#   WT local               : [960]
#   Delta local            : [960]
#
# Concatenate:
#   X_i                    : [1920]
#
# Entire dataset:
#   X                      : [4996, 1920]
#   y                      : [4996]
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
    df[
        "DMS_score"
    ]
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
# Model / Ensemble summary
# ============================================================

summary_model = MLPRegressor(
    input_dim=X.shape[1],
    dropout=DROPOUT,
)


num_parameters = sum(
    p.numel()
    for p in summary_model.parameters()
    if p.requires_grad
)


print()
print(
    "=== Deep Ensemble Architecture ==="
)

print(
    "Single member:"
)

print(
    "  Input        : [B, 1920]"
)

print(
    "  Linear 1     : [B, 1920] -> [B, 512]"
)

print(
    "  ReLU/Dropout : [B, 512] -> [B, 512]"
)

print(
    "  Linear 2     : [B, 512] -> [B, 128]"
)

print(
    "  ReLU/Dropout : [B, 128] -> [B, 128]"
)

print(
    "  Output       : [B, 128] -> [B, 1]"
)

print(
    "  Parameters   :",
    f"{num_parameters:,}",
)

print()
print(
    "Ensemble:"
)

print(
    f"  Members      : {len(ENSEMBLE_SEEDS)}"
)

print(
    f"  Predictions  : [B, {len(ENSEMBLE_SEEDS)}]"
)

print(
    "  Mean         : [B]"
)

print(
    "  Std          : [B]"
)


# ============================================================
# Main evaluation
# ============================================================

all_split_summaries = []


for split_name, fold_column in SPLITS.items():

    print()
    print(
        "=" * 100
    )

    print(
        f"Split: {split_name}"
    )

    print(
        f"Fold column: {fold_column}"
    )

    print(
        "=" * 100
    )


    epoch_mapping = load_epoch_mapping(
        split_name
    )


    print(
        "Epoch mapping:",
        epoch_mapping,
    )


    folds = (
        df[
            fold_column
        ]
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


    # OOF arrays
    oof_mean = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )

    oof_std = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )


    # Store each member's OOF prediction separately.
    oof_members = np.full(
        (
            len(df),
            len(
                ENSEMBLE_SEEDS
            ),
        ),
        np.nan,
        dtype=np.float64,
    )


    fold_results = []


    model_root = (
        RESULT_ROOT
        / "models"
        / split_name
    )


    if SAVE_MODELS:

        model_root.mkdir(
            parents=True,
            exist_ok=True,
        )


    for test_fold in unique_folds:

        print()
        print(
            "-" * 80
        )

        print(
            f"Test fold: {test_fold}"
        )

        print(
            "-" * 80
        )


        train_mask = (
            folds != test_fold
        )

        test_mask = (
            folds == test_fold
        )


        X_train_raw = X[
            train_mask
        ]

        X_test_raw = X[
            test_mask
        ]

        y_train = y[
            train_mask
        ]

        y_test = y[
            test_mask
        ]


        # --------------------------------------------
        # Scaler is fit ONLY on the outer training fold.
        # Same train statistics are used for every
        # ensemble member.
        # --------------------------------------------

        scaler = StandardScaler()


        X_train = (
            scaler
            .fit_transform(
                X_train_raw
            )
            .astype(
                np.float32
            )
        )


        X_test = (
            scaler
            .transform(
                X_test_raw
            )
            .astype(
                np.float32
            )
        )


        num_epochs = epoch_mapping[
            test_fold
        ]


        print(
            f"Train       : {train_mask.sum()}"
        )

        print(
            f"Test        : {test_mask.sum()}"
        )

        print(
            f"Epochs      : {num_epochs}"
        )


        member_predictions = []


        # ====================================================
        # Train independent ensemble members
        # ====================================================

        for member_index, base_seed in enumerate(
            ENSEMBLE_SEEDS
        ):

            # Fold offset guarantees deterministic but
            # different initialization/shuffle per fold.
            member_seed = (
                base_seed
                + test_fold * 100
            )


            model = train_fixed_epochs(
                X_train,
                y_train,
                input_dim=X.shape[1],
                num_epochs=num_epochs,
                seed=member_seed,
            )


            predictions = predict(
                model,
                X_test,
            )


            member_predictions.append(
                predictions
            )


            print(
                f"  Member {member_index + 1}"
                f" | seed={member_seed}"
                f" | prediction mean="
                f"{predictions.mean():.4f}"
            )


            if SAVE_MODELS:

                torch.save(
                    {
                        "state_dict":
                            model.state_dict(),

                        "seed":
                            int(
                                member_seed
                            ),

                        "split":
                            split_name,

                        "test_fold":
                            int(
                                test_fold
                            ),

                        "epochs":
                            int(
                                num_epochs
                            ),

                        "input_dim":
                            1920,
                    },
                    model_root
                    / (
                        f"fold_{test_fold}"
                        f"_member_{member_index}"
                        f"_seed_{member_seed}.pt"
                    ),
                )


        # Shape:
        #   [n_test, ensemble_size]
        member_prediction_matrix = (
            np.stack(
                member_predictions,
                axis=1,
            )
        )


        assert (
            member_prediction_matrix.shape
            == (
                test_mask.sum(),
                len(
                    ENSEMBLE_SEEDS
                ),
            )
        )


        # Deep Ensemble predictive mean.
        ensemble_mean = (
            member_prediction_matrix
            .mean(
                axis=1
            )
        )


        # Model disagreement / epistemic uncertainty.
        # ddof=1 gives sample standard deviation
        # across the five members.
        ensemble_std = (
            member_prediction_matrix
            .std(
                axis=1,
                ddof=1,
            )
        )


        # Save into global OOF arrays.
        oof_mean[
            test_mask
        ] = ensemble_mean


        oof_std[
            test_mask
        ] = ensemble_std


        oof_members[
            test_mask,
            :
        ] = member_prediction_matrix


        # ====================================================
        # Fold-level metrics
        # ====================================================

        pred_metrics = regression_metrics(
            y_test,
            ensemble_mean,
        )


        unc_metrics = uncertainty_metrics(
            y_test,
            ensemble_mean,
            ensemble_std,
        )


        fold_result = {

            "split":
                split_name,

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

            "epochs":
                int(
                    num_epochs
                ),

            "ensemble_size":
                int(
                    len(
                        ENSEMBLE_SEEDS
                    )
                ),

            "mean_uncertainty":
                float(
                    np.mean(
                        ensemble_std
                    )
                ),

            "median_uncertainty":
                float(
                    np.median(
                        ensemble_std
                    )
                ),

            **pred_metrics,
            **unc_metrics,
        }


        fold_results.append(
            fold_result
        )


        print()
        print(
            f"Ensemble mean Spearman : "
            f"{pred_metrics['spearman']:.4f}"
        )

        print(
            f"Ensemble mean RMSE     : "
            f"{pred_metrics['rmse']:.4f}"
        )

        print(
            f"Ensemble mean R²       : "
            f"{pred_metrics['r2']:.4f}"
        )

        print(
            f"Mean uncertainty       : "
            f"{np.mean(ensemble_std):.4f}"
        )

        print(
            f"Uncertainty-error rho  : "
            f"{unc_metrics['uncertainty_error_spearman']:.4f}"
        )

        print(
            f"High-error AUROC       : "
            f"{unc_metrics['high_error_auroc']:.4f}"
        )


    # ========================================================
    # OOF validation
    # ========================================================

    assert np.isfinite(
        oof_mean
    ).all()


    assert np.isfinite(
        oof_std
    ).all()


    assert np.isfinite(
        oof_members
    ).all()


    # ========================================================
    # Global OOF metrics
    # ========================================================

    global_pred_metrics = regression_metrics(
        y,
        oof_mean,
    )


    global_unc_metrics = uncertainty_metrics(
        y,
        oof_mean,
        oof_std,
    )


    fold_df = pd.DataFrame(
        fold_results
    )


    print()
    print(
        "=" * 80
    )

    print(
        f"{split_name.upper()} Deep Ensemble Summary"
    )

    print(
        "=" * 80
    )


    print(
        "Prediction performance"
    )

    print(
        "Spearman : "
        f"{global_pred_metrics['spearman']:.4f}"
    )

    print(
        "RMSE     : "
        f"{global_pred_metrics['rmse']:.4f}"
    )

    print(
        "MAE      : "
        f"{global_pred_metrics['mae']:.4f}"
    )

    print(
        "R²       : "
        f"{global_pred_metrics['r2']:.4f}"
    )


    print()
    print(
        "Uncertainty quality"
    )

    print(
        "Uncertainty-error Spearman : "
        f"{global_unc_metrics['uncertainty_error_spearman']:.4f}"
    )

    print(
        "High-error AUROC           : "
        f"{global_unc_metrics['high_error_auroc']:.4f}"
    )

    print(
        "Mean uncertainty           : "
        f"{np.mean(oof_std):.4f}"
    )


    # ========================================================
    # OOF prediction table
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


    for member_index, base_seed in enumerate(
        ENSEMBLE_SEEDS
    ):

        oof_df[
            f"member_{member_index + 1}_prediction"
        ] = oof_members[
            :,
            member_index
        ]


    oof_df[
        "ensemble_mean"
    ] = oof_mean


    oof_df[
        "ensemble_std"
    ] = oof_std


    oof_df[
        "error"
    ] = (
        oof_mean
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
    # Fold metrics
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
    # Uncertainty quintiles
    # ========================================================

    quintile_df = (
        make_uncertainty_quintile_table(
            y,
            oof_mean,
            oof_std,
        )
    )


    quintile_df.to_csv(
        RESULT_ROOT
        / (
            f"{split_name}"
            f"_uncertainty_quintiles.csv"
        ),
        index=False,
    )


    print()
    print(
        "=== Error by uncertainty quintile ==="
    )

    print(
        quintile_df.to_string(
            index=False
        )
    )


    # ========================================================
    # Selective risk / risk-coverage
    # ========================================================

    risk_coverage_df = (
        make_risk_coverage_table(
            y,
            oof_mean,
            oof_std,
        )
    )


    risk_coverage_df.to_csv(
        RESULT_ROOT
        / (
            f"{split_name}"
            f"_risk_coverage.csv"
        ),
        index=False,
    )


    print()
    print(
        "=== Selective Risk ==="
    )

    print(
        risk_coverage_df.to_string(
            index=False
        )
    )


    # ========================================================
    # Split summary
    # ========================================================

    split_summary = {

        "split":
            split_name,

        "fold_column":
            fold_column,

        "model":
            "Deep Ensemble MLP",

        "input_feature":
            "wt_local + delta_local",

        "input_dim":
            1920,

        "single_member_architecture":
            [
                1920,
                512,
                128,
                1,
            ],

        "single_member_parameters":
            int(
                num_parameters
            ),

        "ensemble_size":
            int(
                len(
                    ENSEMBLE_SEEDS
                )
            ),

        "ensemble_seeds":
            ENSEMBLE_SEEDS,

        "uncertainty_definition":
            (
                "sample standard deviation "
                "across ensemble member predictions"
            ),

        "epoch_source":
            str(
                EPOCH_RESULT_PATHS[
                    split_name
                ]
            ),

        "prediction":
            global_pred_metrics,

        "uncertainty":
            {
                **global_unc_metrics,

                "mean":
                    float(
                        np.mean(
                            oof_std
                        )
                    ),

                "median":
                    float(
                        np.median(
                            oof_std
                        )
                    ),
            },
    }


    with open(
        RESULT_ROOT
        / (
            f"{split_name}"
            f"_summary.json"
        ),
        "w",
    ) as f:

        json.dump(
            split_summary,
            f,
            indent=2,
        )


    all_split_summaries.append(
        {
            "split":
                split_name,

            "spearman":
                global_pred_metrics[
                    "spearman"
                ],

            "rmse":
                global_pred_metrics[
                    "rmse"
                ],

            "mae":
                global_pred_metrics[
                    "mae"
                ],

            "r2":
                global_pred_metrics[
                    "r2"
                ],

            "mean_uncertainty":
                float(
                    np.mean(
                        oof_std
                    )
                ),

            "uncertainty_error_spearman":
                global_unc_metrics[
                    "uncertainty_error_spearman"
                ],

            "high_error_auroc":
                global_unc_metrics[
                    "high_error_auroc"
                ],
        }
    )


# ============================================================
# Final summary
# ============================================================

summary_df = pd.DataFrame(
    all_split_summaries
)


summary_df.to_csv(
    RESULT_ROOT
    / "uncertainty_summary.csv",
    index=False,
)


with open(
    RESULT_ROOT
    / "uncertainty_summary.json",
    "w",
) as f:

    json.dump(
        all_split_summaries,
        f,
        indent=2,
    )


print()
print(
    "=" * 110
)

print(
    "Deep Ensemble Uncertainty Summary"
)

print(
    "=" * 110
)


print(
    summary_df.to_string(
        index=False
    )
)


print()
print(
    "Saved to:",
    RESULT_ROOT,
)
