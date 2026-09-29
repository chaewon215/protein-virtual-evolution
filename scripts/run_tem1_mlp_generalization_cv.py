from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tem1_single_mutants.csv"
CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "esmc300m"
WT_CACHE_PATH = CACHE_ROOT / "wt_embeddings.pt"
MUT_CACHE_PATH = CACHE_ROOT / "tem1_mutant_embeddings.pt"

RESULT_ROOT = PROJECT_ROOT / "results" / "tem1" / "mlp_generalization_robust"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "modulo": "fold_modulo_5",
    "contiguous": "fold_contiguous_5",
}

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20
MIN_DELTA = 1e-5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Model
#
# Input : [B, 1920]
#   - WT local    : [B, 960]
#   - Delta local : [B, 960]
#
# MLP:
#   [B, 1920]
#       -> Linear(1920, 512)
#       -> ReLU
#       -> Dropout(0.2)
#       -> Linear(512, 128)
#       -> ReLU
#       -> Dropout(0.2)
#       -> Linear(128, 1)
#
# Output: predicted DMS_score [B, 1]
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


# ============================================================
# Training helpers
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
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        current_batch_size = X_batch.shape[0]
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

    return total_loss / total_samples


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()

    X_tensor = torch.from_numpy(X.astype(np.float32))

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
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            output = model(X_batch)
            predictions.append(output.squeeze(1).cpu().numpy())

    return np.concatenate(predictions)


def select_best_epoch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    seed: int,
):
    """Select best epoch using one structured inner-validation fold."""

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

    best_val_loss = np.inf
    best_epoch = 1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_predictions = predict(model, X_val)
        val_loss = mean_squared_error(y_val, val_predictions)

        history.append(
            {
                "epoch": epoch,
                "train_mse": float(train_loss),
                "val_mse": float(val_loss),
            }
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    return best_epoch, best_val_loss, pd.DataFrame(history)


def train_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    num_epochs: int,
    seed: int,
) -> nn.Module:
    """Retrain from scratch on the full outer-training set."""

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

    for _ in range(num_epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

    return model


def rounded_median_epoch(epochs) -> int:
    """Return positive median epoch with conventional half-up rounding."""

    median_epoch = np.median(np.asarray(epochs, dtype=np.float64))
    return max(1, int(np.floor(median_epoch + 0.5)))


# ============================================================
# Load dataset
# ============================================================

print("Device:", DEVICE)

df = pd.read_csv(DATA_PATH)
print("Samples:", len(df))


# ============================================================
# Load cached ESM-C embeddings
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
# Validate cache/data alignment
# ============================================================

assert df["mutant"].tolist() == mut_cache["mutants"], (
    "Dataset/cache mutant order mismatch."
)

positions = df["position"].to_numpy(dtype=np.int64)
cache_positions = mut_cache["positions"].numpy()

assert np.array_equal(positions, cache_positions), (
    "Dataset/cache position mismatch."
)

print("Dataset/cache alignment: OK")


# ============================================================
# Build supervised features
#
# Raw sequence level:
#   WT sequence     : 286 aa
#   Mutant sequence : 286 aa
#
# Frozen ESM-C representation:
#   WT residue embedding matrix : [286, 960]
#   Mutant local cache           : [4996, 960]
#
# Feature per mutation:
#   WT local    : [960]
#   Delta local : mutant_local - wt_local = [960]
#   Concatenate : [1920]
#
# Full dataset:
#   X : [4996, 1920]
#   y : [4996]
# ============================================================

wt_residue = wt_cache["residue_embeddings"].numpy()
mutant_local = mut_cache["mutant_local"].numpy()

wt_local = wt_residue[positions - 1]
delta_local = mutant_local - wt_local

X = np.concatenate(
    [wt_local, delta_local],
    axis=1,
)

y = df["DMS_score"].to_numpy(dtype=np.float64)

print()
print("=== Feature Matrix ===")
print("WT local    :", wt_local.shape)
print("Delta local :", delta_local.shape)
print("X           :", X.shape)
print("y           :", y.shape)

assert X.shape == (len(df), 1920)
assert np.isfinite(X).all()
assert np.isfinite(y).all()


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
print("=== MLP Architecture ===")
print(model_summary)
print("Trainable parameters:", f"{num_parameters:,}")


# ============================================================
# Robust generalization evaluation
#
# Key change from the previous script:
# For each OUTER test fold, all remaining 4 folds are used once
# as structured validation folds.
#
# Example: test fold = 0
#   inner val 1 -> inner train = 2,3,4
#   inner val 2 -> inner train = 1,3,4
#   inner val 3 -> inner train = 1,2,4
#   inner val 4 -> inner train = 1,2,3
#
# Each inner run selects one best epoch.
# Final epoch = median(inner best epochs).
#
# Then:
#   final outer train = all folds except test fold
#   final outer test  = test fold
#
# Test labels never participate in:
#   - StandardScaler fitting
#   - epoch selection
#   - final model training
# ============================================================

all_summary_results = []

for split_name, fold_column in SPLITS.items():
    print()
    print("=" * 90)
    print(f"Split: {split_name}")
    print(f"Fold column: {fold_column}")
    print("=" * 90)

    folds = df[fold_column].to_numpy(dtype=np.int64)
    unique_folds = sorted(np.unique(folds).tolist())

    assert unique_folds == [0, 1, 2, 3, 4]

    oof_predictions = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )

    fold_results = []

    split_history_root = RESULT_ROOT / split_name / "histories"
    split_history_root.mkdir(parents=True, exist_ok=True)

    for test_fold in unique_folds:
        print()
        print("-" * 70)
        print(f"Outer test fold: {test_fold}")
        print("-" * 70)

        fold_seed = SEED + test_fold

        outer_test_mask = folds == test_fold
        outer_train_mask = folds != test_fold

        candidate_val_folds = [
            fold
            for fold in unique_folds
            if fold != test_fold
        ]

        inner_best_epochs = []
        inner_best_val_losses = []

        # ====================================================
        # Robust structured inner-CV epoch selection
        # ====================================================
        for val_fold in candidate_val_folds:
            inner_train_mask = (
                (folds != test_fold)
                & (folds != val_fold)
            )

            inner_val_mask = folds == val_fold

            X_inner_train_raw = X[inner_train_mask]
            y_inner_train = y[inner_train_mask]

            X_val_raw = X[inner_val_mask]
            y_val = y[inner_val_mask]

            # Fit preprocessing ONLY on inner train.
            inner_scaler = StandardScaler()

            X_inner_train = (
                inner_scaler
                .fit_transform(X_inner_train_raw)
                .astype(np.float32)
            )

            X_val = (
                inner_scaler
                .transform(X_val_raw)
                .astype(np.float32)
            )

            # Deterministic but distinct seed for each inner run.
            inner_seed = (
                SEED
                + test_fold * 10
                + val_fold
            )

            best_epoch, best_val_loss, history_df = select_best_epoch(
                X_inner_train,
                y_inner_train,
                X_val,
                y_val,
                input_dim=X.shape[1],
                seed=inner_seed,
            )

            inner_best_epochs.append(int(best_epoch))
            inner_best_val_losses.append(float(best_val_loss))

            history_path = (
                split_history_root
                / f"test_{test_fold}_val_{val_fold}.csv"
            )

            history_df.to_csv(
                history_path,
                index=False,
            )

            print(
                f"  Validation fold {val_fold}"
                f" | inner train={inner_train_mask.sum()}"
                f" | val={inner_val_mask.sum()}"
                f" | best epoch={best_epoch}"
                f" | best val MSE={best_val_loss:.4f}"
            )

        # ====================================================
        # Aggregate epoch choice across 4 inner folds
        # ====================================================
        final_epoch = rounded_median_epoch(inner_best_epochs)

        print()
        print("Inner best epochs :", inner_best_epochs)
        print("Final epoch       :", final_epoch)

        # ====================================================
        # Final preprocessing on ALL outer-training samples
        # ====================================================
        X_outer_train_raw = X[outer_train_mask]
        y_outer_train = y[outer_train_mask]

        X_test_raw = X[outer_test_mask]
        y_test = y[outer_test_mask]

        final_scaler = StandardScaler()

        X_outer_train = (
            final_scaler
            .fit_transform(X_outer_train_raw)
            .astype(np.float32)
        )

        X_test = (
            final_scaler
            .transform(X_test_raw)
            .astype(np.float32)
        )

        # ====================================================
        # Retrain from scratch on complete outer train
        #
        # Input  : [B, 1920]
        # MLP    : 1920 -> 512 -> 128 -> 1
        # Output : predicted DMS_score [B, 1]
        # ====================================================
        model = train_fixed_epochs(
            X_outer_train,
            y_outer_train,
            input_dim=X.shape[1],
            num_epochs=final_epoch,
            seed=fold_seed,
        )

        predictions = predict(
            model,
            X_test,
        )

        oof_predictions[outer_test_mask] = predictions

        # ====================================================
        # Metrics
        # ====================================================
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

        fold_result = {
            "split": split_name,
            "fold_column": fold_column,
            "test_fold": int(test_fold),
            "n_outer_train": int(outer_train_mask.sum()),
            "n_test": int(outer_test_mask.sum()),
            "candidate_validation_folds": ",".join(
                map(str, candidate_val_folds)
            ),
            "inner_best_epochs": ",".join(
                map(str, inner_best_epochs)
            ),
            "inner_best_val_mses": ",".join(
                f"{x:.6f}"
                for x in inner_best_val_losses
            ),
            "final_epoch": int(final_epoch),
            "spearman": float(spearman),
            "spearman_p": float(p_value),
            "rmse": float(rmse),
            "r2": float(r2),
        }

        fold_results.append(fold_result)

        print()
        print(f"Outer train : {outer_train_mask.sum()}")
        print(f"Test        : {outer_test_mask.sum()}")
        print(f"Spearman    : {spearman:.4f}")
        print(f"RMSE        : {rmse:.4f}")
        print(f"R²          : {r2:.4f}")

    # ========================================================
    # Validate OOF coverage
    # ========================================================
    assert np.isfinite(oof_predictions).all()

    # ========================================================
    # Split summary
    # ========================================================
    fold_df = pd.DataFrame(fold_results)

    mean_spearman = fold_df["spearman"].mean()
    std_spearman = fold_df["spearman"].std(ddof=1)

    mean_rmse = fold_df["rmse"].mean()
    std_rmse = fold_df["rmse"].std(ddof=1)

    mean_r2 = fold_df["r2"].mean()
    std_r2 = fold_df["r2"].std(ddof=1)

    global_spearman, global_p = spearmanr(
        y,
        oof_predictions,
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
    print("=" * 70)
    print(f"{split_name.upper()} Summary")
    print("=" * 70)
    print(f"Spearman : {mean_spearman:.4f} ± {std_spearman:.4f}")
    print(f"RMSE     : {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"R²       : {mean_r2:.4f} ± {std_r2:.4f}")

    print()
    print("=== Global OOF ===")
    print(f"Spearman : {global_spearman:.4f}")
    print(f"RMSE     : {global_rmse:.4f}")
    print(f"R²       : {global_r2:.4f}")

    # ========================================================
    # Save fold metrics
    # ========================================================
    fold_df.to_csv(
        RESULT_ROOT / f"{split_name}_fold_metrics.csv",
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

    oof_df["prediction"] = oof_predictions
    oof_df["error"] = oof_predictions - y
    oof_df["absolute_error"] = np.abs(oof_df["error"])

    oof_df.to_csv(
        RESULT_ROOT / f"{split_name}_oof_predictions.csv",
        index=False,
    )

    # ========================================================
    # Save split summary
    # ========================================================
    split_summary = {
        "split": split_name,
        "fold_column": fold_column,
        "input_feature": "wt_local + delta_local",
        "input_dim": 1920,
        "model": "MLP",
        "architecture": [1920, 512, 128, 1],
        "activation": "ReLU",
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "epoch_selection": (
            "median of best epochs from all 4 structured inner validation folds"
        ),
        "fold_mean": {
            "spearman": float(mean_spearman),
            "rmse": float(mean_rmse),
            "r2": float(mean_r2),
        },
        "fold_std": {
            "spearman": float(std_spearman),
            "rmse": float(std_rmse),
            "r2": float(std_r2),
        },
        "global_oof": {
            "spearman": float(global_spearman),
            "spearman_p": float(global_p),
            "rmse": float(global_rmse),
            "r2": float(global_r2),
        },
    }

    with open(
        RESULT_ROOT / f"{split_name}_summary.json",
        "w",
    ) as f:
        json.dump(
            split_summary,
            f,
            indent=2,
        )

    all_summary_results.append(
        {
            "split": split_name,
            "fold_column": fold_column,
            "mean_spearman": float(mean_spearman),
            "std_spearman": float(std_spearman),
            "mean_rmse": float(mean_rmse),
            "std_rmse": float(std_rmse),
            "mean_r2": float(mean_r2),
            "std_r2": float(std_r2),
            "global_oof_spearman": float(global_spearman),
            "global_oof_rmse": float(global_rmse),
            "global_oof_r2": float(global_r2),
        }
    )


# ============================================================
# Final generalization comparison
# ============================================================

summary_df = pd.DataFrame(all_summary_results)

summary_df.to_csv(
    RESULT_ROOT / "generalization_summary.csv",
    index=False,
)

with open(
    RESULT_ROOT / "generalization_summary.json",
    "w",
) as f:
    json.dump(
        all_summary_results,
        f,
        indent=2,
    )

print()
print("=" * 90)
print("Robust Generalization Summary")
print("=" * 90)
print(summary_df.to_string(index=False))

print()
print("Saved to:", RESULT_ROOT)
