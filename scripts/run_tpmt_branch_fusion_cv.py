from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = PROJECT_ROOT / "data" / "processed" / "tpmt_single_mutants.csv"
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "esmc300m_tpmt" / "tpmt_variant_features.pt"
META_PATH = PROJECT_ROOT / "data" / "processed" / "esmc300m_tpmt" / "tpmt_variant_feature_metadata.csv"
OUT = PROJECT_ROOT / "results" / "tpmt" / "branch_fusion_cv"
OUT.mkdir(parents=True, exist_ok=True)

BASE_SEED = 42
EPOCHS = 24
BATCH = 64
PRED_BATCH = 512
LR = 1e-3
WD = 1e-4
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = {
    "random": "fold_random_5",
    "position": "fold_position_5",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BranchFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.wt = nn.Sequential(nn.Linear(960, 64), nn.ReLU())
        self.local = nn.Sequential(nn.Linear(960, 64), nn.ReLU())
        self.global_ = nn.Sequential(nn.Linear(960, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x [B,2880]
        z_wt = self.wt(x[:, 0:960])          # [B,64]
        z_local = self.local(x[:, 960:1920]) # [B,64]
        z_global = self.global_(x[:, 1920:2880]) # [B,64]
        z = torch.cat([z_wt, z_local, z_global], dim=1) # [B,192]
        return self.head(z).squeeze(-1) # [B]

def n_params():
    return sum(p.numel() for p in BranchFusionMLP().parameters() if p.requires_grad)

def make_loader(X, y, seed):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g)

def fit_model(X_train, y_train, seed):
    set_seed(seed)
    model = BranchFusionMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    loader = make_loader(X_train, y_train, seed)
    final_loss = None

    for _ in range(EPOCHS):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        final_loss = total / n
    return model, float(final_loss)

@torch.inference_mode()
def predict(model, X):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=PRED_BATCH, shuffle=False)
    out = []
    model.eval()
    for (xb,) in loader:
        out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

def metrics(y, pred):
    return {
        "spearman": float(spearmanr(y, pred).statistic),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }

def run_split(split_name, fold_col, X, df):
    split_dir = OUT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    y = df["DMS_score"].to_numpy(float)
    folds = df[fold_col].to_numpy(int)
    oof = np.full(len(df), np.nan)
    fold_rows = []

    for fold in sorted(np.unique(folds)):
        tr = np.flatnonzero(folds != fold)
        te = np.flatnonzero(folds == fold)

        if split_name == "position":
            train_pos = set(df.iloc[tr]["position"].astype(int))
            test_pos = set(df.iloc[te]["position"].astype(int))
            overlap = train_pos & test_pos
            if overlap:
                raise RuntimeError(f"Position leakage in fold {fold}: {sorted(overlap)[:10]}")

        x_scaler = StandardScaler().fit(X[tr])
        Xtr = x_scaler.transform(X[tr])
        Xte = x_scaler.transform(X[te])

        y_scaler = StandardScaler().fit(y[tr].reshape(-1, 1))
        ytr = y_scaler.transform(y[tr].reshape(-1, 1)).ravel()

        seed = BASE_SEED + int(fold)
        model, final_loss = fit_model(Xtr, ytr, seed)

        pred_scaled = predict(model, Xte)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        oof[te] = pred

        m = metrics(y[te], pred)
        fold_rows.append({
            "fold": int(fold),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "seed": seed,
            "epochs": EPOCHS,
            "final_train_loss_scaled": final_loss,
            **m,
        })

        print(
            f"{split_name.upper()} fold {fold} | "
            f"train={X[tr].shape} test={X[te].shape} | "
            f"rho={m['spearman']:.4f} RMSE={m['rmse']:.4f} "
            f"MAE={m['mae']:.4f} R2={m['r2']:.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof).any():
        raise RuntimeError(f"Missing OOF predictions for {split_name}")

    fold_df = pd.DataFrame(fold_rows)
    global_m = metrics(y, oof)

    oof_df = df[["mutant", "position", "DMS_score", fold_col]].copy()
    oof_df["predicted_DMS_score"] = oof
    oof_df["absolute_error"] = np.abs(y - oof)

    fold_df.to_csv(split_dir / "fold_metrics.csv", index=False)
    oof_df.to_csv(split_dir / "oof_predictions.csv", index=False)

    summary = {
        "split": split_name,
        "fold_column": fold_col,
        "n_samples": int(len(df)),
        "feature_shape": list(X.shape),
        "architecture": "BranchFusionMLP: 960->64 x3, concat 192->64->1",
        "parameters": n_params(),
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LR,
            "weight_decay": WD,
            "dropout": DROPOUT,
            "tpmt_specific_hyperparameter_tuning": False,
        },
        "fold_mean": {k: float(fold_df[k].mean()) for k in ["spearman","rmse","mae","r2"]},
        "fold_std": {k: float(fold_df[k].std(ddof=1)) for k in ["spearman","rmse","mae","r2"]},
        "global_oof": global_m,
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "split": split_name,
        "spearman_mean": float(fold_df["spearman"].mean()),
        "spearman_std": float(fold_df["spearman"].std(ddof=1)),
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=1)),
        "mae_mean": float(fold_df["mae"].mean()),
        "mae_std": float(fold_df["mae"].std(ddof=1)),
        "r2_mean": float(fold_df["r2"].mean()),
        "r2_std": float(fold_df["r2"].std(ddof=1)),
        "global_spearman": global_m["spearman"],
        "global_rmse": global_m["rmse"],
        "global_mae": global_m["mae"],
        "global_r2": global_m["r2"],
    }

def main():
    print("=" * 120)
    print("Step 11-3: TPMT Frozen Branch Fusion External Validation")
    print("=" * 120)
    print(f"Device      : {DEVICE}")
    print(f"Parameters  : {n_params():,}")
    print(f"Epochs      : {EPOCHS}")
    print("TPMT-specific hyperparameter tuning: NO")

    labels = pd.read_csv(
        LABEL_PATH,
        usecols=["mutant","DMS_score","position","fold_random_5","fold_position_5"],
    )
    meta = pd.read_csv(META_PATH, usecols=["feature_index","mutant"])
    aligned = (
        meta.merge(labels, on="mutant", how="left", validate="one_to_one")
        .sort_values("feature_index")
        .reset_index(drop=True)
    )

    if aligned["DMS_score"].isna().any():
        raise ValueError("Feature/label alignment failed.")

    expected = np.arange(len(aligned))
    if not np.array_equal(aligned["feature_index"].to_numpy(int), expected):
        raise ValueError("feature_index is not contiguous/aligned.")

    cache = torch.load(FEATURE_PATH, map_location="cpu")
    X = cache["X"].float().numpy()
    if X.shape != (len(aligned), 2880):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    print(f"X            : {X.shape}")
    print(f"y            : {aligned['DMS_score'].shape}")

    results = []
    for split_name, fold_col in SPLITS.items():
        print()
        print("#" * 120)
        print(f"SPLIT: {split_name.upper()}")
        print("#" * 120)
        results.append(run_split(split_name, fold_col, X, aligned))

    comp = pd.DataFrame(results)
    comp.to_csv(OUT / "split_comparison.csv", index=False)

    summary = {
        "stage": "Step 11-3",
        "protein": "TPMT",
        "external_validation": True,
        "tpmt_specific_hyperparameter_tuning": False,
        "feature_shape": list(X.shape),
        "architecture": "Frozen GFP BranchFusionMLP",
        "training_protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LR,
            "weight_decay": WD,
            "dropout": DROPOUT,
        },
        "results": comp.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 150)
    print("TPMT Frozen Branch Fusion External Validation Summary")
    print("=" * 150)
    print(comp.to_string(index=False))
    print()
    print(f"Saved: {OUT}")
    print()
    print("IMPORTANT:")
    print("The GFP-selected representation, architecture, and 24-epoch protocol were reused unchanged.")
    print("fold_position_5 tests mutation positions absent from the training folds.")

if __name__ == "__main__":
    main()
