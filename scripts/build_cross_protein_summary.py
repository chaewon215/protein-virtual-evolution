from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ============================================================
# Step 11-5 v2
# Reproducible cross-protein summary from actual OOF files.
#
# No metric value is hard-coded.
#
# Handles:
#   1) explicit mu / sigma columns
#   2) member prediction columns only
#      -> recompute mu and sigma(ddof=1)
#
# Known TEM-1 layout:
# results/tem1/deep_ensemble_uncertainty/
#   random_oof_predictions.csv
#   modulo_oof_predictions.csv
#   contiguous_oof_predictions.csv
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "cross_protein"
OUT.mkdir(parents=True, exist_ok=True)

PRED_OUT = OUT / "cross_protein_prediction_summary.csv"
RISK_OUT = OUT / "cross_protein_risk_coverage_summary.csv"
MANIFEST_OUT = OUT / "source_manifest.csv"
SUMMARY_OUT = OUT / "summary.json"

COVERAGES = [0.2, 0.4, 0.6, 0.8, 1.0]
RANDOM_REPEATS = 1000
RANDOM_SEED = 42
HIGH_ERROR_QUANTILE = 0.8


# ============================================================
# Column aliases
# ============================================================

TRUE_ALIASES = [
    "DMS_score",
    "dms_score",
    "y_true",
    "true",
    "target",
    "fitness",
    "fitness_true",
    "true_fitness",
    "experimental_fitness",
    "experimental_score",
    "label",
]

MU_ALIASES = [
    "predicted_fitness_mu",
    "predicted_DMS_score",
    "predicted_dms_score",
    "prediction",
    "predicted_score",
    "y_pred",
    "mu",
    "mean_prediction",
    "prediction_mean",
    "pred_mean",
    "ensemble_mean",
    "ensemble_mu",
    "mean_pred",
    "oof_prediction",
]

SIGMA_ALIASES = [
    "uncertainty_sigma",
    "uncertainty",
    "sigma",
    "pred_std",
    "prediction_std",
    "ensemble_std",
    "ensemble_sigma",
    "std_prediction",
    "prediction_sd",
    "epistemic_uncertainty",
]

MEMBER_PATTERNS = [
    r"^member[_-]?\d+[_-]?(prediction|pred)?$",
    r"^member[_-]?\d+[_-]?prediction$",
    r"^prediction[_-]?member[_-]?\d+$",
    r"^pred[_-]?member[_-]?\d+$",
    r"^ensemble[_-]?member[_-]?\d+$",
    r"^model[_-]?\d+[_-]?(prediction|pred)?$",
]


# ============================================================
# Expected source layouts
# ============================================================

SOURCE_SPECS = [
    {
        "protein": "TEM-1",
        "split": "random",
        "preferred": [
            RESULTS / "tem1" / "deep_ensemble_uncertainty" / "random_oof_predictions.csv",
        ],
        "search_root": RESULTS / "tem1",
        "name_keywords": ["random", "oof"],
    },
    {
        "protein": "TEM-1",
        "split": "modulo",
        "preferred": [
            RESULTS / "tem1" / "deep_ensemble_uncertainty" / "modulo_oof_predictions.csv",
        ],
        "search_root": RESULTS / "tem1",
        "name_keywords": ["modulo", "oof"],
    },
    {
        "protein": "TEM-1",
        "split": "contiguous",
        "preferred": [
            RESULTS / "tem1" / "deep_ensemble_uncertainty" / "contiguous_oof_predictions.csv",
        ],
        "search_root": RESULTS / "tem1",
        "name_keywords": ["contiguous", "oof"],
    },
    {
        "protein": "GFP",
        "split": "random",
        "preferred": [
            RESULTS / "gfp" / "deep_ensemble_uncertainty" / "oof_predictions.csv",
            RESULTS / "gfp" / "deep_ensemble" / "oof_predictions.csv",
            RESULTS / "gfp" / "uncertainty" / "oof_predictions.csv",
        ],
        "search_root": RESULTS / "gfp",
        "name_keywords": ["oof"],
    },
    {
        "protein": "TPMT",
        "split": "random",
        "preferred": [
            RESULTS / "tpmt" / "deep_ensemble_uncertainty" / "random" / "oof_predictions.csv",
        ],
        "search_root": RESULTS / "tpmt",
        "name_keywords": ["random", "oof"],
    },
    {
        "protein": "TPMT",
        "split": "position",
        "preferred": [
            RESULTS / "tpmt" / "deep_ensemble_uncertainty" / "position" / "oof_predictions.csv",
        ],
        "search_root": RESULTS / "tpmt",
        "name_keywords": ["position", "oof"],
    },
]


# ============================================================
# Helpers
# ============================================================

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def find_alias(columns, aliases):
    mapping = {norm(c): c for c in columns}
    for alias in aliases:
        key = norm(alias)
        if key in mapping:
            return mapping[key]
    return None


def find_member_columns(columns):
    found = []
    for col in columns:
        n = norm(col)
        if any(re.fullmatch(pat, n) for pat in MEMBER_PATTERNS):
            found.append(col)

    # fallback: anything containing both "member" and a digit,
    # while excluding metadata-like columns
    if len(found) < 2:
        fallback = []
        for col in columns:
            n = norm(col)
            if "member" in n and re.search(r"\d", n):
                if not any(x in n for x in ["seed", "fold", "loss", "id"]):
                    fallback.append(col)
        found = fallback

    return found


def inspect_file(path):
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)

    true_col = find_alias(columns, TRUE_ALIASES)
    mu_col = find_alias(columns, MU_ALIASES)
    sigma_col = find_alias(columns, SIGMA_ALIASES)
    member_cols = find_member_columns(columns)

    return {
        "path": path,
        "columns": columns,
        "true_col": true_col,
        "mu_col": mu_col,
        "sigma_col": sigma_col,
        "member_cols": member_cols,
    }


def source_is_usable(info):
    if info["true_col"] is None:
        return False

    explicit = (
        info["mu_col"] is not None
        and info["sigma_col"] is not None
    )

    from_members = len(info["member_cols"]) >= 2

    return explicit or from_members


def resolve_source(spec):
    # 1. exact expected paths first
    for path in spec["preferred"]:
        if path.exists():
            info = inspect_file(path)
            if source_is_usable(info):
                return info

            raise RuntimeError(
                f"Found expected file but could not identify required columns:\n"
                f"{path}\n"
                f"Columns: {info['columns']}\n"
                f"Detected truth={info['true_col']}, "
                f"mu={info['mu_col']}, sigma={info['sigma_col']}, "
                f"members={info['member_cols']}"
            )

    # 2. recursive fallback
    root = spec["search_root"]
    if not root.exists():
        raise FileNotFoundError(f"Missing results directory: {root}")

    candidates = []
    for path in root.rglob("*.csv"):
        ptext = norm(str(path.relative_to(root)))

        if all(norm(k) in ptext for k in spec["name_keywords"]):
            try:
                info = inspect_file(path)
            except Exception:
                continue

            if source_is_usable(info):
                candidates.append(info)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        # Diagnostic: show OOF-like files + headers
        diagnostics = []
        for path in root.rglob("*.csv"):
            if "oof" in norm(path.name):
                try:
                    info = inspect_file(path)
                    diagnostics.append(
                        f"{path}\n"
                        f"  columns={info['columns']}\n"
                        f"  truth={info['true_col']} "
                        f"mu={info['mu_col']} sigma={info['sigma_col']} "
                        f"members={info['member_cols']}"
                    )
                except Exception as exc:
                    diagnostics.append(f"{path}\n  ERROR: {exc}")

        raise FileNotFoundError(
            f"Could not resolve {spec['protein']} / {spec['split']}.\n\n"
            + "\n\n".join(diagnostics)
        )

    raise RuntimeError(
        f"Ambiguous sources for {spec['protein']} / {spec['split']}:\n"
        + "\n".join(str(x["path"]) for x in candidates)
    )


# ============================================================
# Load arrays
# ============================================================

def load_arrays(info):
    usecols = [info["true_col"]]

    if info["mu_col"] is not None and info["sigma_col"] is not None:
        usecols += [info["mu_col"], info["sigma_col"]]
    else:
        usecols += info["member_cols"]

    df = pd.read_csv(info["path"], usecols=usecols)

    y = pd.to_numeric(
        df[info["true_col"]],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if info["mu_col"] is not None and info["sigma_col"] is not None:
        mu = pd.to_numeric(
            df[info["mu_col"]],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        sigma = pd.to_numeric(
            df[info["sigma_col"]],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        prediction_source = "explicit_mu_sigma"

    else:
        member_matrix = np.column_stack([
            pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            for c in info["member_cols"]
        ])
        # [N, M]
        mu = member_matrix.mean(axis=1)
        sigma = member_matrix.std(axis=1, ddof=1)
        prediction_source = f"recomputed_from_{member_matrix.shape[1]}_members"

    valid = (
        np.isfinite(y)
        & np.isfinite(mu)
        & np.isfinite(sigma)
    )

    if not valid.all():
        raise ValueError(
            f"Non-finite rows in {info['path']}: {(~valid).sum()}"
        )

    if (sigma < 0).any():
        raise ValueError(f"Negative sigma in {info['path']}")

    return y, mu, sigma, prediction_source


# ============================================================
# Metrics
# ============================================================

def prediction_metrics(y, mu, sigma):
    err = np.abs(y - mu)

    threshold = float(np.quantile(err, HIGH_ERROR_QUANTILE))
    high_error = (err >= threshold).astype(int)

    return {
        "n": int(len(y)),
        "spearman": float(spearmanr(y, mu).statistic),
        "rmse": float(np.sqrt(mean_squared_error(y, mu))),
        "mae": float(mean_absolute_error(y, mu)),
        "r2": float(r2_score(y, mu)),
        "mean_uncertainty": float(sigma.mean()),
        "uncertainty_error_rho": float(spearmanr(sigma, err).statistic),
        "high_error_auroc": float(roc_auc_score(high_error, sigma)),
        "high_error_threshold": threshold,
    }


def risk_coverage(y, mu, sigma):
    n = len(y)
    rng = np.random.default_rng(RANDOM_SEED)

    unc_order = np.argsort(sigma)
    err = np.abs(y - mu)
    oracle_order = np.argsort(err)

    rows = []

    for coverage in COVERAGES:
        subset_n = n if coverage == 1.0 else max(
            1,
            int(np.floor(coverage * n)),
        )

        selected = unc_order[:subset_n]
        oracle = oracle_order[:subset_n]

        selected_rmse = float(
            np.sqrt(mean_squared_error(y[selected], mu[selected]))
        )

        oracle_rmse = float(
            np.sqrt(mean_squared_error(y[oracle], mu[oracle]))
        )

        random_rmses = np.empty(RANDOM_REPEATS, dtype=np.float64)

        for i in range(RANDOM_REPEATS):
            idx = rng.choice(n, size=subset_n, replace=False)
            random_rmses[i] = np.sqrt(
                mean_squared_error(y[idx], mu[idx])
            )

        random_mean = float(random_rmses.mean())

        rows.append({
            "coverage": float(coverage),
            "subset_n": int(subset_n),
            "uncertainty_selected_rmse": selected_rmse,
            "random_rmse_mean": random_mean,
            "random_ci_low": float(np.quantile(random_rmses, 0.025)),
            "random_ci_high": float(np.quantile(random_rmses, 0.975)),
            "oracle_rmse": oracle_rmse,
            "rmse_reduction_vs_random": float(
                (random_mean - selected_rmse) / random_mean
            ) if random_mean != 0 else np.nan,
            "random_percentile": float(
                np.mean(random_rmses <= selected_rmse)
            ),
        })

    return rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-sources-only",
        action="store_true",
    )
    args = parser.parse_args()

    print("=" * 130)
    print("Step 11-5 v2: Reproducible Cross-Protein Summary")
    print("=" * 130)

    resolved = []

    for spec in SOURCE_SPECS:
        info = resolve_source(spec)
        resolved.append((spec, info))

    print()
    print("Resolved sources")
    print("-" * 130)

    for spec, info in resolved:
        method = (
            f"mu={info['mu_col']}, sigma={info['sigma_col']}"
            if info["mu_col"] is not None and info["sigma_col"] is not None
            else f"members={info['member_cols']}"
        )

        print(
            f"{spec['protein']:6s} | "
            f"{spec['split']:10s} | "
            f"{info['path']} | "
            f"truth={info['true_col']} | {method}"
        )

    if args.print_sources_only:
        return

    pred_rows = []
    risk_rows = []
    manifest_rows = []

    for spec, info in resolved:
        y, mu, sigma, pred_source = load_arrays(info)

        metrics = prediction_metrics(y, mu, sigma)

        source_relative = str(info["path"].relative_to(ROOT))

        pred_rows.append({
            "protein": spec["protein"],
            "split": spec["split"],
            **metrics,
            "prediction_source": pred_source,
            "source_oof_file": source_relative,
        })

        for row in risk_coverage(y, mu, sigma):
            risk_rows.append({
                "protein": spec["protein"],
                "split": spec["split"],
                **row,
                "source_oof_file": source_relative,
            })

        manifest_rows.append({
            "protein": spec["protein"],
            "split": spec["split"],
            "source_oof_file": source_relative,
            "n_rows": int(len(y)),
            "truth_column": info["true_col"],
            "mu_column": info["mu_col"],
            "sigma_column": info["sigma_col"],
            "member_columns": "|".join(info["member_cols"]),
            "prediction_source": pred_source,
        })

    pred_df = pd.DataFrame(pred_rows)
    risk_df = pd.DataFrame(risk_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    protein_order = {"TEM-1":0, "GFP":1, "TPMT":2}
    split_order = {"random":0, "modulo":1, "contiguous":2, "position":3}

    pred_df["_p"] = pred_df.protein.map(protein_order)
    pred_df["_s"] = pred_df.split.map(split_order)
    pred_df = (
        pred_df.sort_values(["_p","_s"])
        .drop(columns=["_p","_s"])
        .reset_index(drop=True)
    )

    risk_df["_p"] = risk_df.protein.map(protein_order)
    risk_df["_s"] = risk_df.split.map(split_order)
    risk_df = (
        risk_df.sort_values(["_p","_s","coverage"])
        .drop(columns=["_p","_s"])
        .reset_index(drop=True)
    )

    pred_df.to_csv(PRED_OUT, index=False)
    risk_df.to_csv(RISK_OUT, index=False)
    manifest_df.to_csv(MANIFEST_OUT, index=False)

    summary = {
        "stage": "Step 11-5-v2",
        "hardcoded_metric_values": False,
        "all_metrics_recomputed_from_oof_predictions": True,
        "risk_coverage_protocol": {
            "coverages": COVERAGES,
            "random_repeats": RANDOM_REPEATS,
            "random_seed": RANDOM_SEED,
        },
        "sources": manifest_df.to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 130)
    print("Cross-Protein Prediction Summary")
    print("=" * 130)
    print(
        pred_df[
            [
                "protein","split","n","spearman","rmse","mae","r2",
                "mean_uncertainty","uncertainty_error_rho","high_error_auroc",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 130)
    print("Cross-Protein Risk-Coverage Summary")
    print("=" * 130)
    print(
        risk_df[
            risk_df.coverage < 1.0
        ][
            [
                "protein","split","coverage","subset_n",
                "uncertainty_selected_rmse","random_rmse_mean",
                "rmse_reduction_vs_random","random_percentile",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(PRED_OUT)
    print(RISK_OUT)
    print(MANIFEST_OUT)
    print(SUMMARY_OUT)

    print()
    print("IMPORTANT:")
    print("All metrics were recomputed from actual OOF predictions.")
    print("If mu/sigma were absent, they were recomputed from ensemble member predictions.")


if __name__ == "__main__":
    main()
