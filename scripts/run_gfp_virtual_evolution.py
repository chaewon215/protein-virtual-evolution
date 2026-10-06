from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Step 8-6
# Retrospective Pool-Based Virtual Directed Evolution
#
# Policies:
#   Random
#   Greedy       : score = mu
#   UCB          : score = mu + beta * sigma
#   Conservative : score = mu - beta * sigma
#
# Frozen protocol:
#   Initial labeled set : 1,084 GFP single mutants
#   Candidate pool      : 50,630 multi-mutants
#   Budget              : 100 acquisitions
#   Batch size          : 20
#   Acquisition rounds  : 5
#   Ensemble            : 5 BranchFusionMLP members
#   Epochs/member       : 24
#   UCB beta            : 1.0
#   Conservative beta   : 1.0
#
# IMPORTANT LEAKAGE RULE
# ----------------------
# Acquisition policies NEVER receive true multi-mutant
# DMS_score values.
#
# The hidden oracle owns the complete multi-mutant label table.
# Only after a policy selects mutant IDs does oracle.reveal()
# return the corresponding labels.
#
# Full hidden labels are used ONLY for retrospective evaluation
# metrics after selection (e.g. top-100 recovery).
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

MULTI_LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_multi_mutants.csv"
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
    / "virtual_evolution"
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

ACQUISITION_HISTORY_PATH = (
    OUTPUT_ROOT
    / "acquisition_history.csv"
)

ROUND_METRICS_PATH = (
    OUTPUT_ROOT
    / "round_metrics.csv"
)

AGGREGATE_METRICS_PATH = (
    OUTPUT_ROOT
    / "aggregate_metrics.csv"
)

FINAL_POLICY_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "final_policy_summary.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


# ============================================================
# Frozen experiment settings
# ============================================================

BASE_SEED = 42

POLICIES = [
    "random",
    "greedy",
    "ucb",
    "conservative",
]

# Three complete simulation replicates by default.
# For a quick smoke test, temporarily change to:
# SIMULATION_SEEDS = [0]
#
# After confirming the script runs correctly, keep at least
# 3 repeats. You can later increase to [0,1,2,3,4].
SIMULATION_SEEDS = list(range(10))

ENSEMBLE_SIZE = 5

MEMBER_BASE_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

FINAL_EPOCHS = 24

BUDGET = 100

BATCH_SIZE_ACQUISITION = 20

UCB_BETA = 1.0

CONSERVATIVE_BETA = 1.0


# ============================================================
# Training settings
# ============================================================

TRAIN_BATCH_SIZE = 64

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
# Evaluation-only settings
#
# These are NEVER provided to acquisition policies.
# ============================================================

TOP_K_VALUES = [
    10,
    50,
    100,
]

TOP_PERCENT_FRACTION = 0.01


# ============================================================
# Feature layout
# ============================================================

WT_START = 0
WT_END = 960

LOCAL_START = 960
LOCAL_END = 1920

GLOBAL_START = 1920
GLOBAL_END = 2880


# ============================================================
# Validation
# ============================================================

if BUDGET <= 0:
    raise ValueError(
        "BUDGET must be positive."
    )

if BATCH_SIZE_ACQUISITION <= 0:
    raise ValueError(
        "BATCH_SIZE_ACQUISITION must be positive."
    )

N_ROUNDS = int(
    np.ceil(
        BUDGET
        / BATCH_SIZE_ACQUISITION
    )
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
# Frozen Branch Fusion architecture
#
# Input:
#   [B, 2880]
#
# WT branch:
#   [B, 960] -> [B, 64]
#
# Delta-local branch:
#   [B, 960] -> [B, 64]
#
# Delta-global branch:
#   [B, 960] -> [B, 64]
#
# concatenate:
#   [B, 192]
#
# head:
#   [B, 192] -> [B, 64] -> [B, 1]
#
# Output during training:
#   standardized DMS_score [B]
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
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# DataLoader / training helpers
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
        batch_size=TRAIN_BATCH_SIZE,
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

        predictions.append(
            model(
                xb
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(
        predictions
    )


# ============================================================
# Hidden Oracle
#
# This object owns the hidden multi-mutant labels.
#
# The policy never receives:
#   - label_table
#   - true DMS_score
#   - global rank
#   - top-k membership
#
# The simulation can only call:
#   oracle.reveal(selected_mutants)
#
# after selection has occurred.
#
# Evaluation methods may use all labels retrospectively.
# ============================================================

class HiddenOracle:

    def __init__(
        self,
        multi_label_path: Path,
        expected_candidate_ids,
    ):
        labels = pd.read_csv(
            multi_label_path,
            usecols=[
                "mutant",
                "DMS_score",
                "mutation_count",
            ],
        )

        labels[
            "mutant"
        ] = labels[
            "mutant"
        ].astype(
            str
        )

        if (
            labels[
                "mutant"
            ]
            .duplicated()
            .any()
        ):
            raise ValueError(
                "Duplicate mutant IDs in hidden label table."
            )

        if not (
            labels[
                "mutation_count"
            ]
            >= 2
        ).all():
            raise ValueError(
                "Hidden oracle contains non-multi mutants."
            )

        expected_set = set(
            expected_candidate_ids
        )

        label_set = set(
            labels[
                "mutant"
            ]
        )

        if expected_set != label_set:

            missing_labels = (
                expected_set
                - label_set
            )

            extra_labels = (
                label_set
                - expected_set
            )

            raise ValueError(
                "Candidate/hidden-label mismatch. "
                f"Missing labels={len(missing_labels)}, "
                f"extra labels={len(extra_labels)}"
            )

        self._label_table = (
            labels[
                [
                    "mutant",
                    "DMS_score",
                    "mutation_count",
                ]
            ]
            .copy()
            .set_index(
                "mutant"
            )
        )

        self._sorted_true = (
            labels.sort_values(
                "DMS_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        self._global_best = float(
            labels[
                "DMS_score"
            ].max()
        )

        self._top_sets = {
            k:
                set(
                    self._sorted_true[
                        "mutant"
                    ]
                    .head(
                        min(
                            k,
                            len(
                                self._sorted_true
                            )
                        )
                    )
                )

            for k in TOP_K_VALUES
        }

        n_top_percent = max(
            1,
            int(
                np.ceil(
                    TOP_PERCENT_FRACTION
                    * len(
                        labels
                    )
                )
            ),
        )

        self._top_percent_n = int(
            n_top_percent
        )

        self._top_percent_set = set(
            self._sorted_true[
                "mutant"
            ]
            .head(
                n_top_percent
            )
        )

        self._top_percent_threshold = float(
            self._sorted_true[
                "DMS_score"
            ]
            .iloc[
                n_top_percent - 1
            ]
        )


    def reveal(
        self,
        mutant_ids,
    ):
        mutant_ids = list(
            mutant_ids
        )

        if len(
            mutant_ids
        ) == 0:
            return pd.DataFrame(
                columns=[
                    "mutant",
                    "DMS_score",
                    "mutation_count",
                ]
            )

        revealed = (
            self._label_table.loc[
                mutant_ids
            ]
            .reset_index()
            .copy()
        )

        return revealed


    def evaluate_selected(
        self,
        selected_mutant_ids,
    ):
        selected_mutant_ids = list(
            selected_mutant_ids
        )

        if len(
            selected_mutant_ids
        ) == 0:

            result = {
                "n_acquired":
                    0,

                "best_acquired_multi":
                    np.nan,

                "mean_acquired_fitness":
                    np.nan,

                "top5_acquired_mean":
                    np.nan,

                "regret_to_global_multi_best":
                    np.nan,

                "top_percent_hits":
                    0,

                "top_percent_hit_rate":
                    0.0,
            }

            for k in TOP_K_VALUES:
                result[
                    f"top{k}_recovered"
                ] = 0

                result[
                    f"top{k}_recovery_fraction"
                ] = 0.0

            return result

        selected_set = set(
            selected_mutant_ids
        )

        selected_scores = (
            self._label_table.loc[
                selected_mutant_ids,
                "DMS_score",
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        top5_n = min(
            5,
            len(
                selected_scores
            ),
        )

        top5_mean = float(
            np.sort(
                selected_scores
            )[
                -top5_n:
            ]
            .mean()
        )

        best_acquired = float(
            selected_scores.max()
        )

        top_percent_hits = len(
            selected_set
            & self._top_percent_set
        )

        result = {
            "n_acquired":
                int(
                    len(
                        selected_mutant_ids
                    )
                ),

            "best_acquired_multi":
                best_acquired,

            "mean_acquired_fitness":
                float(
                    selected_scores.mean()
                ),

            "top5_acquired_mean":
                top5_mean,

            "regret_to_global_multi_best":
                float(
                    self._global_best
                    - best_acquired
                ),

            "top_percent_hits":
                int(
                    top_percent_hits
                ),

            "top_percent_hit_rate":
                float(
                    top_percent_hits
                    / len(
                        selected_mutant_ids
                    )
                ),
        }

        for k in TOP_K_VALUES:

            recovered = len(
                selected_set
                & self._top_sets[
                    k
                ]
            )

            result[
                f"top{k}_recovered"
            ] = int(
                recovered
            )

            result[
                f"top{k}_recovery_fraction"
            ] = float(
                recovered
                / min(
                    k,
                    len(
                        self._label_table
                    ),
                )
            )

        return result


    @property
    def global_best_multi(
        self,
    ):
        return self._global_best


    @property
    def top_percent_threshold(
        self,
    ):
        return self._top_percent_threshold


    @property
    def top_percent_n(
        self,
    ):
        return self._top_percent_n


# ============================================================
# Policy selection
#
# IMPORTANT:
# candidate_view contains NO DMS_score.
#
# Expected columns:
#   mutant
#   mutation_count
#   predicted_fitness_mu
#   uncertainty_sigma
#
# No oracle object is passed here.
# ============================================================

def select_batch(
    policy_name,
    candidate_view,
    batch_size,
    rng,
):
    if "DMS_score" in candidate_view.columns:
        raise RuntimeError(
            "Leakage detected: DMS_score was passed "
            "to acquisition policy."
        )

    batch_size = min(
        batch_size,
        len(
            candidate_view
        ),
    )

    if batch_size <= 0:
        return candidate_view.iloc[
            0:0
        ].copy()

    if policy_name == "random":

        selected_positions = rng.choice(
            len(
                candidate_view
            ),
            size=batch_size,
            replace=False,
        )

        selected = (
            candidate_view.iloc[
                selected_positions
            ]
            .copy()
        )

        selected[
            "acquisition_score"
        ] = np.nan

        return selected


    scoring_df = candidate_view.copy()


    if policy_name == "greedy":

        scoring_df[
            "acquisition_score"
        ] = scoring_df[
            "predicted_fitness_mu"
        ]


    elif policy_name == "ucb":

        scoring_df[
            "acquisition_score"
        ] = (
            scoring_df[
                "predicted_fitness_mu"
            ]
            + UCB_BETA
            * scoring_df[
                "uncertainty_sigma"
            ]
        )


    elif policy_name == "conservative":

        scoring_df[
            "acquisition_score"
        ] = (
            scoring_df[
                "predicted_fitness_mu"
            ]
            - CONSERVATIVE_BETA
            * scoring_df[
                "uncertainty_sigma"
            ]
        )


    else:
        raise ValueError(
            f"Unknown policy: {policy_name}"
        )


    selected = (
        scoring_df.sort_values(
            by=[
                "acquisition_score",
                "predicted_fitness_mu",
                "mutant",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(
            batch_size
        )
        .copy()
    )

    return selected


# ============================================================
# Train current 5-member ensemble
#
# Inputs:
#   X_labeled_raw [N_labeled, 2880]
#   y_labeled     [N_labeled]
#
# Outputs:
#   mu            [N_candidates]
#   sigma         [N_candidates]
#   member_pred   [N_candidates, 5]
#
# Each round:
#   scalers are REFIT on the currently labeled set only.
# ============================================================

def fit_ensemble_and_predict(
    X_labeled_raw,
    y_labeled_raw,
    X_candidate_raw,
    simulation_seed,
    round_index,
):
    x_scaler = StandardScaler()

    X_labeled = x_scaler.fit_transform(
        X_labeled_raw
    )

    X_candidate = x_scaler.transform(
        X_candidate_raw
    )

    y_scaler = StandardScaler()

    y_labeled = (
        y_scaler.fit_transform(
            y_labeled_raw.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )

    member_predictions = []

    final_train_losses = []

    for member_idx, base_seed in enumerate(
        MEMBER_BASE_SEEDS
    ):

        # Same seed schedule across policies.
        #
        # Therefore, if two policies happen to have the same
        # labeled set at the same simulation/round, they use
        # the same training randomness.
        member_seed = (
            base_seed
            + simulation_seed * 100000
            + round_index * 1000
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
            X_labeled,
            y_labeled,
        )

        final_loss = np.nan

        for _ in range(
            FINAL_EPOCHS
        ):

            final_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
            )

        pred_scaled = predict_scaled(
            model,
            X_candidate,
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

        member_predictions.append(
            pred
        )

        final_train_losses.append(
            float(
                final_loss
            )
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    member_predictions = np.column_stack(
        member_predictions
    )

    mu = member_predictions.mean(
        axis=1
    )

    sigma = member_predictions.std(
        axis=1,
        ddof=1,
    )

    return {
        "mu":
            mu,

        "sigma":
            sigma,

        "member_predictions":
            member_predictions,

        "mean_final_train_loss_scaled":
            float(
                np.mean(
                    final_train_losses
                )
            ),
    }


# ============================================================
# Load label-free feature metadata and feature cache
# ============================================================

print(
    "=" * 120
)

print(
    "Step 8-6: GFP Retrospective Virtual Directed Evolution"
)

print(
    "=" * 120
)

print(
    f"Device              : {DEVICE}"
)

print(
    f"Policies            : {POLICIES}"
)

print(
    f"Simulation seeds    : {SIMULATION_SEEDS}"
)

print(
    f"Budget              : {BUDGET}"
)

print(
    f"Batch size          : {BATCH_SIZE_ACQUISITION}"
)

print(
    f"Rounds              : {N_ROUNDS}"
)

print(
    f"Ensemble size       : {ENSEMBLE_SIZE}"
)

print(
    f"Epochs/member       : {FINAL_EPOCHS}"
)

print(
    f"UCB beta            : {UCB_BETA}"
)

print(
    f"Conservative beta   : {CONSERVATIVE_BETA}"
)


feature_metadata = pd.read_csv(
    FEATURE_METADATA_PATH
)


required_metadata_columns = {
    "mutant",
    "feature_index",
    "mutation_count",
}


missing_metadata = (
    required_metadata_columns
    - set(
        feature_metadata.columns
    )
)


if missing_metadata:
    raise ValueError(
        f"Missing feature metadata columns: "
        f"{sorted(missing_metadata)}"
    )


feature_metadata[
    "mutant"
] = feature_metadata[
    "mutant"
].astype(
    str
)


if (
    feature_metadata[
        "mutant"
    ]
    .duplicated()
    .any()
):
    raise ValueError(
        "Duplicate mutant IDs in feature metadata."
    )


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
        f"Feature cache missing keys: "
        f"{sorted(missing_feature_keys)}"
    )


# ============================================================
# Build ALL feature matrix
#
# [N_all, 2880]
# ============================================================

all_wt = (
    feature_cache[
        "wt_context_mean"
    ]
    .float()
    .numpy()
)


all_local = (
    feature_cache[
        "delta_local_sum"
    ]
    .float()
    .numpy()
)


all_global = (
    feature_cache[
        "delta_global"
    ]
    .float()
    .numpy()
)


X_all_by_feature_index = np.concatenate(
    [
        all_wt,
        all_local,
        all_global,
    ],
    axis=1,
)


if X_all_by_feature_index.shape[
    1
] != 2880:
    raise ValueError(
        "Unexpected cached feature dimension: "
        f"{X_all_by_feature_index.shape}"
    )


# ============================================================
# Initial labeled singles
# ============================================================

single_df = pd.read_csv(
    SINGLE_PATH,
    usecols=[
        "mutant",
        "DMS_score",
        "mutation_count",
    ],
)


single_df[
    "mutant"
] = single_df[
    "mutant"
].astype(
    str
)


if not (
    single_df[
        "mutation_count"
    ]
    == 1
).all():
    raise ValueError(
        "Initial label set contains non-single mutants."
    )


single_aligned = single_df.merge(
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
    single_aligned[
        "feature_index"
    ]
    .isna()
    .any()
):
    raise ValueError(
        "Single-mutant feature alignment failed."
    )


single_feature_indices = (
    single_aligned[
        "feature_index"
    ]
    .astype(
        int
    )
    .to_numpy()
)


X_initial_single = (
    X_all_by_feature_index[
        single_feature_indices
    ]
)


y_initial_single = (
    single_aligned[
        "DMS_score"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


initial_best_single = float(
    y_initial_single.max()
)


# ============================================================
# Label-free candidate pool
# ============================================================

candidate_meta = (
    feature_metadata.loc[
        feature_metadata[
            "mutation_count"
        ]
        >= 2,
        [
            "mutant",
            "feature_index",
            "mutation_count",
        ],
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


candidate_feature_indices = (
    candidate_meta[
        "feature_index"
    ]
    .astype(
        int
    )
    .to_numpy()
)


X_candidate_all = (
    X_all_by_feature_index[
        candidate_feature_indices
    ]
)


candidate_ids = (
    candidate_meta[
        "mutant"
    ]
    .astype(
        str
    )
    .to_numpy()
)


if len(
    candidate_meta
) != 50630:

    print(
        "WARNING: expected 50,630 multi-mutants, "
        f"found {len(candidate_meta):,}."
    )


# ============================================================
# Hidden oracle creation
#
# THIS is the first point in the project where full
# multi-mutant DMS_score is intentionally loaded.
#
# It remains encapsulated inside HiddenOracle.
# ============================================================

oracle = HiddenOracle(
    MULTI_LABEL_PATH,
    expected_candidate_ids=candidate_ids,
)


print()
print(
    "Data shapes"
)

print(
    f"Initial single X      : "
    f"{X_initial_single.shape}"
)

print(
    f"Initial single y      : "
    f"{y_initial_single.shape}"
)

print(
    f"Candidate multi X     : "
    f"{X_candidate_all.shape}"
)

print(
    f"Candidate IDs         : "
    f"{candidate_ids.shape}"
)

print(
    f"Initial best single   : "
    f"{initial_best_single:.6f}"
)

print(
    f"Global best multi     : "
    f"{oracle.global_best_multi:.6f} "
    f"(evaluation only)"
)

print(
    f"Top 1% threshold      : "
    f"{oracle.top_percent_threshold:.6f} "
    f"(evaluation only)"
)


probe = BranchFusionMLP()


print(
    f"Parameters/member     : "
    f"{count_parameters(probe):,}"
)


del probe


# ============================================================
# Convenience lookup:
#
# candidate mutant -> row index in candidate_meta/X_candidate
#
# No labels stored here.
# ============================================================

candidate_id_to_row = {
    mutant:
        row_idx

    for row_idx, mutant in enumerate(
        candidate_ids
    )
}


# ============================================================
# Simulation
# ============================================================

all_acquisition_rows = []

all_round_rows = []


for simulation_seed in SIMULATION_SEEDS:

    print()
    print(
        "#" * 120
    )

    print(
        f"SIMULATION SEED {simulation_seed}"
    )

    print(
        "#" * 120
    )


    for policy_idx, policy_name in enumerate(
        POLICIES
    ):

        print()
        print(
            "=" * 110
        )

        print(
            f"Policy: {policy_name.upper()} "
            f"| simulation_seed={simulation_seed}"
        )

        print(
            "=" * 110
        )


        # ----------------------------------------------------
        # Current labeled set starts from all singles
        # ----------------------------------------------------

        labeled_feature_rows = list(
            single_feature_indices
        )

        labeled_scores = list(
            y_initial_single.astype(
                float
            )
        )


        # Multi-mutant acquisition history
        acquired_mutants = []

        acquired_set = set()


        # ----------------------------------------------------
        # Random policy RNG
        #
        # Policy-specific offset prevents identical random
        # streams from accidentally being reused elsewhere.
        # ----------------------------------------------------

        policy_rng = np.random.default_rng(
            BASE_SEED
            + simulation_seed * 10000
            + policy_idx * 1000
        )


        # ----------------------------------------------------
        # Budget-0 evaluation
        # ----------------------------------------------------

        initial_eval = oracle.evaluate_selected(
            acquired_mutants
        )


        all_round_rows.append(
            {
                "simulation_seed":
                    int(
                        simulation_seed
                    ),

                "policy":
                    policy_name,

                "round":
                    0,

                "budget_used":
                    0,

                "n_labeled_total":
                    int(
                        len(
                            labeled_scores
                        )
                    ),

                "initial_best_single":
                    initial_best_single,

                "best_overall_including_initial":
                    initial_best_single,

                "improvement_over_initial_best_single":
                    0.0,

                "mean_selected_mu_this_round":
                    np.nan,

                "mean_selected_sigma_this_round":
                    np.nan,

                "mean_selected_mutation_count_this_round":
                    np.nan,

                "mean_final_train_loss_scaled":
                    np.nan,

                **initial_eval,
            }
        )


        # ====================================================
        # Acquisition rounds
        # ====================================================

        for round_index in range(
            1,
            N_ROUNDS + 1,
        ):

            budget_used_before = len(
                acquired_mutants
            )


            remaining_budget = (
                BUDGET
                - budget_used_before
            )


            if remaining_budget <= 0:
                break


            this_batch_size = min(
                BATCH_SIZE_ACQUISITION,
                remaining_budget,
            )


            # -----------------------------------------------
            # Remaining candidate indices
            # -----------------------------------------------

            remaining_rows = np.array(
                [
                    row_idx

                    for row_idx, mutant
                    in enumerate(
                        candidate_ids
                    )

                    if mutant not in acquired_set
                ],
                dtype=np.int64,
            )


            remaining_meta = (
                candidate_meta.iloc[
                    remaining_rows
                ]
                .reset_index(
                    drop=True
                )
                .copy()
            )


            # -----------------------------------------------
            # RANDOM:
            # no model is needed for acquisition.
            #
            # GREEDY/UCB/CONSERVATIVE:
            # retrain ensemble on current labeled set and
            # predict all unrevealed candidates.
            # -----------------------------------------------

            if policy_name == "random":

                candidate_view = (
                    remaining_meta[
                        [
                            "mutant",
                            "mutation_count",
                        ]
                    ]
                    .copy()
                )


                candidate_view[
                    "predicted_fitness_mu"
                ] = np.nan


                candidate_view[
                    "uncertainty_sigma"
                ] = np.nan


                mean_train_loss = np.nan


            else:

                X_labeled_raw = (
                    X_all_by_feature_index[
                        np.asarray(
                            labeled_feature_rows,
                            dtype=np.int64,
                        )
                    ]
                )


                y_labeled_raw = np.asarray(
                    labeled_scores,
                    dtype=np.float64,
                )


                X_remaining_raw = (
                    X_candidate_all[
                        remaining_rows
                    ]
                )


                print(
                    f"Round {round_index}/{N_ROUNDS} "
                    f"| labeled={len(y_labeled_raw)} "
                    f"| remaining={len(remaining_rows)}"
                )


                ensemble_output = (
                    fit_ensemble_and_predict(
                        X_labeled_raw,
                        y_labeled_raw,
                        X_remaining_raw,
                        simulation_seed=
                            simulation_seed,
                        round_index=
                            round_index - 1,
                    )
                )


                candidate_view = (
                    remaining_meta[
                        [
                            "mutant",
                            "mutation_count",
                        ]
                    ]
                    .copy()
                )


                candidate_view[
                    "predicted_fitness_mu"
                ] = ensemble_output[
                    "mu"
                ]


                candidate_view[
                    "uncertainty_sigma"
                ] = ensemble_output[
                    "sigma"
                ]


                mean_train_loss = (
                    ensemble_output[
                        "mean_final_train_loss_scaled"
                    ]
                )


            # -----------------------------------------------
            # SELECTION
            #
            # No DMS_score exists in candidate_view.
            # -----------------------------------------------

            selected = select_batch(
                policy_name=
                    policy_name,

                candidate_view=
                    candidate_view,

                batch_size=
                    this_batch_size,

                rng=
                    policy_rng,
            )


            selected_mutants = (
                selected[
                    "mutant"
                ]
                .astype(
                    str
                )
                .tolist()
            )


            # -----------------------------------------------
            # Hidden label reveal happens ONLY now.
            # -----------------------------------------------

            revealed = oracle.reveal(
                selected_mutants
            )


            revealed_lookup = {
                row.mutant:
                    float(
                        row.DMS_score
                    )

                for row in revealed.itertuples(
                    index=False
                )
            }


            # -----------------------------------------------
            # Add newly revealed samples to labeled set
            # -----------------------------------------------

            for selection_order, selected_row in enumerate(
                selected.itertuples(
                    index=False
                ),
                start=1,
            ):

                mutant = str(
                    selected_row.mutant
                )


                true_score = (
                    revealed_lookup[
                        mutant
                    ]
                )


                candidate_row_idx = (
                    candidate_id_to_row[
                        mutant
                    ]
                )


                feature_index = int(
                    candidate_meta.iloc[
                        candidate_row_idx
                    ][
                        "feature_index"
                    ]
                )


                labeled_feature_rows.append(
                    feature_index
                )


                labeled_scores.append(
                    true_score
                )


                acquired_mutants.append(
                    mutant
                )


                acquired_set.add(
                    mutant
                )


                predicted_mu = (
                    float(
                        selected_row.predicted_fitness_mu
                    )
                    if np.isfinite(
                        selected_row.predicted_fitness_mu
                    )
                    else np.nan
                )


                predicted_sigma = (
                    float(
                        selected_row.uncertainty_sigma
                    )
                    if np.isfinite(
                        selected_row.uncertainty_sigma
                    )
                    else np.nan
                )


                acquisition_score = (
                    float(
                        selected_row.acquisition_score
                    )
                    if np.isfinite(
                        selected_row.acquisition_score
                    )
                    else np.nan
                )


                all_acquisition_rows.append(
                    {
                        "simulation_seed":
                            int(
                                simulation_seed
                            ),

                        "policy":
                            policy_name,

                        "round":
                            int(
                                round_index
                            ),

                        "selection_order_in_round":
                            int(
                                selection_order
                            ),

                        "budget_index":
                            int(
                                len(
                                    acquired_mutants
                                )
                            ),

                        "mutant":
                            mutant,

                        "mutation_count":
                            int(
                                selected_row.mutation_count
                            ),

                        "predicted_fitness_mu":
                            predicted_mu,

                        "uncertainty_sigma":
                            predicted_sigma,

                        "acquisition_score":
                            acquisition_score,

                        # This label appears only because the
                        # candidate has now been selected/revealed.
                        "revealed_DMS_score":
                            float(
                                true_score
                            ),
                    }
                )


            # -----------------------------------------------
            # Retrospective evaluation
            # -----------------------------------------------

            cumulative_eval = (
                oracle.evaluate_selected(
                    acquired_mutants
                )
            )


            best_multi = (
                cumulative_eval[
                    "best_acquired_multi"
                ]
            )


            best_overall = max(
                initial_best_single,
                best_multi,
            )


            selected_mu_values = (
                selected[
                    "predicted_fitness_mu"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )


            selected_sigma_values = (
                selected[
                    "uncertainty_sigma"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )


            mean_selected_mu = (
                float(
                    np.nanmean(
                        selected_mu_values
                    )
                )
                if np.isfinite(
                    selected_mu_values
                ).any()
                else np.nan
            )


            mean_selected_sigma = (
                float(
                    np.nanmean(
                        selected_sigma_values
                    )
                )
                if np.isfinite(
                    selected_sigma_values
                ).any()
                else np.nan
            )


            mean_selected_k = float(
                selected[
                    "mutation_count"
                ]
                .mean()
            )


            round_row = {
                "simulation_seed":
                    int(
                        simulation_seed
                    ),

                "policy":
                    policy_name,

                "round":
                    int(
                        round_index
                    ),

                "budget_used":
                    int(
                        len(
                            acquired_mutants
                        )
                    ),

                "n_labeled_total":
                    int(
                        len(
                            labeled_scores
                        )
                    ),

                "initial_best_single":
                    initial_best_single,

                "best_overall_including_initial":
                    float(
                        best_overall
                    ),

                "improvement_over_initial_best_single":
                    float(
                        best_overall
                        - initial_best_single
                    ),

                "mean_selected_mu_this_round":
                    mean_selected_mu,

                "mean_selected_sigma_this_round":
                    mean_selected_sigma,

                "mean_selected_mutation_count_this_round":
                    mean_selected_k,

                "mean_final_train_loss_scaled":
                    (
                        float(
                            mean_train_loss
                        )
                        if np.isfinite(
                            mean_train_loss
                        )
                        else np.nan
                    ),

                **cumulative_eval,
            }


            all_round_rows.append(
                round_row
            )


            print(
                f"  acquired budget = "
                f"{len(acquired_mutants):3d}"
            )

            print(
                f"  best acquired multi = "
                f"{best_multi:.6f}"
            )

            print(
                f"  top100 recovered    = "
                f"{cumulative_eval['top100_recovered']}"
            )

            print(
                f"  top1% hits          = "
                f"{cumulative_eval['top_percent_hits']}"
            )


# ============================================================
# Save raw trajectories
# ============================================================

acquisition_df = pd.DataFrame(
    all_acquisition_rows
)


round_df = pd.DataFrame(
    all_round_rows
)


acquisition_df.to_csv(
    ACQUISITION_HISTORY_PATH,
    index=False,
)


round_df.to_csv(
    ROUND_METRICS_PATH,
    index=False,
)


# ============================================================
# Aggregate across simulation seeds
#
# For each policy x budget:
# mean/std across independent simulation seeds.
# ============================================================

aggregate_metric_names = [
    "best_acquired_multi",
    "best_overall_including_initial",
    "improvement_over_initial_best_single",
    "mean_acquired_fitness",
    "top5_acquired_mean",
    "regret_to_global_multi_best",
    "top_percent_hits",
    "top_percent_hit_rate",
    "top10_recovered",
    "top10_recovery_fraction",
    "top50_recovered",
    "top50_recovery_fraction",
    "top100_recovered",
    "top100_recovery_fraction",
]


aggregate_rows = []


for (
    policy_name,
    budget_used,
), group in round_df.groupby(
    [
        "policy",
        "budget_used",
    ],
    sort=False,
):

    row = {
        "policy":
            policy_name,

        "budget_used":
            int(
                budget_used
            ),

        "n_simulations":
            int(
                len(
                    group
                )
            ),
    }


    for metric_name in aggregate_metric_names:

        values = (
            group[
                metric_name
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        finite_values = values[
            np.isfinite(
                values
            )
        ]


        if len(
            finite_values
        ) == 0:

            row[
                f"{metric_name}_mean"
            ] = np.nan

            row[
                f"{metric_name}_std"
            ] = np.nan

        else:

            row[
                f"{metric_name}_mean"
            ] = float(
                finite_values.mean()
            )


            row[
                f"{metric_name}_std"
            ] = (
                float(
                    finite_values.std(
                        ddof=1
                    )
                )
                if len(
                    finite_values
                ) > 1
                else 0.0
            )


    aggregate_rows.append(
        row
    )


aggregate_df = pd.DataFrame(
    aggregate_rows
)


aggregate_df.to_csv(
    AGGREGATE_METRICS_PATH,
    index=False,
)


# ============================================================
# Final-budget policy summary
# ============================================================

final_rows = (
    round_df[
        round_df[
            "budget_used"
        ]
        == BUDGET
    ]
    .copy()
)


final_summary_rows = []


for policy_name, group in final_rows.groupby(
    "policy",
    sort=False,
):

    row = {
        "policy":
            policy_name,

        "n_simulations":
            int(
                len(
                    group
                )
            ),
    }


    for metric_name in [
        "best_acquired_multi",
        "improvement_over_initial_best_single",
        "mean_acquired_fitness",
        "top5_acquired_mean",
        "top_percent_hits",
        "top_percent_hit_rate",
        "top10_recovered",
        "top50_recovered",
        "top100_recovered",
        "regret_to_global_multi_best",
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
        ] = (
            float(
                values.std(
                    ddof=1
                )
            )
            if len(
                values
            ) > 1
            else 0.0
        )


    final_summary_rows.append(
        row
    )


final_summary_df = pd.DataFrame(
    final_summary_rows
)


final_summary_df.to_csv(
    FINAL_POLICY_SUMMARY_PATH,
    index=False,
)


# ============================================================
# Figures
#
# One chart per figure.
# ============================================================

def plot_metric_by_budget(
    metric_base_name,
    ylabel,
    title,
    filename,
):
    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )


    for policy_name in POLICIES:

        policy_data = (
            aggregate_df[
                aggregate_df[
                    "policy"
                ]
                == policy_name
            ]
            .sort_values(
                "budget_used"
            )
        )


        x = (
            policy_data[
                "budget_used"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        y_mean = (
            policy_data[
                f"{metric_base_name}_mean"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        y_std = (
            policy_data[
                f"{metric_base_name}_std"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        ax.plot(
            x,
            y_mean,
            marker="o",
            linewidth=2,
            label=policy_name,
        )


        if len(
            SIMULATION_SEEDS
        ) > 1:

            ax.fill_between(
                x,
                y_mean - y_std,
                y_mean + y_std,
                alpha=0.12,
            )


    ax.set_xlabel(
        "Acquisition budget"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
    )


    ax.legend()


    fig.tight_layout()


    fig.savefig(
        FIGURE_ROOT
        / filename,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


plot_metric_by_budget(
    metric_base_name=
        "best_acquired_multi",

    ylabel=
        "Best acquired multi-mutant DMS_score",

    title=
        "GFP Virtual Evolution: Best Acquired Multi-Mutant",

    filename=
        "best_acquired_multi_vs_budget.png",
)


plot_metric_by_budget(
    metric_base_name=
        "top5_acquired_mean",

    ylabel=
        "Mean DMS_score of top-5 acquired variants",

    title=
        "GFP Virtual Evolution: Top-5 Acquired Fitness",

    filename=
        "top5_acquired_mean_vs_budget.png",
)


plot_metric_by_budget(
    metric_base_name=
        "top100_recovery_fraction",

    ylabel=
        "Fraction of global top-100 recovered",

    title=
        "GFP Virtual Evolution: Top-100 Recovery",

    filename=
        "top100_recovery_vs_budget.png",
)


plot_metric_by_budget(
    metric_base_name=
        "top_percent_hit_rate",

    ylabel=
        "Fraction of acquired variants in global top 1%",

    title=
        "GFP Virtual Evolution: Top-1% Hit Rate",

    filename=
        "top1pct_hit_rate_vs_budget.png",
)


# ============================================================
# Summary JSON
# ============================================================

summary = {
    "stage":
        "Step 8-6",

    "protocol": {
        "type":
            (
                "retrospective pool-based "
                "batch active learning"
            ),

        "initial_labeled_set":
            (
                "1084 GFP single mutants"
            ),

        "candidate_pool":
            (
                "50630 GFP multi-mutants"
            ),

        "initial_labels_count_toward_budget":
            False,

        "budget":
            BUDGET,

        "batch_size":
            BATCH_SIZE_ACQUISITION,

        "rounds":
            N_ROUNDS,

        "policies":
            {
                "random":
                    "uniform random",

                "greedy":
                    "mu",

                "ucb":
                    f"mu + {UCB_BETA} * sigma",

                "conservative":
                    (
                        f"mu - "
                        f"{CONSERVATIVE_BETA} * sigma"
                    ),
            },

        "simulation_seeds":
            SIMULATION_SEEDS,

        "ensemble": {
            "architecture":
                (
                    "BranchFusionMLP: "
                    "WT[960]->64; "
                    "DeltaLocal[960]->64; "
                    "DeltaGlobal[960]->64; "
                    "concat[192]->64->1"
                ),

            "members":
                ENSEMBLE_SIZE,

            "epochs_per_member":
                FINAL_EPOCHS,

            "member_base_seeds":
                MEMBER_BASE_SEEDS,

            "scaler_refit_each_round":
                True,

            "ensemble_retrained_each_round":
                True,
        },
    },

    "leakage_control": {
        "policy_receives_dms_score":
            False,

        "policy_receives_only":
            [
                "mutant",
                "mutation_count",
                "predicted_fitness_mu",
                "uncertainty_sigma",
            ],

        "label_reveal":
            (
                "HiddenOracle.reveal() is called "
                "only after a batch has been selected."
            ),

        "full_hidden_labels_used_for":
            (
                "retrospective evaluation metrics only"
            ),
    },

    "evaluation": {
        "initial_best_single":
            initial_best_single,

        "global_best_multi":
            oracle.global_best_multi,

        "top_percent_fraction":
            TOP_PERCENT_FRACTION,

        "top_percent_count":
            oracle.top_percent_n,

        "top_percent_threshold":
            oracle.top_percent_threshold,

        "top_k_values":
            TOP_K_VALUES,
    },

    "artifacts": {
        "acquisition_history":
            str(
                ACQUISITION_HISTORY_PATH
            ),

        "round_metrics":
            str(
                ROUND_METRICS_PATH
            ),

        "aggregate_metrics":
            str(
                AGGREGATE_METRICS_PATH
            ),

        "final_policy_summary":
            str(
                FINAL_POLICY_SUMMARY_PATH
            ),

        "figures":
            [
                str(
                    FIGURE_ROOT
                    / "best_acquired_multi_vs_budget.png"
                ),

                str(
                    FIGURE_ROOT
                    / "top5_acquired_mean_vs_budget.png"
                ),

                str(
                    FIGURE_ROOT
                    / "top100_recovery_vs_budget.png"
                ),

                str(
                    FIGURE_ROOT
                    / "top1pct_hit_rate_vs_budget.png"
                ),
            ],
    },
}


with open(
    SUMMARY_JSON_PATH,
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
    "=" * 130
)

print(
    "GFP Virtual Evolution Final Summary"
)

print(
    "=" * 130
)


display_columns = [
    "policy",
    "n_simulations",
    "best_acquired_multi_mean",
    "best_acquired_multi_std",
    "improvement_over_initial_best_single_mean",
    "mean_acquired_fitness_mean",
    "top5_acquired_mean_mean",
    "top_percent_hits_mean",
    "top_percent_hit_rate_mean",
    "top10_recovered_mean",
    "top50_recovered_mean",
    "top100_recovered_mean",
    "regret_to_global_multi_best_mean",
]


print(
    final_summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print()
print(
    "Evaluation references"
)

print(
    f"Initial best single : "
    f"{initial_best_single:.6f}"
)

print(
    f"Global best multi   : "
    f"{oracle.global_best_multi:.6f}"
)

print(
    f"Global top 1%       : "
    f"{oracle.top_percent_n} candidates"
)

print(
    f"Top 1% threshold    : "
    f"{oracle.top_percent_threshold:.6f}"
)


print()
print(
    "Saved:"
)

print(
    ACQUISITION_HISTORY_PATH
)

print(
    ROUND_METRICS_PATH
)

print(
    AGGREGATE_METRICS_PATH
)

print(
    FINAL_POLICY_SUMMARY_PATH
)

print(
    SUMMARY_JSON_PATH
)

print(
    FIGURE_ROOT
    / "best_acquired_multi_vs_budget.png"
)

print(
    FIGURE_ROOT
    / "top5_acquired_mean_vs_budget.png"
)

print(
    FIGURE_ROOT
    / "top100_recovery_vs_budget.png"
)

print(
    FIGURE_ROOT
    / "top1pct_hit_rate_vs_budget.png"
)


print()
print(
    "IMPORTANT:"
)

print(
    "Multi-mutant DMS_score was loaded only "
    "inside HiddenOracle for retrospective "
    "reveal/evaluation."
)

print(
    "Acquisition policies never receive "
    "unrevealed DMS_score values."
)

print(
    "This is pool-based active learning, "
    "not a strict one-mutation-path simulation."
)
