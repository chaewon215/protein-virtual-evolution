from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from esm.models.esmc import EsmcModel, EsmcTokenizer
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Step 9-2
# Score ESM3-generated GFP candidates
#
# Pipeline
# --------
# Generated GFP sequences
#     ↓
# ESM-C 300M
#     ↓
# WT context mean       [N, 960]
# Delta local sum       [N, 960]
# Delta global          [N, 960]
#     ↓ concatenate
# X                     [N, 2880]
#     ↓ frozen scaler
#     ↓
# 5 frozen BranchFusionMLP members
#     ↓
# member predictions    [N, 5]
#     ↓
# mu                    [N]
# sigma                 [N]
#
# IMPORTANT
# ---------
# No ProteinGym multi-mutant DMS_score is loaded.
# Generated candidates have no experimental labels.
# ============================================================


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WT_FASTA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_wt.fasta"
)

GENERATED_PATH = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "esm3_generation"
    / "esm3_generated_candidates.csv"
)

FINAL_ENSEMBLE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "final_ensemble"
)

SCALER_PATH = (
    FINAL_ENSEMBLE_ROOT
    / "scalers.pkl"
)

CHECKPOINT_ROOT = (
    FINAL_ENSEMBLE_ROOT
    / "checkpoints"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "esm3_scoring"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SCORED_PATH = (
    OUTPUT_ROOT
    / "esm3_generated_candidates_scored.csv"
)

FEATURE_PATH = (
    OUTPUT_ROOT
    / "esm3_generated_features.pt"
)

MEMBER_PREDICTIONS_PATH = (
    OUTPUT_ROOT
    / "esm3_member_predictions.npy"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


# ============================================================
# Frozen settings
# ============================================================

ESMC_MODEL_NAME = "biohub/ESMC-300M"

ESMC_HIDDEN_DIM = 960

ESMC_BATCH_SIZE = 8

PRED_BATCH_SIZE = 256

ENSEMBLE_SIZE = 5

DROPOUT = 0.2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
# Utilities
# ============================================================

AA20 = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


def read_fasta_sequence(
    path: Path,
):
    sequence = "".join(
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if (
            line.strip()
            and not line.startswith(
                ">"
            )
        )
    )

    if not sequence:
        raise ValueError(
            f"Empty FASTA: {path}"
        )

    if not set(
        sequence
    ).issubset(
        AA20
    ):
        raise ValueError(
            "WT sequence contains "
            "non-standard amino acids."
        )

    return sequence


def mutation_positions_from_sequences(
    wt_sequence: str,
    mutant_sequence: str,
):
    if len(
        wt_sequence
    ) != len(
        mutant_sequence
    ):
        raise ValueError(
            "WT / mutant sequence length mismatch."
        )

    positions = [
        idx
        for idx, (
            wt_aa,
            mut_aa,
        ) in enumerate(
            zip(
                wt_sequence,
                mutant_sequence,
            )
        )
        if wt_aa != mut_aa
    ]

    return positions


# ============================================================
# Frozen Branch Fusion model
#
# Input:
#   [B, 2880]
#
# WT branch:
#   [B,960] -> [B,64]
#
# Delta-local branch:
#   [B,960] -> [B,64]
#
# Delta-global branch:
#   [B,960] -> [B,64]
#
# concatenate:
#   [B,192]
#
# head:
#   [B,192] -> [B,64] -> [B,1]
#
# Output:
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


# ============================================================
# ESM-C embedding
#
# Sequence length GFP:
#   L = 238
#
# Token representation:
#   [B, L+2, 960]
#   = [B, 240, 960]
#
# after removing BOS/EOS:
#   [B, 238, 960]
# ============================================================

@torch.inference_mode()
def embed_sequences(
    sequences,
    tokenizer,
    model,
    sequence_length,
):
    all_residue_embeddings = []

    for start in range(
        0,
        len(
            sequences
        ),
        ESMC_BATCH_SIZE,
    ):
        batch_sequences = sequences[
            start:
            start
            + ESMC_BATCH_SIZE
        ]

        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key:
                value.to(
                    DEVICE
                )
            for key, value
            in inputs.items()
        }

        output = model(
            **inputs
        )

        hidden = (
            output.last_hidden_state
        )

        # All GFP sequences have identical length 238.
        # Token layout:
        # BOS + 238 residues + EOS
        expected_token_length = (
            sequence_length
            + 2
        )

        if hidden.shape[
            1
        ] != expected_token_length:

            raise ValueError(
                "Unexpected ESM-C token dimension: "
                f"{tuple(hidden.shape)}; "
                f"expected token length "
                f"{expected_token_length}"
            )

        residue_hidden = hidden[
            :,
            1:
            1
            + sequence_length,
            :
        ]

        expected_shape = (
            len(
                batch_sequences
            ),
            sequence_length,
            ESMC_HIDDEN_DIM,
        )

        if tuple(
            residue_hidden.shape
        ) != expected_shape:

            raise ValueError(
                "Unexpected residue embedding shape: "
                f"{tuple(residue_hidden.shape)} "
                f"!= {expected_shape}"
            )

        all_residue_embeddings.append(
            residue_hidden
            .float()
            .cpu()
        )

        print(
            f"Embedded "
            f"{min(start + len(batch_sequences), len(sequences))}"
            f"/{len(sequences)}"
        )

    return torch.cat(
        all_residue_embeddings,
        dim=0,
    )


# ============================================================
# Ensemble prediction
# ============================================================

@torch.inference_mode()
def predict_model(
    model,
    X_scaled,
):
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(
            X_scaled.astype(
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
# Main
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "Step 9-2: Score ESM3-Generated GFP Candidates"
    )

    print(
        "=" * 110
    )

    print(
        f"Device: {DEVICE}"
    )


    # ========================================================
    # Load WT
    # ========================================================

    wt_sequence = read_fasta_sequence(
        WT_FASTA_PATH
    )

    sequence_length = len(
        wt_sequence
    )


    # ========================================================
    # Load generated candidates
    # ========================================================

    generated_df = pd.read_csv(
        GENERATED_PATH
    )


    required_columns = {
        "candidate_id",
        "seed_mutant",
        "seed_mu",
        "seed_sigma",
        "seed_conservative_score",
        "generated_mutant",
        "mutation_count",
        "generated_sequence",
    }


    missing_columns = (
        required_columns
        - set(
            generated_df.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "Generated candidate file is missing: "
            f"{sorted(missing_columns)}"
        )


    if len(
        generated_df
    ) == 0:

        raise ValueError(
            "No generated candidates found."
        )


    generated_df[
        "generated_sequence"
    ] = generated_df[
        "generated_sequence"
    ].astype(
        str
    )


    if (
        generated_df[
            "generated_sequence"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate generated sequences found."
        )


    sequences = (
        generated_df[
            "generated_sequence"
        ]
        .tolist()
    )


    for candidate_id, sequence in zip(
        generated_df[
            "candidate_id"
        ],
        sequences,
    ):

        if len(
            sequence
        ) != sequence_length:

            raise ValueError(
                f"{candidate_id}: "
                f"sequence length {len(sequence)} "
                f"!= WT length {sequence_length}"
            )

        if not set(
            sequence
        ).issubset(
            AA20
        ):

            raise ValueError(
                f"{candidate_id}: "
                "non-standard amino acid found."
            )


    n_candidates = len(
        sequences
    )


    print()
    print(
        "Generated candidate input"
    )

    print(
        f"N candidates        : "
        f"{n_candidates}"
    )

    print(
        f"Sequence length      : "
        f"{sequence_length}"
    )


    # ========================================================
    # Load ESM-C
    # ========================================================

    print()
    print(
        f"Loading ESM-C: "
        f"{ESMC_MODEL_NAME}"
    )


    tokenizer = (
        EsmcTokenizer.from_pretrained(
            ESMC_MODEL_NAME
        )
    )


    esmc_model = (
        EsmcModel.from_pretrained(
            ESMC_MODEL_NAME
        )
        .to(
            DEVICE
        )
    )


    esmc_model.eval()


    # ========================================================
    # WT embedding
    #
    # raw ESM-C:
    # [1, 240, 960]
    #
    # residues:
    # [1, 238, 960]
    #
    # remove batch:
    # [238, 960]
    # ========================================================

    print()
    print(
        "Embedding WT..."
    )


    wt_residue_embeddings = embed_sequences(
        [
            wt_sequence
        ],
        tokenizer,
        esmc_model,
        sequence_length,
    )[
        0
    ]


    if tuple(
        wt_residue_embeddings.shape
    ) != (
        sequence_length,
        ESMC_HIDDEN_DIM,
    ):

        raise ValueError(
            "Unexpected WT embedding shape: "
            f"{tuple(wt_residue_embeddings.shape)}"
        )


    wt_global = (
        wt_residue_embeddings.mean(
            dim=0
        )
    )


    # ========================================================
    # Generated candidate ESM-C embeddings
    #
    # [N, 238, 960]
    # ========================================================

    print()
    print(
        "Embedding generated candidates..."
    )


    mutant_residue_embeddings = embed_sequences(
        sequences,
        tokenizer,
        esmc_model,
        sequence_length,
    )


    del esmc_model


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # ========================================================
    # Construct final features
    #
    # For candidate i with K_i mutated positions:
    #
    # WT positions:
    #   [K_i, 960]
    #       ↓ mean
    #   [960]
    #
    # local deltas:
    #   [K_i, 960]
    #       ↓ sum
    #   [960]
    #
    # global delta:
    #   mean(mutant residues) - mean(WT residues)
    #   [960]
    #
    # concatenate:
    #   [2880]
    #
    # all candidates:
    #   [N, 2880]
    # ========================================================

    wt_context_features = []

    delta_local_features = []

    delta_global_features = []

    validated_mutation_counts = []


    for candidate_idx, sequence in enumerate(
        sequences
    ):

        mutation_positions = (
            mutation_positions_from_sequences(
                wt_sequence,
                sequence,
            )
        )


        if len(
            mutation_positions
        ) == 0:

            raise ValueError(
                "Generated candidate equals WT: "
                f"{generated_df.iloc[candidate_idx]['candidate_id']}"
            )


        cached_k = int(
            generated_df.iloc[
                candidate_idx
            ][
                "mutation_count"
            ]
        )


        if len(
            mutation_positions
        ) != cached_k:

            raise ValueError(
                f"Mutation count mismatch for "
                f"{generated_df.iloc[candidate_idx]['candidate_id']}: "
                f"sequence={len(mutation_positions)}, "
                f"CSV={cached_k}"
            )


        position_tensor = torch.tensor(
            mutation_positions,
            dtype=torch.long,
        )


        wt_at_positions = (
            wt_residue_embeddings[
                position_tensor
            ]
        )


        mutant_at_positions = (
            mutant_residue_embeddings[
                candidate_idx,
                position_tensor,
                :
            ]
        )


        wt_context = (
            wt_at_positions.mean(
                dim=0
            )
        )


        delta_local = (
            mutant_at_positions
            - wt_at_positions
        ).sum(
            dim=0
        )


        mutant_global = (
            mutant_residue_embeddings[
                candidate_idx
            ]
            .mean(
                dim=0
            )
        )


        delta_global = (
            mutant_global
            - wt_global
        )


        wt_context_features.append(
            wt_context
        )


        delta_local_features.append(
            delta_local
        )


        delta_global_features.append(
            delta_global
        )


        validated_mutation_counts.append(
            len(
                mutation_positions
            )
        )


    wt_context_features = torch.stack(
        wt_context_features,
        dim=0,
    )


    delta_local_features = torch.stack(
        delta_local_features,
        dim=0,
    )


    delta_global_features = torch.stack(
        delta_global_features,
        dim=0,
    )


    X_raw = torch.cat(
        [
            wt_context_features,
            delta_local_features,
            delta_global_features,
        ],
        dim=1,
    )


    expected_feature_shape = (
        n_candidates,
        2880,
    )


    if tuple(
        X_raw.shape
    ) != expected_feature_shape:

        raise ValueError(
            "Unexpected final feature shape: "
            f"{tuple(X_raw.shape)} "
            f"!= {expected_feature_shape}"
        )


    print()
    print(
        "Feature shapes"
    )

    print(
        f"WT context      : "
        f"{tuple(wt_context_features.shape)}"
    )

    print(
        f"Delta local     : "
        f"{tuple(delta_local_features.shape)}"
    )

    print(
        f"Delta global    : "
        f"{tuple(delta_global_features.shape)}"
    )

    print(
        f"Final X         : "
        f"{tuple(X_raw.shape)}"
    )


    # ========================================================
    # Save extracted features
    # ========================================================

    torch.save(
        {
            "candidate_id":
                generated_df[
                    "candidate_id"
                ]
                .astype(
                    str
                )
                .tolist(),

            "wt_context_mean":
                wt_context_features,

            "delta_local_sum":
                delta_local_features,

            "delta_global":
                delta_global_features,

            "X":
                X_raw,

            "mutation_count":
                validated_mutation_counts,

            "esmc_model":
                ESMC_MODEL_NAME,
        },
        FEATURE_PATH,
    )


    # ========================================================
    # Load frozen scalers
    # ========================================================

    with open(
        SCALER_PATH,
        "rb",
    ) as f:

        scaler_bundle = pickle.load(
            f
        )


    x_scaler = scaler_bundle[
        "x_scaler"
    ]


    y_scaler = scaler_bundle[
        "y_scaler"
    ]


    X_scaled = (
        x_scaler.transform(
            X_raw.numpy()
        )
    )


    # ========================================================
    # Load frozen ensemble checkpoints
    #
    # X:
    # [N, 2880]
    #
    # member outputs:
    # [N] × 5
    #
    # stack:
    # [N, 5]
    # ========================================================

    member_predictions = []


    for member_idx in range(
        ENSEMBLE_SIZE
    ):

        checkpoint_path = (
            CHECKPOINT_ROOT
            / (
                f"branch_fusion_member_"
                f"{member_idx}.pt"
            )
        )


        if not checkpoint_path.exists():

            raise FileNotFoundError(
                checkpoint_path
            )


        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )


        model = (
            BranchFusionMLP()
            .to(
                DEVICE
            )
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        pred_scaled = predict_model(
            model,
            X_scaled,
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


        print(
            f"Scored member "
            f"{member_idx + 1}/{ENSEMBLE_SIZE}"
        )


        del model


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    member_predictions = np.column_stack(
        member_predictions
    )


    expected_member_shape = (
        n_candidates,
        ENSEMBLE_SIZE,
    )


    if member_predictions.shape != (
        expected_member_shape
    ):

        raise ValueError(
            "Unexpected ensemble output shape: "
            f"{member_predictions.shape}"
        )


    np.save(
        MEMBER_PREDICTIONS_PATH,
        member_predictions,
    )


    # ========================================================
    # Ensemble score
    # ========================================================

    mu = member_predictions.mean(
        axis=1
    )


    sigma = member_predictions.std(
        axis=1,
        ddof=1,
    )


    conservative_score = (
        mu
        - sigma
    )


    ucb_score = (
        mu
        + sigma
    )


    # ========================================================
    # Compare generated candidates against their seed
    #
    # This is prediction-vs-prediction only.
    # There are NO experimental labels for generated variants.
    # ========================================================

    scored_df = generated_df.copy()


    scored_df[
        "predicted_fitness_mu"
    ] = mu


    scored_df[
        "uncertainty_sigma"
    ] = sigma


    scored_df[
        "conservative_score"
    ] = conservative_score


    scored_df[
        "ucb_score"
    ] = ucb_score


    scored_df[
        "predicted_delta_mu_vs_seed"
    ] = (
        scored_df[
            "predicted_fitness_mu"
        ]
        - scored_df[
            "seed_mu"
        ]
    )


    scored_df[
        "delta_conservative_vs_seed"
    ] = (
        scored_df[
            "conservative_score"
        ]
        - scored_df[
            "seed_conservative_score"
        ]
    )


    for member_idx in range(
        ENSEMBLE_SIZE
    ):

        scored_df[
            f"member_{member_idx}_prediction"
        ] = member_predictions[
            :,
            member_idx
        ]


    # ========================================================
    # Rank candidates
    # ========================================================

    scored_df[
        "rank_by_mu"
    ] = (
        scored_df[
            "predicted_fitness_mu"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            int
        )
    )


    scored_df[
        "rank_by_conservative"
    ] = (
        scored_df[
            "conservative_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            int
        )
    )


    scored_df = (
        scored_df.sort_values(
            [
                "conservative_score",
                "predicted_fitness_mu",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    scored_df.to_csv(
        SCORED_PATH,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    n_mu_improved = int(
        (
            scored_df[
                "predicted_delta_mu_vs_seed"
            ]
            > 0
        ).sum()
    )


    n_conservative_improved = int(
        (
            scored_df[
                "delta_conservative_vs_seed"
            ]
            > 0
        ).sum()
    )


    summary = {
        "stage":
            "Step 9-2",

        "label_policy": {
            "experimental_labels_for_generated_candidates":
                False,

            "proteingym_multi_mutant_dms_score_loaded":
                False,
        },

        "input": {
            "n_candidates":
                int(
                    n_candidates
                ),

            "sequence_length":
                int(
                    sequence_length
                ),
        },

        "esmc": {
            "model":
                ESMC_MODEL_NAME,

            "residue_embedding_shape":
                [
                    int(
                        n_candidates
                    ),
                    int(
                        sequence_length
                    ),
                    960,
                ],

            "wt_context_shape":
                [
                    int(
                        n_candidates
                    ),
                    960,
                ],

            "delta_local_shape":
                [
                    int(
                        n_candidates
                    ),
                    960,
                ],

            "delta_global_shape":
                [
                    int(
                        n_candidates
                    ),
                    960,
                ],

            "final_feature_shape":
                [
                    int(
                        n_candidates
                    ),
                    2880,
                ],
        },

        "ensemble": {
            "members":
                ENSEMBLE_SIZE,

            "member_prediction_shape":
                [
                    int(
                        n_candidates
                    ),
                    ENSEMBLE_SIZE,
                ],

            "mu_shape":
                [
                    int(
                        n_candidates
                    )
                ],

            "sigma_shape":
                [
                    int(
                        n_candidates
                    )
                ],
        },

        "prediction_distribution": {
            "mu_min":
                float(
                    mu.min()
                ),

            "mu_median":
                float(
                    np.median(
                        mu
                    )
                ),

            "mu_mean":
                float(
                    mu.mean()
                ),

            "mu_max":
                float(
                    mu.max()
                ),

            "sigma_min":
                float(
                    sigma.min()
                ),

            "sigma_median":
                float(
                    np.median(
                        sigma
                    )
                ),

            "sigma_mean":
                float(
                    sigma.mean()
                ),

            "sigma_max":
                float(
                    sigma.max()
                ),
        },

        "seed_comparison": {
            "n_predicted_mu_improved_vs_seed":
                n_mu_improved,

            "fraction_predicted_mu_improved_vs_seed":
                float(
                    n_mu_improved
                    / n_candidates
                ),

            "n_conservative_improved_vs_seed":
                n_conservative_improved,

            "fraction_conservative_improved_vs_seed":
                float(
                    n_conservative_improved
                    / n_candidates
                ),
        },

        "interpretation":
            (
                "All generated-candidate values are "
                "model predictions. They are not "
                "experimental fitness measurements."
            ),

        "artifacts": {
            "features":
                str(
                    FEATURE_PATH
                ),

            "member_predictions":
                str(
                    MEMBER_PREDICTIONS_PATH
                ),

            "scored_candidates":
                str(
                    SCORED_PATH
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


    # ========================================================
    # Console report
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "ESM3-Generated GFP Candidate Scoring Summary"
    )

    print(
        "=" * 120
    )


    print()
    print(
        "Tensor shapes"
    )

    print(
        f"Generated sequences : "
        f"({n_candidates}, {sequence_length} aa)"
    )

    print(
        f"ESM-C residues      : "
        f"{tuple(mutant_residue_embeddings.shape)}"
    )

    print(
        f"WT context          : "
        f"{tuple(wt_context_features.shape)}"
    )

    print(
        f"Delta local         : "
        f"{tuple(delta_local_features.shape)}"
    )

    print(
        f"Delta global        : "
        f"{tuple(delta_global_features.shape)}"
    )

    print(
        f"Final X             : "
        f"{tuple(X_raw.shape)}"
    )

    print(
        f"Member predictions  : "
        f"{member_predictions.shape}"
    )

    print(
        f"mu                   : "
        f"{mu.shape}"
    )

    print(
        f"sigma                : "
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
        "Predicted improvement over seed"
    )

    print(
        f"mu improved           : "
        f"{n_mu_improved}/{n_candidates} "
        f"({n_mu_improved / n_candidates:.1%})"
    )

    print(
        f"conservative improved : "
        f"{n_conservative_improved}/{n_candidates} "
        f"({n_conservative_improved / n_candidates:.1%})"
    )


    print()
    print(
        "Top 15 by conservative score"
    )


    display_columns = [
        "candidate_id",
        "seed_mutant",
        "generated_mutant",
        "mutation_count",
        "predicted_fitness_mu",
        "uncertainty_sigma",
        "conservative_score",
        "predicted_delta_mu_vs_seed",
        "delta_conservative_vs_seed",
    ]


    print(
        scored_df[
            display_columns
        ]
        .head(
            15
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
        FEATURE_PATH
    )

    print(
        MEMBER_PREDICTIONS_PATH
    )

    print(
        SCORED_PATH
    )

    print(
        SUMMARY_PATH
    )


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No experimental labels exist for "
        "these ESM3-generated candidates."
    )

    print(
        "predicted_fitness_mu and "
        "uncertainty_sigma are outputs "
        "of the frozen Step-8 ensemble."
    )

    print(
        "A candidate with higher predicted "
        "fitness is NOT experimentally "
        "validated as improved GFP."
    )


if __name__ == "__main__":
    main()
