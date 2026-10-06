from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from esm.models.esmc import EsmcModel, EsmcTokenizer


# ============================================================
# Step 11-2
# TPMT ESM-C feature extraction
#
# Final frozen representation from GFP:
#
#   WT local        [960]
#   Delta local     [960]
#   Delta global    [960]
#        ↓ concat
#   Feature         [2880]
#
# For N TPMT single mutants:
#
#   X               [N, 2880]
#
# IMPORTANT
# ---------
# DMS_score is NOT loaded in this script.
# Representation extraction is label-free.
# ============================================================


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tpmt_single_mutants.csv"
)

WT_FASTA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tpmt_wt.fasta"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esmc300m_tpmt"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

WT_EMBEDDINGS_PATH = (
    OUTPUT_ROOT
    / "tpmt_wt_embeddings.pt"
)

FEATURE_PATH = (
    OUTPUT_ROOT
    / "tpmt_variant_features.pt"
)

METADATA_PATH = (
    OUTPUT_ROOT
    / "tpmt_variant_feature_metadata.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "feature_summary.json"
)


# ============================================================
# Model / runtime
# ============================================================

MODEL_NAME = "biohub/ESMC-300M"

HIDDEN_DIM = 960

BATCH_SIZE = 16

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
            and not line.startswith(">")
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
            "WT FASTA contains non-standard amino acids."
        )

    return sequence


@torch.inference_mode()
def embed_sequences(
    sequences,
    tokenizer,
    model,
    sequence_length,
):
    """
    Input:
        sequences:
            list[str], length B

    ESM-C tokenizer output:
        input_ids
            [B, L+2]
            for TPMT L=245 -> [B,247]

    ESM-C hidden:
        [B, L+2, 960]
        -> TPMT [B,247,960]

    Remove BOS/EOS:
        [B, L, 960]
        -> TPMT [B,245,960]
    """

    inputs = tokenizer(
        sequences,
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


    expected_token_length = (
        sequence_length
        + 2
    )


    if hidden.shape[
        1
    ] != expected_token_length:

        raise ValueError(
            "Unexpected token dimension: "
            f"{tuple(hidden.shape)}; "
            f"expected sequence axis "
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
            sequences
        ),
        sequence_length,
        HIDDEN_DIM,
    )


    if tuple(
        residue_hidden.shape
    ) != expected_shape:

        raise ValueError(
            "Unexpected residue embedding shape: "
            f"{tuple(residue_hidden.shape)} "
            f"!= {expected_shape}"
        )


    return (
        residue_hidden
        .float()
        .cpu()
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "Step 11-2: TPMT ESM-C Feature Extraction"
    )

    print(
        "=" * 110
    )

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Model       : {MODEL_NAME}"
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


    print(
        f"WT length   : {sequence_length}"
    )


    # ========================================================
    # Load LABEL-FREE metadata only
    #
    # DMS_score is intentionally excluded.
    # ========================================================

    header = pd.read_csv(
        INPUT_CSV,
        nrows=0,
    )


    required_columns = {
        "mutant",
        "mutated_sequence",
        "position",
        "python_index",
        "mutation_count",
        "fold_random_5",
        "fold_position_5",
    }


    missing = (
        required_columns
        - set(
            header.columns
        )
    )


    if missing:

        raise ValueError(
            "TPMT processed CSV missing columns: "
            f"{sorted(missing)}"
        )


    df = pd.read_csv(
        INPUT_CSV,
        usecols=[
            "mutant",
            "mutated_sequence",
            "position",
            "python_index",
            "mutation_count",
            "fold_random_5",
            "fold_position_5",
        ],
    )


    if not (
        df[
            "mutation_count"
        ]
        == 1
    ).all():

        raise ValueError(
            "TPMT extraction expects single mutants only."
        )


    if (
        df[
            "mutant"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate mutant IDs found."
        )


    sequences = (
        df[
            "mutated_sequence"
        ]
        .astype(
            str
        )
        .tolist()
    )


    for idx, sequence in enumerate(
        sequences
    ):

        if len(
            sequence
        ) != sequence_length:

            raise ValueError(
                f"Sequence {idx} length mismatch: "
                f"{len(sequence)} != {sequence_length}"
            )


    n_variants = len(
        df
    )


    print(
        f"Variants    : {n_variants:,}"
    )


    # ========================================================
    # Load ESM-C
    # ========================================================

    print()
    print(
        "Loading ESM-C..."
    )


    tokenizer = (
        EsmcTokenizer.from_pretrained(
            MODEL_NAME
        )
    )


    model = (
        EsmcModel.from_pretrained(
            MODEL_NAME
        )
        .to(
            DEVICE
        )
    )


    model.eval()


    # ========================================================
    # WT
    #
    # Raw:
    #   [1,247,960]
    #
    # Residues:
    #   [1,245,960]
    #
    # Remove batch:
    #   [245,960]
    #
    # Global mean:
    #   [960]
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
        model,
        sequence_length,
    )[
        0
    ]


    wt_global = (
        wt_residue_embeddings.mean(
            dim=0
        )
    )


    print(
        f"WT residue embeddings : "
        f"{tuple(wt_residue_embeddings.shape)}"
    )

    print(
        f"WT global embedding   : "
        f"{tuple(wt_global.shape)}"
    )


    torch.save(
        {
            "model_name":
                MODEL_NAME,

            "sequence":
                wt_sequence,

            "residue_embeddings":
                wt_residue_embeddings,

            "global_embedding":
                wt_global,
        },
        WT_EMBEDDINGS_PATH,
    )


    # ========================================================
    # Variant features
    #
    # For each single-mutant i:
    #
    # position p_i
    #
    # WT local:
    #   h_WT(p_i)
    #   [960]
    #
    # Mutant local:
    #   h_MUT_i(p_i)
    #   [960]
    #
    # Delta local:
    #   h_MUT_i(p_i) - h_WT(p_i)
    #   [960]
    #
    # Mutant global:
    #   mean residues
    #   [960]
    #
    # Delta global:
    #   mutant global - WT global
    #   [960]
    #
    # Final:
    #   [WT local, Delta local, Delta global]
    #   [2880]
    # ========================================================

    wt_local_list = []

    mutant_local_list = []

    delta_local_list = []

    mutant_global_list = []

    delta_global_list = []


    for start in range(
        0,
        n_variants,
        BATCH_SIZE,
    ):

        end = min(
            start
            + BATCH_SIZE,
            n_variants,
        )


        batch_sequences = sequences[
            start:end
        ]


        batch_embeddings = embed_sequences(
            batch_sequences,
            tokenizer,
            model,
            sequence_length,
        )


        batch_python_indices = (
            df.iloc[
                start:end
            ][
                "python_index"
            ]
            .astype(
                int
            )
            .to_numpy()
        )


        for local_idx, python_index in enumerate(
            batch_python_indices
        ):

            wt_local = (
                wt_residue_embeddings[
                    python_index
                ]
            )


            mutant_local = (
                batch_embeddings[
                    local_idx,
                    python_index,
                    :
                ]
            )


            delta_local = (
                mutant_local
                - wt_local
            )


            mutant_global = (
                batch_embeddings[
                    local_idx
                ]
                .mean(
                    dim=0
                )
            )


            delta_global = (
                mutant_global
                - wt_global
            )


            wt_local_list.append(
                wt_local
            )


            mutant_local_list.append(
                mutant_local
            )


            delta_local_list.append(
                delta_local
            )


            mutant_global_list.append(
                mutant_global
            )


            delta_global_list.append(
                delta_global
            )


        print(
            f"Embedded "
            f"{end:,}/{n_variants:,}"
        )


    # ========================================================
    # Stack
    # ========================================================

    wt_local = torch.stack(
        wt_local_list,
        dim=0,
    )


    mutant_local = torch.stack(
        mutant_local_list,
        dim=0,
    )


    delta_local = torch.stack(
        delta_local_list,
        dim=0,
    )


    mutant_global = torch.stack(
        mutant_global_list,
        dim=0,
    )


    delta_global = torch.stack(
        delta_global_list,
        dim=0,
    )


    X = torch.cat(
        [
            wt_local,
            delta_local,
            delta_global,
        ],
        dim=1,
    )


    # ========================================================
    # Validate shapes
    # ========================================================

    expected_feature_shape = (
        n_variants,
        2880,
    )


    if tuple(
        X.shape
    ) != expected_feature_shape:

        raise ValueError(
            "Unexpected final feature shape: "
            f"{tuple(X.shape)} "
            f"!= {expected_feature_shape}"
        )


    print()
    print(
        "Final feature shapes"
    )

    print(
        f"WT local       : "
        f"{tuple(wt_local.shape)}"
    )

    print(
        f"Mutant local   : "
        f"{tuple(mutant_local.shape)}"
    )

    print(
        f"Delta local    : "
        f"{tuple(delta_local.shape)}"
    )

    print(
        f"Mutant global  : "
        f"{tuple(mutant_global.shape)}"
    )

    print(
        f"Delta global   : "
        f"{tuple(delta_global.shape)}"
    )

    print(
        f"Final X        : "
        f"{tuple(X.shape)}"
    )


    # ========================================================
    # Save feature tensor cache
    # ========================================================

    torch.save(
        {
            "model_name":
                MODEL_NAME,

            "wt_local":
                wt_local,

            "mutant_local":
                mutant_local,

            "delta_local":
                delta_local,

            "mutant_global":
                mutant_global,

            "delta_global":
                delta_global,

            "X":
                X,
        },
        FEATURE_PATH,
    )


    # ========================================================
    # Save alignment metadata
    #
    # feature_index gives exact row alignment with X.
    # ========================================================

    metadata = df[
        [
            "mutant",
            "position",
            "python_index",
            "mutation_count",
            "fold_random_5",
            "fold_position_5",
        ]
    ].copy()


    metadata.insert(
        0,
        "feature_index",
        np.arange(
            n_variants,
            dtype=np.int64,
        ),
    )


    metadata.to_csv(
        METADATA_PATH,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "stage":
            "Step 11-2",

        "protein":
            "TPMT",

        "model":
            MODEL_NAME,

        "label_free_extraction":
            True,

        "n_variants":
            int(
                n_variants
            ),

        "sequence_length":
            int(
                sequence_length
            ),

        "tensor_shapes": {
            "wt_residue_embeddings":
                list(
                    wt_residue_embeddings.shape
                ),

            "wt_global":
                list(
                    wt_global.shape
                ),

            "wt_local":
                list(
                    wt_local.shape
                ),

            "mutant_local":
                list(
                    mutant_local.shape
                ),

            "delta_local":
                list(
                    delta_local.shape
                ),

            "mutant_global":
                list(
                    mutant_global.shape
                ),

            "delta_global":
                list(
                    delta_global.shape
                ),

            "final_X":
                list(
                    X.shape
                ),
        },

        "feature_definition":
            (
                "[WT local, Delta local, Delta global]"
            ),

        "feature_dim":
            2880,

        "artifacts": {
            "wt_embeddings":
                str(
                    WT_EMBEDDINGS_PATH
                ),

            "variant_features":
                str(
                    FEATURE_PATH
                ),

            "feature_metadata":
                str(
                    METADATA_PATH
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
    # Cleanup
    # ========================================================

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        "TPMT ESM-C Feature Extraction Summary"
    )

    print(
        "=" * 110
    )


    print(
        f"Variants         : "
        f"{n_variants:,}"
    )

    print(
        f"WT length        : "
        f"{sequence_length}"
    )

    print(
        f"WT residues      : "
        f"{tuple(wt_residue_embeddings.shape)}"
    )

    print(
        f"WT local         : "
        f"{tuple(wt_local.shape)}"
    )

    print(
        f"Mutant local     : "
        f"{tuple(mutant_local.shape)}"
    )

    print(
        f"Delta local      : "
        f"{tuple(delta_local.shape)}"
    )

    print(
        f"Mutant global    : "
        f"{tuple(mutant_global.shape)}"
    )

    print(
        f"Delta global     : "
        f"{tuple(delta_global.shape)}"
    )

    print(
        f"Final X          : "
        f"{tuple(X.shape)}"
    )


    print()
    print(
        "Saved:"
    )

    print(
        WT_EMBEDDINGS_PATH
    )

    print(
        FEATURE_PATH
    )

    print(
        METADATA_PATH
    )

    print(
        SUMMARY_PATH
    )


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "DMS_score was not loaded or used."
    )

    print(
        "The TPMT representation is frozen to "
        "the same 2,880-dimensional feature design "
        "selected during the GFP experiments."
    )


if __name__ == "__main__":
    main()
