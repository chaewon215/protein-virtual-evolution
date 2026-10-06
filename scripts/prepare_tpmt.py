from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold


# ============================================================
# Step 11-1
# TPMT ProteinGym preprocessing
#
# Assay:
#   TPMT_HUMAN_Matreyek_2018
#
# Official ProteinGym metadata:
#   Sequence length     : 245 aa
#   Single mutants     : 3,648
#   Multiple mutants   : 0
#   Phenotype          : protein abundance
#   Assay              : GFP-fused target abundance by FACS
#
# Output:
#   data/processed/
#       tpmt_single_mutants.csv
#       tpmt_wt.fasta
#       tpmt_position_summary.csv
#       tpmt_summary.json
#
# Splits generated:
#   fold_random_5
#       ordinary shuffled row-level 5-fold CV
#
#   fold_position_5
#       GroupKFold by mutation position
#       -> substitutions at the same residue never cross
#          train/test boundaries
#
# IMPORTANT:
#   These are OUR reproducible validation splits.
#   fold_position_5 is not claimed to be an official
#   ProteinGym split file.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


ASSAY_ID = (
    "TPMT_HUMAN_Matreyek_2018"
)

ASSAY_FILENAME = (
    "TPMT_HUMAN_Matreyek_2018.csv"
)


# ProteinGym target_seq for TPMT_HUMAN_Matreyek_2018
WT_SEQUENCE = (
    "MDGTRTSLDIEEYSDTEVQKNQVLTLEEWQDKWVNGKTAFHQEQGHQLLKKHL"
    "DTFLKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSY"
    "SEEPITEIPGTKVFKSSSGNISLYCCSIFDLPRTNIGKFDMIWDRGALVAINPG"
    "DRKCYADTMFSLLGKKFQYLLCVLSYDPTKHPGPPFYVPHAEIERLFGKICNIR"
    "CLEKVDAFEERHKSWGIDCLFEKLYLLTEK"
)

EXPECTED_LENGTH = 245

EXPECTED_N_SINGLE = 3648

MUTATION_RE = re.compile(
    r"^([A-Z])(\d+)([A-Z])$"
)


# ============================================================
# Locate input
# ============================================================

def find_assay_csv(
    explicit_path=None,
):
    if explicit_path is not None:

        path = Path(
            explicit_path
        ).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        return path


    candidate_paths = [
        PROJECT_ROOT
        / "data"
        / "raw"
        / ASSAY_FILENAME,

        PROJECT_ROOT
        / "data"
        / "raw"
        / "DMS_ProteinGym_substitutions"
        / ASSAY_FILENAME,

        PROJECT_ROOT
        / "data"
        / "raw"
        / "substitutions"
        / ASSAY_FILENAME,

        PROJECT_ROOT
        / "data"
        / ASSAY_FILENAME,
    ]


    for path in candidate_paths:

        if path.exists():
            return path


    raw_root = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )


    if raw_root.exists():

        matches = list(
            raw_root.rglob(
                ASSAY_FILENAME
            )
        )

        if len(
            matches
        ) == 1:

            return matches[
                0
            ]


        if len(
            matches
        ) > 1:

            raise RuntimeError(
                "Multiple TPMT assay files found:\n"
                + "\n".join(
                    str(
                        path
                    )
                    for path in matches
                )
                + "\nUse --input-csv to choose one."
            )


    raise FileNotFoundError(
        "Could not locate "
        f"{ASSAY_FILENAME} under data/raw.\n"
        "Pass it explicitly with:\n"
        "  --input-csv /path/to/"
        f"{ASSAY_FILENAME}"
    )


# ============================================================
# Mutation parsing / validation
# ============================================================

def parse_single_mutation(
    mutant,
):
    mutant = str(
        mutant
    ).strip()

    match = MUTATION_RE.fullmatch(
        mutant
    )

    if match is None:

        raise ValueError(
            "TPMT external-validation dataset "
            "must contain single substitutions only, "
            f"but found: {mutant!r}"
        )


    wt_aa, position, mut_aa = (
        match.groups()
    )

    position = int(
        position
    )


    return (
        wt_aa,
        position,
        mut_aa,
    )


def apply_mutation(
    wt_sequence,
    position,
    mut_aa,
):
    chars = list(
        wt_sequence
    )

    chars[
        position
        - 1
    ] = mut_aa

    return "".join(
        chars
    )


# ============================================================
# CV splits
# ============================================================

def assign_random_folds(
    n_samples,
    seed=42,
):
    folds = np.full(
        n_samples,
        -1,
        dtype=np.int64,
    )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )


    dummy_x = np.zeros(
        (
            n_samples,
            1,
        )
    )


    for fold_idx, (
        _,
        test_idx,
    ) in enumerate(
        kfold.split(
            dummy_x
        )
    ):

        folds[
            test_idx
        ] = fold_idx


    if (
        folds
        < 0
    ).any():

        raise RuntimeError(
            "Random fold assignment failed."
        )


    return folds


def assign_position_grouped_folds(
    positions,
):
    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    folds = np.full(
        len(
            positions
        ),
        -1,
        dtype=np.int64,
    )


    splitter = GroupKFold(
        n_splits=5
    )


    dummy_x = np.zeros(
        (
            len(
                positions
            ),
            1,
        )
    )


    for fold_idx, (
        _,
        test_idx,
    ) in enumerate(
        splitter.split(
            dummy_x,
            groups=positions,
        )
    ):

        folds[
            test_idx
        ] = fold_idx


    if (
        folds
        < 0
    ).any():

        raise RuntimeError(
            "Position-grouped fold assignment failed."
        )


    # Leakage check:
    # each mutation position must belong to exactly one fold.
    check = pd.DataFrame(
        {
            "position":
                positions,

            "fold":
                folds,
        }
    )


    fold_counts_per_position = (
        check.groupby(
            "position"
        )[
            "fold"
        ]
        .nunique()
    )


    if (
        fold_counts_per_position
        != 1
    ).any():

        raise RuntimeError(
            "Position leakage detected "
            "in fold_position_5."
        )


    return folds


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare ProteinGym TPMT "
            "single-mutant assay."
        )
    )


    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit path to "
            "TPMT_HUMAN_Matreyek_2018.csv"
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # Sanity-check embedded WT
    # ========================================================

    if len(
        WT_SEQUENCE
    ) != EXPECTED_LENGTH:

        raise RuntimeError(
            "Embedded TPMT WT sequence has "
            f"length {len(WT_SEQUENCE)}, "
            f"expected {EXPECTED_LENGTH}."
        )


    input_path = find_assay_csv(
        args.input_csv
    )


    print(
        "=" * 110
    )

    print(
        "Step 11-1: TPMT ProteinGym Preprocessing"
    )

    print(
        "=" * 110
    )

    print(
        f"Assay       : {ASSAY_ID}"
    )

    print(
        f"Input       : {input_path}"
    )

    print(
        f"WT length   : {len(WT_SEQUENCE)}"
    )


    # ========================================================
    # Load ProteinGym processed assay
    # ========================================================

    df = pd.read_csv(
        input_path
    )


    required_columns = {
        "mutant",
        "mutated_sequence",
        "DMS_score",
    }


    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "Input assay is missing columns: "
            f"{sorted(missing_columns)}"
        )


    keep_columns = [
        "mutant",
        "mutated_sequence",
        "DMS_score",
    ]


    if (
        "DMS_score_bin"
        in df.columns
    ):

        keep_columns.append(
            "DMS_score_bin"
        )


    df = df[
        keep_columns
    ].copy()


    if df[
        "mutant"
    ].duplicated().any():

        duplicate_count = int(
            df[
                "mutant"
            ]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Duplicate mutant IDs found: "
            f"{duplicate_count}"
        )


    if df[
        "DMS_score"
    ].isna().any():

        raise ValueError(
            "Missing DMS_score values found."
        )


    # ========================================================
    # Parse mutations
    # ========================================================

    wt_aas = []

    positions = []

    mutant_aas = []

    python_indices = []


    for row in df.itertuples(
        index=False
    ):

        (
            wt_aa,
            position,
            mut_aa,
        ) = parse_single_mutation(
            row.mutant
        )


        if not (
            1
            <= position
            <= EXPECTED_LENGTH
        ):

            raise ValueError(
                f"Position out of range: "
                f"{row.mutant}"
            )


        expected_wt_aa = (
            WT_SEQUENCE[
                position
                - 1
            ]
        )


        if wt_aa != expected_wt_aa:

            raise ValueError(
                "WT amino-acid mismatch: "
                f"{row.mutant}; "
                f"expected "
                f"{expected_wt_aa}{position}"
            )


        sequence = str(
            row.mutated_sequence
        )


        if len(
            sequence
        ) != EXPECTED_LENGTH:

            raise ValueError(
                "Mutated sequence length mismatch: "
                f"{row.mutant}; "
                f"{len(sequence)} "
                f"!= {EXPECTED_LENGTH}"
            )


        expected_mutated_sequence = (
            apply_mutation(
                WT_SEQUENCE,
                position,
                mut_aa,
            )
        )


        if (
            sequence
            != expected_mutated_sequence
        ):

            mismatch_count = sum(
                a != b
                for a, b in zip(
                    sequence,
                    expected_mutated_sequence,
                )
            )

            raise ValueError(
                "mutated_sequence does not match "
                f"the parsed mutation {row.mutant}; "
                f"sequence mismatches={mismatch_count}"
            )


        wt_aas.append(
            wt_aa
        )

        positions.append(
            position
        )

        mutant_aas.append(
            mut_aa
        )

        python_indices.append(
            position
            - 1
        )


    df[
        "wt_aa"
    ] = wt_aas


    df[
        "position"
    ] = positions


    df[
        "python_index"
    ] = python_indices


    df[
        "mut_aa"
    ] = mutant_aas


    df[
        "mutation_count"
    ] = 1


    # ========================================================
    # Validation splits
    # ========================================================

    df[
        "fold_random_5"
    ] = assign_random_folds(
        len(
            df
        ),
        seed=42,
    )


    df[
        "fold_position_5"
    ] = (
        assign_position_grouped_folds(
            df[
                "position"
            ].to_numpy()
        )
    )


    # ========================================================
    # Dataset-level validation
    # ========================================================

    n_variants = len(
        df
    )


    n_positions = int(
        df[
            "position"
        ].nunique()
    )


    if (
        n_variants
        != EXPECTED_N_SINGLE
    ):

        print(
            "WARNING:"
        )

        print(
            f"Official metadata reports "
            f"{EXPECTED_N_SINGLE:,} single mutants, "
            f"but this file contains "
            f"{n_variants:,}."
        )

        print(
            "This may reflect a different "
            "ProteinGym release/version."
        )


    # ========================================================
    # Position summary
    # ========================================================

    position_summary = (
        df.groupby(
            "position"
        )
        .agg(
            wt_aa=(
                "wt_aa",
                "first",
            ),

            n_substitutions=(
                "mutant",
                "size",
            ),

            dms_score_mean=(
                "DMS_score",
                "mean",
            ),

            dms_score_std=(
                "DMS_score",
                "std",
            ),

            dms_score_min=(
                "DMS_score",
                "min",
            ),

            dms_score_max=(
                "DMS_score",
                "max",
            ),

            position_fold=(
                "fold_position_5",
                "first",
            ),
        )
        .reset_index()
    )


    # ========================================================
    # Save
    # ========================================================

    processed_path = (
        OUTPUT_ROOT
        / "tpmt_single_mutants.csv"
    )


    fasta_path = (
        OUTPUT_ROOT
        / "tpmt_wt.fasta"
    )


    position_summary_path = (
        OUTPUT_ROOT
        / "tpmt_position_summary.csv"
    )


    summary_path = (
        OUTPUT_ROOT
        / "tpmt_summary.json"
    )


    df.to_csv(
        processed_path,
        index=False,
    )


    position_summary.to_csv(
        position_summary_path,
        index=False,
    )


    fasta_path.write_text(
        (
            ">TPMT_HUMAN_Matreyek_2018\n"
            + WT_SEQUENCE
            + "\n"
        ),
        encoding="utf-8",
    )


    score = df[
        "DMS_score"
    ]


    random_fold_sizes = (
        df[
            "fold_random_5"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    position_fold_sizes = (
        df[
            "fold_position_5"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    positions_per_fold = (
        df[
            [
                "position",
                "fold_position_5",
            ]
        ]
        .drop_duplicates()
        [
            "fold_position_5"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    summary = {
        "stage":
            "Step 11-1",

        "assay":
            ASSAY_ID,

        "source_file":
            str(
                input_path
            ),

        "sequence_length":
            EXPECTED_LENGTH,

        "n_variants":
            int(
                n_variants
            ),

        "n_unique_mutation_positions":
            int(
                n_positions
            ),

        "all_single_mutants":
            True,

        "dms_score": {
            "min":
                float(
                    score.min()
                ),

            "q25":
                float(
                    score.quantile(
                        0.25
                    )
                ),

            "median":
                float(
                    score.median()
                ),

            "mean":
                float(
                    score.mean()
                ),

            "q75":
                float(
                    score.quantile(
                        0.75
                    )
                ),

            "max":
                float(
                    score.max()
                ),

            "std":
                float(
                    score.std()
                ),
        },

        "splits": {
            "fold_random_5": {
                "description":
                    (
                        "Shuffled row-level "
                        "5-fold CV, seed=42."
                    ),

                "fold_sizes":
                    {
                        str(
                            key
                        ):
                        int(
                            value
                        )

                        for key, value
                        in random_fold_sizes.items()
                    },
            },

            "fold_position_5": {
                "description":
                    (
                        "5-fold GroupKFold by "
                        "mutation position. "
                        "All substitutions from "
                        "a residue remain in the "
                        "same fold."
                    ),

                "fold_sizes":
                    {
                        str(
                            key
                        ):
                        int(
                            value
                        )

                        for key, value
                        in position_fold_sizes.items()
                    },

                "unique_positions_per_fold":
                    {
                        str(
                            key
                        ):
                        int(
                            value
                        )

                        for key, value
                        in positions_per_fold.items()
                    },
            },
        },

        "artifacts": {
            "processed_csv":
                str(
                    processed_path
                ),

            "wt_fasta":
                str(
                    fasta_path
                ),

            "position_summary":
                str(
                    position_summary_path
                ),
        },
    }


    with open(
        summary_path,
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
        "=" * 110
    )

    print(
        "TPMT Preprocessing Summary"
    )

    print(
        "=" * 110
    )


    print(
        f"Variants              : "
        f"{n_variants:,}"
    )

    print(
        f"Unique positions      : "
        f"{n_positions}"
    )

    print(
        f"Sequence length       : "
        f"{EXPECTED_LENGTH}"
    )


    print()
    print(
        "DMS_score"
    )

    print(
        "min/q25/median/mean/q75/max = "
        f"{score.min():.6f} / "
        f"{score.quantile(0.25):.6f} / "
        f"{score.median():.6f} / "
        f"{score.mean():.6f} / "
        f"{score.quantile(0.75):.6f} / "
        f"{score.max():.6f}"
    )


    print()
    print(
        "Random 5-fold sizes"
    )

    print(
        random_fold_sizes
    )


    print()
    print(
        "Position-grouped 5-fold sizes"
    )

    print(
        position_fold_sizes
    )


    print(
        "Unique positions / fold"
    )

    print(
        positions_per_fold
    )


    print()
    print(
        "Output columns"
    )

    print(
        df.columns.tolist()
    )


    print()
    print(
        "Saved:"
    )

    print(
        processed_path
    )

    print(
        fasta_path
    )

    print(
        position_summary_path
    )

    print(
        summary_path
    )


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "fold_random_5 is ordinary "
        "row-level CV."
    )

    print(
        "fold_position_5 prevents "
        "substitutions at the same residue "
        "from appearing in both train and test."
    )

    print(
        "No representation or model fitting "
        "is performed in Step 11-1."
    )


if __name__ == "__main__":
    main()
