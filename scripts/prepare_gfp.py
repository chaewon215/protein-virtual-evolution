from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DMS_ID = "GFP_AEQVI_Sarkisyan_2016"

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference_files"
    / "DMS_substitutions.csv"
)

DMS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "DMS_ProteinGym_substitutions"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


ALL_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "gfp_all_variants.csv"
)

SINGLE_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "gfp_single_mutants.csv"
)

MULTI_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "gfp_multi_mutants.csv"
)

WT_FASTA_PATH = (
    OUTPUT_ROOT
    / "gfp_wt.fasta"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "gfp_pool_summary.json"
)

COUNT_DISTRIBUTION_PATH = (
    OUTPUT_ROOT
    / "gfp_mutation_count_distribution.csv"
)


# ============================================================
# Mutation parser
#
# Examples:
#   H25R
#   F8L:F46L
#
# Each component:
#   WT amino acid
#   1-based residue position
#   mutant amino acid
# ============================================================

MUTATION_PATTERN = re.compile(
    r"^([A-Z])(\d+)([A-Z])$"
)


def parse_mutation_component(
    component: str,
):

    match = MUTATION_PATTERN.fullmatch(
        component
    )

    if match is None:

        raise ValueError(
            f"Invalid substitution notation: {component}"
        )

    wt_aa, position, mut_aa = (
        match.groups()
    )

    return (
        wt_aa,
        int(position),
        mut_aa,
    )


def parse_variant(
    mutant: str,
):

    components = mutant.split(
        ":"
    )

    parsed = [
        parse_mutation_component(
            component
        )
        for component in components
    ]

    return parsed


# ============================================================
# Load reference metadata
# ============================================================

if not REFERENCE_PATH.exists():

    raise FileNotFoundError(
        f"Reference file not found: {REFERENCE_PATH}"
    )


reference_df = pd.read_csv(
    REFERENCE_PATH,
    low_memory=False,
)


if "DMS_id" not in reference_df.columns:

    raise ValueError(
        "Reference file does not contain DMS_id."
    )


rows = reference_df[
    reference_df[
        "DMS_id"
    ]
    == DMS_ID
]


if len(rows) != 1:

    raise ValueError(
        f"Expected exactly one metadata row for "
        f"{DMS_ID}, found {len(rows)}."
    )


metadata = rows.iloc[0]


if "target_seq" not in metadata.index:

    raise ValueError(
        "Reference metadata does not contain target_seq."
    )


wt_sequence = str(
    metadata[
        "target_seq"
    ]
).strip().upper()


if not wt_sequence:

    raise ValueError(
        "Empty WT sequence."
    )


# Prefer official metadata filename when available.
if (
    "DMS_filename"
    in metadata.index
    and pd.notna(
        metadata[
            "DMS_filename"
        ]
    )
):

    dms_filename = str(
        metadata[
            "DMS_filename"
        ]
    )

else:

    dms_filename = (
        f"{DMS_ID}.csv"
    )


dms_path = (
    DMS_ROOT
    / dms_filename
)


# Fallback for datasets whose metadata filename differs.
if not dms_path.exists():

    fallback = (
        DMS_ROOT
        / f"{DMS_ID}.csv"
    )

    if fallback.exists():

        dms_path = fallback

    else:

        matches = list(
            DMS_ROOT.glob(
                f"*{DMS_ID}*.csv"
            )
        )

        if len(matches) == 1:

            dms_path = matches[0]

        else:

            raise FileNotFoundError(
                f"Could not locate GFP assay CSV in "
                f"{DMS_ROOT}. "
                f"Expected {dms_filename}."
            )


print(
    "DMS ID       :",
    DMS_ID,
)

print(
    "DMS file     :",
    dms_path,
)

print(
    "WT length    :",
    len(
        wt_sequence
    ),
)


# ============================================================
# Load GFP assay
# ============================================================

df = pd.read_csv(
    dms_path,
    low_memory=False,
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
        f"Missing required columns: {missing_columns}"
    )


print(
    "Raw variants :",
    len(
        df
    ),
)


# ============================================================
# Basic validation
# ============================================================

if df[
    "mutant"
].isna().any():

    raise ValueError(
        "mutant contains missing values."
    )


if df[
    "mutated_sequence"
].isna().any():

    raise ValueError(
        "mutated_sequence contains missing values."
    )


if df[
    "DMS_score"
].isna().any():

    raise ValueError(
        "DMS_score contains missing values."
    )


df[
    "mutant"
] = (
    df[
        "mutant"
    ]
    .astype(
        str
    )
    .str.strip()
)


df[
    "mutated_sequence"
] = (
    df[
        "mutated_sequence"
    ]
    .astype(
        str
    )
    .str.strip()
    .str.upper()
)


df[
    "DMS_score"
] = pd.to_numeric(
    df[
        "DMS_score"
    ],
    errors="raise",
)


# ============================================================
# Parse mutation sets
#
# For each variant:
#
# mutant
#   "F8L:F46L"
#
# becomes
#
# mutation_count = 2
# positions      = [8, 46]
# wt_aas         = ["F", "F"]
# mut_aas        = ["L", "L"]
# ============================================================

parsed_variants = []


for row_index, mutant in enumerate(
    df[
        "mutant"
    ]
):

    try:

        parsed = parse_variant(
            mutant
        )

    except Exception as exc:

        raise ValueError(
            f"Failed to parse row {row_index}: "
            f"{mutant}"
        ) from exc


    parsed_variants.append(
        parsed
    )


df[
    "mutation_count"
] = [
    len(
        parsed
    )
    for parsed in parsed_variants
]


df[
    "mutation_positions"
] = [
    ",".join(
        str(
            position
        )
        for _, position, _
        in parsed
    )
    for parsed in parsed_variants
]


df[
    "wt_aas"
] = [
    ",".join(
        wt_aa
        for wt_aa, _, _
        in parsed
    )
    for parsed in parsed_variants
]


df[
    "mut_aas"
] = [
    ",".join(
        mut_aa
        for _, _, mut_aa
        in parsed
    )
    for parsed in parsed_variants
]


# ============================================================
# Sequence-level validation
#
# For each variant:
#
# 1. sequence length must equal WT length
# 2. mutation labels must agree with WT sequence
# 3. mutation labels must agree with mutated sequence
# 4. actual Hamming distance must equal mutation_count
#
# This guarantees that a "3-mutation" candidate really differs
# from WT at exactly three labeled positions.
# ============================================================

for i, parsed in enumerate(
    parsed_variants
):

    mutant_label = df.iloc[
        i
    ][
        "mutant"
    ]


    mutated_sequence = df.iloc[
        i
    ][
        "mutated_sequence"
    ]


    if (
        len(
            mutated_sequence
        )
        != len(
            wt_sequence
        )
    ):

        raise ValueError(
            f"Sequence length mismatch at row {i}: "
            f"{mutant_label} | "
            f"{len(mutated_sequence)} != {len(wt_sequence)}"
        )


    seen_positions = set()


    for (
        wt_aa,
        position,
        mut_aa,
    ) in parsed:

        if not (
            1
            <= position
            <= len(
                wt_sequence
            )
        ):

            raise ValueError(
                f"Out-of-range position at row {i}: "
                f"{mutant_label}"
            )


        if position in seen_positions:

            raise ValueError(
                f"Duplicate position in mutation label "
                f"at row {i}: {mutant_label}"
            )


        seen_positions.add(
            position
        )


        wt_from_sequence = (
            wt_sequence[
                position - 1
            ]
        )


        mutant_from_sequence = (
            mutated_sequence[
                position - 1
            ]
        )


        if (
            wt_from_sequence
            != wt_aa
        ):

            raise ValueError(
                f"WT mismatch at row {i}: "
                f"{mutant_label} | "
                f"position {position} | "
                f"label={wt_aa}, "
                f"WT sequence={wt_from_sequence}"
            )


        if (
            mutant_from_sequence
            != mut_aa
        ):

            raise ValueError(
                f"Mutant sequence mismatch at row {i}: "
                f"{mutant_label} | "
                f"position {position} | "
                f"label={mut_aa}, "
                f"sequence={mutant_from_sequence}"
            )


        if wt_aa == mut_aa:

            raise ValueError(
                f"Non-changing substitution at row {i}: "
                f"{mutant_label}"
            )


    actual_difference_count = sum(
        wt_residue
        != mutant_residue
        for (
            wt_residue,
            mutant_residue
        )
        in zip(
            wt_sequence,
            mutated_sequence,
        )
    )


    expected_difference_count = len(
        parsed
    )


    if (
        actual_difference_count
        != expected_difference_count
    ):

        raise ValueError(
            f"Hamming-distance mismatch at row {i}: "
            f"{mutant_label} | "
            f"expected={expected_difference_count}, "
            f"actual={actual_difference_count}"
        )


print(
    "Sequence validation: OK"
)


# ============================================================
# Pool construction
#
# Initial labeled training pool:
#   exactly one substitution
#
# Hidden candidate pool:
#   two or more substitutions
#
# IMPORTANT:
# The multi-mutant DMS_score is stored in the retrospective
# benchmark file, but must NOT be accessed by future candidate
# selection logic until that candidate is "revealed".
# ============================================================

single_df = (
    df[
        df[
            "mutation_count"
        ]
        == 1
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


multi_df = (
    df[
        df[
            "mutation_count"
        ]
        >= 2
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if len(
    single_df
) == 0:

    raise ValueError(
        "No single-mutant variants found."
    )


if len(
    multi_df
) == 0:

    raise ValueError(
        "No multi-mutant variants found."
    )


# ============================================================
# Mutation count distribution
# ============================================================

count_distribution = (
    df[
        "mutation_count"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "mutation_count"
    )
    .reset_index(
        name="n_variants"
    )
)


count_distribution[
    "fraction"
] = (
    count_distribution[
        "n_variants"
    ]
    / len(
        df
    )
)


# ============================================================
# Score statistics
# ============================================================

def score_summary(
    frame: pd.DataFrame,
):

    scores = frame[
        "DMS_score"
    ].to_numpy(
        dtype=np.float64
    )


    return {
        "n":
            int(
                len(
                    frame
                )
            ),

        "min":
            float(
                np.min(
                    scores
                )
            ),

        "q25":
            float(
                np.quantile(
                    scores,
                    0.25,
                )
            ),

        "median":
            float(
                np.median(
                    scores
                )
            ),

        "mean":
            float(
                np.mean(
                    scores
                )
            ),

        "q75":
            float(
                np.quantile(
                    scores,
                    0.75,
                )
            ),

        "max":
            float(
                np.max(
                    scores
                )
            ),
    }


# ============================================================
# Save outputs
# ============================================================

df.to_csv(
    ALL_OUTPUT_PATH,
    index=False,
)


single_df.to_csv(
    SINGLE_OUTPUT_PATH,
    index=False,
)


multi_df.to_csv(
    MULTI_OUTPUT_PATH,
    index=False,
)


count_distribution.to_csv(
    COUNT_DISTRIBUTION_PATH,
    index=False,
)


with open(
    WT_FASTA_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        f">{DMS_ID}_WT\n"
    )

    f.write(
        wt_sequence
        + "\n"
    )


summary = {
    "DMS_id":
        DMS_ID,

    "source_file":
        str(
            dms_path
        ),

    "wt_length":
        int(
            len(
                wt_sequence
            )
        ),

    "total_variants":
        int(
            len(
                df
            )
        ),

    "single_mutants":
        int(
            len(
                single_df
            )
        ),

    "multi_mutants":
        int(
            len(
                multi_df
            )
        ),

    "max_mutation_count":
        int(
            df[
                "mutation_count"
            ].max()
        ),

    "single_score_summary":
        score_summary(
            single_df
        ),

    "multi_score_summary":
        score_summary(
            multi_df
        ),

    "retrospective_protocol":
        {
            "initial_labeled_pool":
                "mutation_count == 1",

            "hidden_candidate_pool":
                "mutation_count >= 2",

            "candidate_label_rule":
                (
                    "DMS_score of a multi-mutant candidate "
                    "must remain hidden until the simulation "
                    "selects and reveals that candidate."
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


# ============================================================
# Console report
# ============================================================

print()
print(
    "=" * 80
)

print(
    "GFP Retrospective Pool Summary"
)

print(
    "=" * 80
)


print(
    f"WT length           : "
    f"{len(wt_sequence)}"
)

print(
    f"Total variants      : "
    f"{len(df)}"
)

print(
    f"Single mutants      : "
    f"{len(single_df)}"
)

print(
    f"Multi-mutant pool   : "
    f"{len(multi_df)}"
)

print(
    f"Max mutation count  : "
    f"{df['mutation_count'].max()}"
)


print()
print(
    "=== Mutation Count Distribution ==="
)

print(
    count_distribution.to_string(
        index=False
    )
)


print()
print(
    "=== Single-mutant DMS Score ==="
)

for key, value in score_summary(
    single_df
).items():

    print(
        f"{key:>8s}: {value}"
    )


print()
print(
    "=== Multi-mutant DMS Score ==="
)

for key, value in score_summary(
    multi_df
).items():

    print(
        f"{key:>8s}: {value}"
    )


print()
print(
    "Saved:"
)

print(
    ALL_OUTPUT_PATH
)

print(
    SINGLE_OUTPUT_PATH
)

print(
    MULTI_OUTPUT_PATH
)

print(
    WT_FASTA_PATH
)

print(
    COUNT_DISTRIBUTION_PATH
)

print(
    SUMMARY_PATH
)


print()
print(
    "IMPORTANT:"
)

print(
    "Future candidate-selection code must not use "
    "gfp_multi_mutants.csv::DMS_score before a candidate "
    "is explicitly revealed by the retrospective simulator."
)
