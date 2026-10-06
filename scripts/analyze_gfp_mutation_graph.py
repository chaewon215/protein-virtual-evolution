from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Step 8-5.5
# GFP Mutation-Graph Accessibility Audit
#
# LABEL-FREE:
# - DMS_score is never loaded.
# - Uses mutation strings / mutation_count only.
# - Optionally joins final ensemble mu/sigma predictions.
#
# Directed edge definition:
#
# parent -> child
#
# iff:
#   1. child has exactly one more WT-relative substitution
#   2. every parent substitution is preserved in child
#
# Example:
#   F46L
#      -> F46L:S65T
#
#   F46L:S65T
#      -> F46L:S65T:T203Y
#
# This creates a forward-addition DAG:
#   K mutations -> K+1 mutations
#
# IMPORTANT:
# "unreachable" here means:
#   no complete one-addition path exists within the OBSERVED
#   ProteinGym GFP benchmark variants, starting from the
#   observed single-mutant set.
#
# It does NOT mean the protein is biologically impossible
# to generate in a real laboratory.
# ============================================================


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esmc300m_gfp"
    / "gfp_variant_feature_metadata.csv"
)

FINAL_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "final_ensemble"
    / "multi_mutant_predictions.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "mutation_graph_audit"
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


VARIANT_ACCESSIBILITY_PATH = (
    OUTPUT_ROOT
    / "variant_accessibility.csv"
)

LEVEL_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "mutation_count_summary.csv"
)

TOP_PREDICTED_PATH = (
    OUTPUT_ROOT
    / "top_predicted_accessibility.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


# ============================================================
# Settings
# ============================================================

TOP_N_VALUES = [
    10,
    50,
    100,
    500,
    1000,
]

MUTATION_RE = re.compile(
    r"^([A-Z])(\d+)([A-Z])$"
)


# ============================================================
# Mutation parsing
#
# Canonical representation:
#
# tuple(
#   (position, WT_aa, mutant_aa),
#   ...
# )
#
# sorted by biological position
#
# Example:
#   "F46L:S65T"
#
# ->
#   (
#     (46, "F", "L"),
#     (65, "S", "T"),
#   )
#
# This tuple is hashable and can be used as a dictionary key.
# ============================================================

def parse_mutant_string(
    mutant: str,
):
    mutant = str(
        mutant
    ).strip()

    if not mutant:
        raise ValueError(
            "Empty mutant string."
        )

    substitutions = []

    seen_positions = set()

    for token in mutant.split(
        ":"
    ):
        match = MUTATION_RE.fullmatch(
            token
        )

        if match is None:
            raise ValueError(
                f"Invalid mutation token: "
                f"{token!r} "
                f"in {mutant!r}"
            )

        wt_aa, pos, mut_aa = (
            match.groups()
        )

        pos = int(
            pos
        )

        if pos in seen_positions:
            raise ValueError(
                f"Duplicate mutated position "
                f"{pos} in {mutant!r}"
            )

        seen_positions.add(
            pos
        )

        substitutions.append(
            (
                pos,
                wt_aa,
                mut_aa,
            )
        )

    substitutions.sort(
        key=lambda item: item[0]
    )

    return tuple(
        substitutions
    )


def mutation_key_to_string(
    key,
):
    return ":".join(
        f"{wt}{pos}{mut}"
        for (
            pos,
            wt,
            mut,
        ) in key
    )


# ============================================================
# Load label-free metadata
# ============================================================

print(
    "=" * 110
)

print(
    "Step 8-5.5: GFP Mutation-Graph Accessibility Audit"
)

print(
    "=" * 110
)


metadata = pd.read_csv(
    FEATURE_METADATA_PATH
)


required_columns = {
    "mutant",
    "mutation_count",
    "feature_index",
}


missing_columns = (
    required_columns
    - set(
        metadata.columns
    )
)


if missing_columns:
    raise ValueError(
        "Feature metadata is missing "
        f"required columns: "
        f"{sorted(missing_columns)}"
    )


metadata = metadata[
    [
        "mutant",
        "mutation_count",
        "feature_index",
    ]
].copy()


metadata[
    "mutant"
] = metadata[
    "mutant"
].astype(
    str
)


metadata[
    "mutation_count"
] = metadata[
    "mutation_count"
].astype(
    int
)


if metadata[
    "mutant"
].duplicated().any():
    raise ValueError(
        "Duplicate mutant strings found."
    )


# ============================================================
# Parse and validate mutation counts
# ============================================================

parsed_keys = []

for row in metadata.itertuples(
    index=False
):
    key = parse_mutant_string(
        row.mutant
    )

    if len(
        key
    ) != row.mutation_count:
        raise ValueError(
            f"Mutation-count mismatch: "
            f"{row.mutant} | "
            f"metadata={row.mutation_count}, "
            f"parsed={len(key)}"
        )

    parsed_keys.append(
        key
    )


metadata[
    "_mutation_key"
] = parsed_keys


# ============================================================
# Build lookup:
#
# mutation tuple -> row index
#
# N ~ 51,714 total observed variants
# ============================================================

key_to_row = {
    key: row_idx
    for row_idx, key in enumerate(
        metadata[
            "_mutation_key"
        ]
    )
}


if len(
    key_to_row
) != len(
    metadata
):
    raise ValueError(
        "Canonical mutation-key collision found."
    )


# ============================================================
# Separate singles and multis
# ============================================================

single_mask = (
    metadata[
        "mutation_count"
    ]
    == 1
)


multi_mask = (
    metadata[
        "mutation_count"
    ]
    >= 2
)


n_single = int(
    single_mask.sum()
)


n_multi = int(
    multi_mask.sum()
)


max_k = int(
    metadata[
        "mutation_count"
    ].max()
)


print()
print(
    "Observed benchmark variants"
)

print(
    f"Single mutants : {n_single}"
)

print(
    f"Multi mutants  : {n_multi}"
)

print(
    f"Total          : {len(metadata)}"
)

print(
    f"Max K          : {max_k}"
)


# ============================================================
# Parent enumeration
#
# For a K-mutation variant:
#
# key = (m1, m2, ..., mK)
#
# possible strict additive parents are obtained by removing
# exactly one mutation:
#
# K possible parent keys
#
# Example:
#
# F46L:S65T:T203Y
#
# candidate parents:
#   S65T:T203Y
#   F46L:T203Y
#   F46L:S65T
#
# Only parents actually present in the benchmark count as
# observed parents.
# ============================================================

observed_parent_rows = []

observed_parent_counts = np.zeros(
    len(
        metadata
    ),
    dtype=np.int32,
)


for row_idx, key in enumerate(
    metadata[
        "_mutation_key"
    ]
):
    k = len(
        key
    )

    if k == 1:
        observed_parent_rows.append(
            []
        )
        continue

    parents = []

    for remove_idx in range(
        k
    ):
        parent_key = (
            key[
                :remove_idx
            ]
            + key[
                remove_idx + 1:
            ]
        )

        parent_row = key_to_row.get(
            parent_key
        )

        if parent_row is not None:
            parents.append(
                parent_row
            )

    observed_parent_rows.append(
        parents
    )

    observed_parent_counts[
        row_idx
    ] = len(
        parents
    )


# ============================================================
# Reachability dynamic programming
#
# All observed singles are initial reachable nodes.
#
# A K-mutant is reachable iff at least one observed parent
# with K-1 mutations is itself reachable.
#
# Because edges always increase K by 1, this is a DAG and
# can be solved level-by-level.
# ============================================================

reachable = np.zeros(
    len(
        metadata
    ),
    dtype=bool,
)


shortest_depth_from_single = np.full(
    len(
        metadata
    ),
    -1,
    dtype=np.int32,
)


# Starting labeled set:
# all observed single mutants.
single_indices = np.flatnonzero(
    single_mask.to_numpy()
)


reachable[
    single_indices
] = True


shortest_depth_from_single[
    single_indices
] = 0


for k in range(
    2,
    max_k + 1,
):
    level_indices = np.flatnonzero(
        (
            metadata[
                "mutation_count"
            ]
            .to_numpy()
            == k
        )
    )

    for row_idx in level_indices:
        parents = observed_parent_rows[
            row_idx
        ]

        reachable_parents = [
            parent_row
            for parent_row in parents
            if reachable[
                parent_row
            ]
        ]

        if reachable_parents:
            reachable[
                row_idx
            ] = True

            shortest_depth_from_single[
                row_idx
            ] = (
                min(
                    shortest_depth_from_single[
                        parent_row
                    ]
                    for parent_row
                    in reachable_parents
                )
                + 1
            )


# ============================================================
# Reachable-parent counts
# ============================================================

reachable_parent_counts = np.zeros(
    len(
        metadata
    ),
    dtype=np.int32,
)


for row_idx, parents in enumerate(
    observed_parent_rows
):
    if not parents:
        continue

    reachable_parent_counts[
        row_idx
    ] = int(
        sum(
            reachable[
                parent_row
            ]
            for parent_row in parents
        )
    )


# ============================================================
# Child counts
#
# useful to characterize graph connectivity
# ============================================================

observed_child_counts = np.zeros(
    len(
        metadata
    ),
    dtype=np.int32,
)


reachable_child_counts = np.zeros(
    len(
        metadata
    ),
    dtype=np.int32,
)


for child_idx, parents in enumerate(
    observed_parent_rows
):
    for parent_idx in parents:
        observed_child_counts[
            parent_idx
        ] += 1

        if reachable[
            child_idx
        ]:
            reachable_child_counts[
                parent_idx
            ] += 1


# ============================================================
# Minimum missing intermediates diagnostic
#
# strict reachability tells whether a COMPLETE observed path
# exists.
#
# "has_observed_parent" tells whether at least the immediately
# preceding K-1 level contains a matching variant.
# ============================================================

has_observed_parent = (
    observed_parent_counts
    > 0
)


has_reachable_parent = (
    reachable_parent_counts
    > 0
)


# ============================================================
# Variant-level table
# ============================================================

variant_df = metadata[
    [
        "mutant",
        "mutation_count",
        "feature_index",
    ]
].copy()


variant_df[
    "observed_parent_count"
] = observed_parent_counts


variant_df[
    "reachable_parent_count"
] = reachable_parent_counts


variant_df[
    "has_observed_parent"
] = has_observed_parent


variant_df[
    "has_reachable_parent"
] = has_reachable_parent


variant_df[
    "reachable_from_single"
] = reachable


variant_df[
    "steps_from_single"
] = shortest_depth_from_single


variant_df[
    "observed_child_count"
] = observed_child_counts


variant_df[
    "reachable_child_count"
] = reachable_child_counts


variant_df.to_csv(
    VARIANT_ACCESSIBILITY_PATH,
    index=False,
)


# ============================================================
# Per-mutation-count summary
# ============================================================

level_rows = []


for k in range(
    1,
    max_k + 1,
):
    mask = (
        metadata[
            "mutation_count"
        ]
        .to_numpy()
        == k
    )


    indices = np.flatnonzero(
        mask
    )


    if len(
        indices
    ) == 0:
        continue


    total = len(
        indices
    )


    n_with_parent = int(
        has_observed_parent[
            indices
        ].sum()
    )


    n_reachable = int(
        reachable[
            indices
        ].sum()
    )


    if k == 1:
        parent_fraction = np.nan
    else:
        parent_fraction = (
            n_with_parent
            / total
        )


    level_rows.append(
        {
            "mutation_count":
                int(
                    k
                ),

            "n_variants":
                int(
                    total
                ),

            "n_with_observed_parent":
                int(
                    n_with_parent
                ),

            "fraction_with_observed_parent":
                (
                    float(
                        parent_fraction
                    )
                    if np.isfinite(
                        parent_fraction
                    )
                    else np.nan
                ),

            "n_reachable_from_single":
                int(
                    n_reachable
                ),

            "fraction_reachable_from_single":
                float(
                    n_reachable
                    / total
                ),

            "mean_observed_parent_count":
                float(
                    observed_parent_counts[
                        indices
                    ].mean()
                ),

            "max_observed_parent_count":
                int(
                    observed_parent_counts[
                        indices
                    ].max()
                ),

            "mean_reachable_parent_count":
                float(
                    reachable_parent_counts[
                        indices
                    ].mean()
                ),

            "mean_observed_child_count":
                float(
                    observed_child_counts[
                        indices
                    ].mean()
                ),

            "max_observed_child_count":
                int(
                    observed_child_counts[
                        indices
                    ].max()
                ),
        }
    )


level_df = pd.DataFrame(
    level_rows
)


level_df.to_csv(
    LEVEL_SUMMARY_PATH,
    index=False,
)


# ============================================================
# Overall multi-mutant reachability
# ============================================================

multi_indices = np.flatnonzero(
    multi_mask.to_numpy()
)


n_multi_reachable = int(
    reachable[
        multi_indices
    ].sum()
)


multi_reachable_fraction = float(
    n_multi_reachable
    / len(
        multi_indices
    )
)


# Directly accessible doubles from observed singles.
double_indices = np.flatnonzero(
    (
        metadata[
            "mutation_count"
        ]
        .to_numpy()
        == 2
    )
)


n_double = int(
    len(
        double_indices
    )
)


n_double_reachable = int(
    reachable[
        double_indices
    ].sum()
)


double_reachable_fraction = (
    float(
        n_double_reachable
        / n_double
    )
    if n_double > 0
    else np.nan
)


# ============================================================
# Join final label-free predictions, if available
# ============================================================

top_predicted_records = []


if FINAL_PREDICTIONS_PATH.exists():

    predictions = pd.read_csv(
        FINAL_PREDICTIONS_PATH,
        usecols=[
            "mutant",
            "mutation_count",
            "feature_index",
            "predicted_fitness_mu",
            "uncertainty_sigma",
        ],
    )


    if (
        predictions[
            "mutant"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate candidates in "
            "final prediction file."
        )


    prediction_accessibility = (
        predictions.merge(
            variant_df[
                [
                    "mutant",
                    "reachable_from_single",
                    "steps_from_single",
                    "observed_parent_count",
                    "reachable_parent_count",
                    "has_observed_parent",
                    "has_reachable_parent",
                ]
            ],

            on="mutant",

            how="left",

            validate="one_to_one",
        )
    )


    if (
        prediction_accessibility[
            "reachable_from_single"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Prediction/accessibility join failed."
        )


    prediction_accessibility = (
        prediction_accessibility.sort_values(
            "predicted_fitness_mu",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    prediction_accessibility[
        "mu_rank"
    ] = (
        np.arange(
            len(
                prediction_accessibility
            )
        )
        + 1
    )


    prediction_accessibility.to_csv(
        TOP_PREDICTED_PATH,
        index=False,
    )


    for top_n in TOP_N_VALUES:

        n = min(
            top_n,
            len(
                prediction_accessibility
            ),
        )


        subset = (
            prediction_accessibility.iloc[
                :n
            ]
        )


        top_predicted_records.append(
            {
                "top_n":
                    int(
                        n
                    ),

                "n_reachable":
                    int(
                        subset[
                            "reachable_from_single"
                        ].sum()
                    ),

                "fraction_reachable":
                    float(
                        subset[
                            "reachable_from_single"
                        ].mean()
                    ),

                "mean_mutation_count":
                    float(
                        subset[
                            "mutation_count"
                        ].mean()
                    ),

                "mean_mu":
                    float(
                        subset[
                            "predicted_fitness_mu"
                        ].mean()
                    ),

                "mean_sigma":
                    float(
                        subset[
                            "uncertainty_sigma"
                        ].mean()
                    ),
            }
        )


else:

    prediction_accessibility = None


# ============================================================
# Figures
# ============================================================

# ------------------------------------------------------------
# Figure 1:
# fraction reachable vs mutation count
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


ax.plot(
    level_df[
        "mutation_count"
    ],
    level_df[
        "fraction_reachable_from_single"
    ],
    marker="o",
    linewidth=2,
)


ax.set_xlabel(
    "Mutation count (K)"
)

ax.set_ylabel(
    "Fraction reachable from observed singles"
)

ax.set_ylim(
    -0.02,
    1.02,
)

ax.set_title(
    (
        "GFP Benchmark Stepwise Accessibility "
        "by Mutation Count"
    )
)


fig.tight_layout()


fig.savefig(
    FIGURE_ROOT
    / "reachable_fraction_by_mutation_count.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------
# Figure 2:
# observed count vs reachable count
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        9,
        5,
    )
)


x = np.arange(
    len(
        level_df
    )
)


width = 0.38


ax.bar(
    x - width / 2,
    level_df[
        "n_variants"
    ],
    width=width,
    label="Observed variants",
)


ax.bar(
    x + width / 2,
    level_df[
        "n_reachable_from_single"
    ],
    width=width,
    label="Reachable variants",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    level_df[
        "mutation_count"
    ]
)


ax.set_xlabel(
    "Mutation count (K)"
)

ax.set_ylabel(
    "Number of variants"
)

ax.set_title(
    (
        "Observed vs Stepwise-Reachable "
        "GFP Variants"
    )
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_ROOT
    / "observed_vs_reachable_counts.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# Summary JSON
# ============================================================

summary = {
    "stage":
        "Step 8-5.5",

    "definition": {
        "edge":
            (
                "parent -> child if child preserves "
                "all parent WT-relative substitutions "
                "and adds exactly one additional "
                "substitution"
            ),

        "initial_reachable_set":
            (
                "all observed GFP single mutants"
            ),

        "interpretation_of_unreachable":
            (
                "No complete strict one-addition path "
                "exists within the observed benchmark "
                "variant set. This is a benchmark "
                "coverage statement, not a claim of "
                "biological impossibility."
            ),

        "uses_dms_score":
            False,
    },

    "dataset_counts": {
        "single_mutants":
            n_single,

        "multi_mutants":
            n_multi,

        "all_variants":
            int(
                len(
                    metadata
                )
            ),

        "max_mutation_count":
            max_k,
    },

    "overall_accessibility": {
        "reachable_multi_mutants":
            n_multi_reachable,

        "multi_mutant_reachable_fraction":
            multi_reachable_fraction,

        "double_mutants":
            n_double,

        "reachable_double_mutants":
            n_double_reachable,

        "double_reachable_fraction":
            (
                float(
                    double_reachable_fraction
                )
                if np.isfinite(
                    double_reachable_fraction
                )
                else None
            ),
    },

    "by_mutation_count":
        level_df.to_dict(
            orient="records"
        ),

    "top_predicted_accessibility":
        top_predicted_records,

    "artifacts": {
        "variant_accessibility":
            str(
                VARIANT_ACCESSIBILITY_PATH
            ),

        "mutation_count_summary":
            str(
                LEVEL_SUMMARY_PATH
            ),

        "top_predicted_accessibility":
            (
                str(
                    TOP_PREDICTED_PATH
                )
                if FINAL_PREDICTIONS_PATH.exists()
                else None
            ),

        "figures":
            [
                str(
                    FIGURE_ROOT
                    / "reachable_fraction_by_mutation_count.png"
                ),

                str(
                    FIGURE_ROOT
                    / "observed_vs_reachable_counts.png"
                ),
            ],
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
    "=" * 110
)

print(
    "GFP Mutation-Graph Accessibility Summary"
)

print(
    "=" * 110
)


print()
print(
    "Strict edge definition"
)

print(
    "parent -> child only when child adds "
    "exactly one WT-relative substitution "
    "while preserving every parent substitution."
)


print()
print(
    "Overall"
)

print(
    f"Observed singles                 : "
    f"{n_single}"
)

print(
    f"Observed multi-mutants           : "
    f"{n_multi}"
)

print(
    f"Reachable multi-mutants          : "
    f"{n_multi_reachable}"
)

print(
    f"Multi-mutant reachable fraction  : "
    f"{multi_reachable_fraction:.6f}"
)

print(
    f"Observed double mutants          : "
    f"{n_double}"
)

print(
    f"Reachable double mutants         : "
    f"{n_double_reachable}"
)

if np.isfinite(
    double_reachable_fraction
):

    print(
        f"Double reachable fraction        : "
        f"{double_reachable_fraction:.6f}"
    )


print()
print(
    "By mutation count"
)


display_columns = [
    "mutation_count",
    "n_variants",
    "n_with_observed_parent",
    "fraction_with_observed_parent",
    "n_reachable_from_single",
    "fraction_reachable_from_single",
    "mean_observed_parent_count",
]


print(
    level_df[
        display_columns
    ].to_string(
        index=False
    )
)


if prediction_accessibility is not None:

    print()
    print(
        "Top predicted candidates: accessibility"
    )


    top_summary_df = pd.DataFrame(
        top_predicted_records
    )


    print(
        top_summary_df.to_string(
            index=False
        )
    )


    print()
    print(
        "Top 20 by predicted fitness"
    )


    print(
        prediction_accessibility[
            [
                "mu_rank",
                "mutant",
                "mutation_count",
                "predicted_fitness_mu",
                "uncertainty_sigma",
                "reachable_from_single",
                "reachable_parent_count",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )


else:

    print()
    print(
        "Final prediction file not found; "
        "prediction-rank accessibility was skipped."
    )


print()
print(
    "Saved:"
)

print(
    VARIANT_ACCESSIBILITY_PATH
)

print(
    LEVEL_SUMMARY_PATH
)

if FINAL_PREDICTIONS_PATH.exists():

    print(
        TOP_PREDICTED_PATH
    )

print(
    SUMMARY_PATH
)

print(
    FIGURE_ROOT
    / "reachable_fraction_by_mutation_count.png"
)

print(
    FIGURE_ROOT
    / "observed_vs_reachable_counts.png"
)


print()
print(
    "IMPORTANT:"
)

print(
    "No DMS_score was loaded or used."
)

print(
    "Reachability describes connectivity "
    "inside the observed benchmark only."
)

print(
    "An unreachable benchmark variant may "
    "still be experimentally constructible; "
    "its required intermediate variants are "
    "simply absent from this dataset."
)
