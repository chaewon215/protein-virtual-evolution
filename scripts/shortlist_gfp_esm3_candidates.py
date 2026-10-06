from pathlib import Path
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# Step 9-3
# Diverse uncertainty-aware shortlist of ESM3 GFP candidates
#
# This stage uses ONLY model predictions.
# No experimental DMS_score exists for generated candidates.
#
# Default shortlist rules:
#   1. generated candidate must improve over its seed in:
#        - predicted mu
#        - conservative score (mu - sigma)
#   2. rank by conservative score
#   3. max 1 candidate per seed
#   4. max 2 candidates per newly designed position
#   5. return top 8 diverse candidates
#
# A Pareto flag is also computed:
#   maximize mu
#   minimize sigma
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORED_PATH = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "esm3_scoring"
    / "esm3_generated_candidates_scored.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "esm3_shortlist"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SHORTLIST_PATH = (
    OUTPUT_ROOT
    / "esm3_candidate_shortlist.csv"
)

RANKED_PATH = (
    OUTPUT_ROOT
    / "esm3_candidates_ranked.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


def pareto_optimal_mask(
    mu,
    sigma,
):
    """
    Candidate i is dominated if another candidate j has:

        mu_j >= mu_i
        sigma_j <= sigma_i

    with at least one strict inequality.

    Objectives:
        maximize mu
        minimize sigma
    """

    n = len(
        mu
    )

    optimal = np.ones(
        n,
        dtype=bool,
    )

    for i in range(
        n
    ):
        dominated = (
            (mu >= mu[i])
            &
            (sigma <= sigma[i])
            &
            (
                (mu > mu[i])
                |
                (sigma < sigma[i])
            )
        )

        if dominated.any():
            optimal[i] = False

    return optimal


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-per-seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-per-position",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-sigma",
        type=float,
        default=None,
        help=(
            "Optional hard uncertainty threshold. "
            "Default: no hard threshold."
        ),
    )

    parser.add_argument(
        "--allow-nonimproved",
        action="store_true",
        help=(
            "Allow candidates that do not improve "
            "conservative score over their seed."
        ),
    )

    args = parser.parse_args()


    if args.top_n <= 0:
        raise ValueError(
            "--top-n must be > 0"
        )


    df = pd.read_csv(
        SCORED_PATH
    )


    required_columns = {
        "candidate_id",
        "seed_mutant",
        "seed_mu",
        "seed_sigma",
        "seed_conservative_score",
        "design_position",
        "generated_aa",
        "generated_mutant",
        "mutation_count",
        "predicted_fitness_mu",
        "uncertainty_sigma",
        "conservative_score",
        "ucb_score",
        "predicted_delta_mu_vs_seed",
        "delta_conservative_vs_seed",
    }


    missing = (
        required_columns
        - set(
            df.columns
        )
    )


    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )


    # ========================================================
    # Pareto analysis
    # ========================================================

    df[
        "pareto_optimal_mu_sigma"
    ] = pareto_optimal_mask(
        df[
            "predicted_fitness_mu"
        ]
        .to_numpy(
            dtype=np.float64
        ),

        df[
            "uncertainty_sigma"
        ]
        .to_numpy(
            dtype=np.float64
        ),
    )


    # ========================================================
    # Transparent ranking columns
    # ========================================================

    df[
        "rank_by_mu_global"
    ] = (
        df[
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


    df[
        "rank_by_conservative_global"
    ] = (
        df[
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


    df[
        "rank_by_low_uncertainty"
    ] = (
        df[
            "uncertainty_sigma"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(
            int
        )
    )


    # ========================================================
    # Eligibility
    # ========================================================

    eligible = df.copy()


    if not args.allow_nonimproved:

        eligible = eligible[
            (
                eligible[
                    "predicted_delta_mu_vs_seed"
                ]
                > 0
            )
            &
            (
                eligible[
                    "delta_conservative_vs_seed"
                ]
                > 0
            )
        ].copy()


    if args.max_sigma is not None:

        eligible = eligible[
            eligible[
                "uncertainty_sigma"
            ]
            <= args.max_sigma
        ].copy()


    # Primary ordering:
    # 1. conservative score
    # 2. mu
    # 3. lower sigma
    eligible = (
        eligible.sort_values(
            [
                "conservative_score",
                "predicted_fitness_mu",
                "uncertainty_sigma",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Diversity-aware greedy shortlist
    # ========================================================

    seed_counter = Counter()

    position_counter = Counter()

    selected_indices = []


    for row_idx, row in eligible.iterrows():

        seed = str(
            row[
                "seed_mutant"
            ]
        )

        position = int(
            row[
                "design_position"
            ]
        )


        if (
            seed_counter[
                seed
            ]
            >= args.max_per_seed
        ):
            continue


        if (
            position_counter[
                position
            ]
            >= args.max_per_position
        ):
            continue


        selected_indices.append(
            row_idx
        )

        seed_counter[
            seed
        ] += 1

        position_counter[
            position
        ] += 1


        if len(
            selected_indices
        ) >= args.top_n:
            break


    shortlist = (
        eligible.loc[
            selected_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    shortlist[
        "shortlist_rank"
    ] = (
        np.arange(
            len(
                shortlist
            )
        )
        + 1
    )


    # ========================================================
    # Human-readable rationale
    # ========================================================

    rationale = []


    for row in shortlist.itertuples(
        index=False
    ):

        pieces = [
            (
                f"conservative={row.conservative_score:.3f}"
            ),
            (
                f"delta_cons={row.delta_conservative_vs_seed:+.3f}"
            ),
            (
                f"sigma={row.uncertainty_sigma:.3f}"
            ),
        ]


        if bool(
            row.pareto_optimal_mu_sigma
        ):
            pieces.append(
                "Pareto-optimal(mu,sigma)"
            )


        rationale.append(
            "; ".join(
                pieces
            )
        )


    shortlist[
        "selection_rationale"
    ] = rationale


    # ========================================================
    # Save
    # ========================================================

    df.sort_values(
        [
            "conservative_score",
            "predicted_fitness_mu",
        ],
        ascending=[
            False,
            False,
        ],
    ).to_csv(
        RANKED_PATH,
        index=False,
    )


    shortlist.to_csv(
        SHORTLIST_PATH,
        index=False,
    )


    summary = {
        "stage":
            "Step 9-3",

        "input_candidates":
            int(
                len(
                    df
                )
            ),

        "eligible_candidates":
            int(
                len(
                    eligible
                )
            ),

        "shortlist_size":
            int(
                len(
                    shortlist
                )
            ),

        "selection": {
            "primary_score":
                "mu - sigma",

            "require_mu_improvement_vs_seed":
                not args.allow_nonimproved,

            "require_conservative_improvement_vs_seed":
                not args.allow_nonimproved,

            "max_sigma":
                args.max_sigma,

            "max_per_seed":
                int(
                    args.max_per_seed
                ),

            "max_per_design_position":
                int(
                    args.max_per_position
                ),

            "pareto_objectives":
                [
                    "maximize predicted_fitness_mu",
                    "minimize uncertainty_sigma",
                ],
        },

        "shortlist_statistics": {
            "mu_mean":
                (
                    float(
                        shortlist[
                            "predicted_fitness_mu"
                        ].mean()
                    )
                    if len(
                        shortlist
                    )
                    else None
                ),

            "sigma_mean":
                (
                    float(
                        shortlist[
                            "uncertainty_sigma"
                        ].mean()
                    )
                    if len(
                        shortlist
                    )
                    else None
                ),

            "conservative_mean":
                (
                    float(
                        shortlist[
                            "conservative_score"
                        ].mean()
                    )
                    if len(
                        shortlist
                    )
                    else None
                ),

            "mean_delta_mu_vs_seed":
                (
                    float(
                        shortlist[
                            "predicted_delta_mu_vs_seed"
                        ].mean()
                    )
                    if len(
                        shortlist
                    )
                    else None
                ),

            "mean_delta_conservative_vs_seed":
                (
                    float(
                        shortlist[
                            "delta_conservative_vs_seed"
                        ].mean()
                    )
                    if len(
                        shortlist
                    )
                    else None
                ),

            "pareto_optimal_count":
                int(
                    shortlist[
                        "pareto_optimal_mu_sigma"
                    ].sum()
                )
                if len(
                    shortlist
                )
                else 0,
        },

        "artifacts": {
            "ranked_candidates":
                str(
                    RANKED_PATH
                ),

            "shortlist":
                str(
                    SHORTLIST_PATH
                ),
        },

        "interpretation":
            (
                "Shortlist values are model predictions only. "
                "No generated candidate has an experimental "
                "fitness measurement."
            ),
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
    # Console summary
    # ========================================================

    print(
        "=" * 125
    )

    print(
        "Step 9-3: Diverse ESM3 GFP Candidate Shortlist"
    )

    print(
        "=" * 125
    )


    print(
        f"Input candidates     : "
        f"{len(df)}"
    )

    print(
        f"Eligible candidates  : "
        f"{len(eligible)}"
    )

    print(
        f"Shortlist size       : "
        f"{len(shortlist)}"
    )

    print(
        f"Max / seed           : "
        f"{args.max_per_seed}"
    )

    print(
        f"Max / design position: "
        f"{args.max_per_position}"
    )

    print(
        f"Hard max sigma       : "
        f"{args.max_sigma}"
    )


    if len(
        shortlist
    ):

        print()
        print(
            "Final shortlist"
        )


        display_columns = [
            "shortlist_rank",
            "candidate_id",
            "seed_mutant",
            "design_position",
            "generated_aa",
            "generated_mutant",
            "mutation_count",
            "predicted_fitness_mu",
            "uncertainty_sigma",
            "conservative_score",
            "predicted_delta_mu_vs_seed",
            "delta_conservative_vs_seed",
            "pareto_optimal_mu_sigma",
        ]


        print(
            shortlist[
                display_columns
            ].to_string(
                index=False
            )
        )


    print()
    print(
        "Saved:"
    )

    print(
        RANKED_PATH
    )

    print(
        SHORTLIST_PATH
    )

    print(
        SUMMARY_PATH
    )


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This shortlist is a computational "
        "prioritization result, not experimental "
        "validation."
    )

    print(
        "The diversity constraints intentionally "
        "prevent the final list from being dominated "
        "by one seed or one design hotspot."
    )


if __name__ == "__main__":
    main()
