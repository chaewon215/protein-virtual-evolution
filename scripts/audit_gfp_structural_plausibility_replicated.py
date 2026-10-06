from pathlib import Path
import argparse
import json
import random
import re
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig


# ============================================================
# Step 10-v2
# Replicated / matched-seed structural plausibility audit
#
# Why:
#   ESM3 structure generation can be stochastic.
#   Comparing WT, seed, and candidate folded with unrelated
#   RNG seeds confounds sequence effects with fold-generation
#   variability.
#
# Design:
#   replicate seeds = [42, 43, 44] by default
#
#   For each replicate r:
#       WT        folded with RNG seed r
#       every seed folded with RNG seed r
#       every candidate folded with RNG seed r
#
#   This yields matched comparisons:
#       candidate_r vs seed_r
#       candidate_r vs WT_r
#       seed_r      vs WT_r
#
#   We additionally estimate self-variability:
#       seed_r1 vs seed_r2
#       candidate_r1 vs candidate_r2
#       WT_r1 vs WT_r2
#
# Outputs are diagnostics only. No experimental validation.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WT_FASTA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_wt.fasta"
)

ALL_VARIANTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_all_variants.csv"
)

SHORTLIST_PATH = (
    PROJECT_ROOT
    / "results"
    / "gfp"
    / "esm3_shortlist"
    / "esm3_candidate_shortlist.csv"
)

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
    / "structural_plausibility_replicated"
)

PDB_ROOT = (
    OUTPUT_ROOT
    / "pdb"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PDB_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

REPLICATE_METRICS_PATH = (
    OUTPUT_ROOT
    / "replicate_metrics.csv"
)

AGGREGATE_METRICS_PATH = (
    OUTPUT_ROOT
    / "aggregate_structural_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


AA20 = set(
    "ACDEFGHIKLMNPQRSTVWY"
)

MUTATION_RE = re.compile(
    r"^([A-Z])(\d+)([A-Z])$"
)

MODEL_NAMES = [
    "esm3_sm_open_v1",
    "esm3-sm-open-v1",
]


# ============================================================
# Utility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta(path):
    sequence = "".join(
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.startswith(">")
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
            "WT contains non-standard amino acids."
        )

    return sequence


def safe_name(text):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )[:180]


def parse_mutation_positions(mutant):
    positions = []

    for token in str(
        mutant
    ).split(
        ":"
    ):
        match = MUTATION_RE.fullmatch(
            token
        )

        if match is None:
            raise ValueError(
                f"Invalid mutation token: {token}"
            )

        _, pos, _ = match.groups()

        positions.append(
            int(pos) - 1
        )

    return positions


def to_numpy(value):
    if value is None:
        return None

    if torch.is_tensor(
        value
    ):
        return (
            value
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value
    )


# ============================================================
# ESM3
# ============================================================

def load_esm3(device):
    errors = []

    for model_name in MODEL_NAMES:
        try:
            print(
                f"Trying ESM3 model: {model_name}"
            )

            model = (
                ESM3.from_pretrained(
                    model_name
                )
                .to(
                    device
                )
            )

            model.eval()

            print(
                f"Loaded ESM3 model: {model_name}"
            )

            return (
                model,
                model_name,
            )

        except Exception as exc:
            errors.append(
                (
                    model_name,
                    repr(exc),
                )
            )

    raise RuntimeError(
        "Could not load ESM3:\n"
        + "\n".join(
            f"{name}: {err}"
            for name, err in errors
        )
    )


def extract_ca(folded, length):
    coordinates = to_numpy(
        getattr(
            folded,
            "coordinates",
            None,
        )
    )

    if coordinates is None:
        raise ValueError(
            "No coordinates in folded ESMProtein."
        )

    if (
        coordinates.ndim == 4
        and coordinates.shape[0] == 1
    ):
        coordinates = coordinates[0]

    if (
        coordinates.ndim == 3
        and coordinates.shape[0] == length
        and coordinates.shape[-1] == 3
        and coordinates.shape[1] >= 2
    ):
        ca = coordinates[
            :,
            1,
            :
        ]

    elif coordinates.shape == (
        length,
        3,
    ):
        ca = coordinates

    else:
        raise ValueError(
            f"Unsupported coordinate shape: "
            f"{coordinates.shape}"
        )

    if not np.isfinite(
        ca
    ).all():
        raise ValueError(
            "Non-finite CA coordinates."
        )

    return (
        ca.astype(
            np.float64
        ),
        tuple(
            coordinates.shape
        ),
    )


def extract_confidence(folded):
    result = {
        "mean_plddt":
            np.nan,

        "ptm":
            np.nan,
    }

    plddt = getattr(
        folded,
        "plddt",
        None,
    )

    if plddt is not None:
        arr = to_numpy(
            plddt
        )

        finite = arr[
            np.isfinite(
                arr
            )
        ]

        if finite.size:
            result[
                "mean_plddt"
            ] = float(
                finite.mean()
            )

    ptm = getattr(
        folded,
        "ptm",
        None,
    )

    if ptm is not None:
        arr = to_numpy(
            ptm
        )

        finite = arr[
            np.isfinite(
                arr
            )
        ]

        if finite.size:
            result[
                "ptm"
            ] = float(
                finite.mean()
            )

    return result


def fold_sequence(
    model,
    sequence,
    pdb_path,
    num_steps,
    rng_seed,
):
    # Matched common random seed.
    set_seed(
        rng_seed
    )

    protein = ESMProtein(
        sequence=sequence
    )

    folded = model.generate(
        protein,
        GenerationConfig(
            track="structure",
            schedule="cosine",
            num_steps=num_steps,
        ),
    )

    if isinstance(
        folded,
        ESMProteinError,
    ):
        raise RuntimeError(
            f"ESM3 folding error: {folded}"
        )

    if not isinstance(
        folded,
        ESMProtein,
    ):
        raise TypeError(
            f"Unexpected output type: "
            f"{type(folded)}"
        )

    folded.to_pdb(
        str(
            pdb_path
        )
    )

    ca, raw_shape = extract_ca(
        folded,
        len(sequence),
    )

    confidence = extract_confidence(
        folded
    )

    return {
        "ca":
            ca,

        "raw_coordinate_shape":
            raw_shape,

        "mean_plddt":
            confidence[
                "mean_plddt"
            ],

        "ptm":
            confidence[
                "ptm"
            ],
    }


# ============================================================
# Geometry
# ============================================================

def kabsch_align(
    mobile,
    reference,
):
    if mobile.shape != reference.shape:
        raise ValueError(
            "Kabsch shape mismatch."
        )

    mobile_center = mobile.mean(
        axis=0
    )

    reference_center = reference.mean(
        axis=0
    )

    p = (
        mobile
        - mobile_center
    )

    q = (
        reference
        - reference_center
    )

    covariance = (
        p.T
        @ q
    )

    u, _, vt = np.linalg.svd(
        covariance
    )

    rotation = (
        u
        @ vt
    )

    if np.linalg.det(
        rotation
    ) < 0:
        u[
            :,
            -1
        ] *= -1

        rotation = (
            u
            @ vt
        )

    aligned = (
        p
        @ rotation
        + reference_center
    )

    return aligned


def aligned_rmsd(
    mobile,
    reference,
):
    aligned = kabsch_align(
        mobile,
        reference,
    )

    value = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        aligned
                        - reference
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )

    return (
        value,
        aligned,
    )


def radius_of_gyration(ca):
    center = ca.mean(
        axis=0
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        ca
                        - center
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )


def pairwise_self_rmsds(
    fold_list,
):
    values = []

    for i, j in combinations(
        range(
            len(
                fold_list
            )
        ),
        2,
    ):
        value, _ = aligned_rmsd(
            fold_list[
                i
            ][
                "ca"
            ],
            fold_list[
                j
            ][
                "ca"
            ],
        )

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def finite_mean(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if not len(
        values
    ):
        return np.nan

    return float(
        values.mean()
    )


def finite_std(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) <= 1:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.replicates < 2:
        raise ValueError(
            "--replicates must be >= 2"
        )

    replicate_seeds = [
        args.base_seed
        + i

        for i in range(
            args.replicates
        )
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    wt_sequence = read_fasta(
        WT_FASTA_PATH
    )

    sequence_length = len(
        wt_sequence
    )

    num_steps = max(
        8,
        int(
            sequence_length
            / 16
        ),
    )

    print(
        "=" * 120
    )

    print(
        "Step 10-v2: Replicated GFP Structural Plausibility Audit"
    )

    print(
        "=" * 120
    )

    print(
        f"Sequence length      : "
        f"{sequence_length}"
    )

    print(
        f"Replicate seeds      : "
        f"{replicate_seeds}"
    )

    print(
        f"Structure steps      : "
        f"{num_steps}"
    )

    print(
        f"Device               : "
        f"{device}"
    )


    # ========================================================
    # Load shortlist
    # ========================================================

    shortlist = pd.read_csv(
        SHORTLIST_PATH
    )

    if (
        "generated_sequence"
        not in shortlist.columns
    ):
        scored = pd.read_csv(
            SCORED_PATH,
            usecols=[
                "candidate_id",
                "generated_sequence",
            ],
        )

        shortlist = shortlist.merge(
            scored,
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )

    shortlist[
        "generated_sequence"
    ] = shortlist[
        "generated_sequence"
    ].astype(
        str
    )

    if shortlist[
        "generated_sequence"
    ].isna().any():
        raise ValueError(
            "Could not recover candidate sequences."
        )


    # ========================================================
    # Seed sequences
    # ========================================================

    variants = pd.read_csv(
        ALL_VARIANTS_PATH,
        usecols=[
            "mutant",
            "mutated_sequence",
        ],
    )

    variants[
        "mutant"
    ] = variants[
        "mutant"
    ].astype(
        str
    )

    seed_lookup = (
        variants.drop_duplicates(
            "mutant"
        )
        .set_index(
            "mutant"
        )[
            "mutated_sequence"
        ]
        .astype(
            str
        )
        .to_dict()
    )

    unique_seeds = (
        shortlist[
            "seed_mutant"
        ]
        .drop_duplicates()
        .tolist()
    )

    for seed_mutant in unique_seeds:
        if seed_mutant not in seed_lookup:
            raise ValueError(
                f"Missing seed: {seed_mutant}"
            )


    # ========================================================
    # Model
    # ========================================================

    model, model_name = load_esm3(
        device
    )


    # ========================================================
    # Replicated folding
    #
    # WT:
    #   replicate r -> [238,37,3] raw -> [238,3] CA
    #
    # Seeds:
    #   6 × replicates
    #
    # Candidates:
    #   6 × replicates
    # ========================================================

    wt_folds = []

    seed_folds = {
        seed:
            []

        for seed in unique_seeds
    }

    candidate_folds = {
        candidate_id:
            []

        for candidate_id
        in shortlist[
            "candidate_id"
        ]
    }


    for replicate_idx, rng_seed in enumerate(
        replicate_seeds
    ):
        rep_dir = (
            PDB_ROOT
            / f"replicate_{replicate_idx}"
        )

        rep_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "#" * 100
        )

        print(
            f"Replicate "
            f"{replicate_idx + 1}/"
            f"{args.replicates} "
            f"| RNG seed={rng_seed}"
        )

        print(
            "#" * 100
        )


        print(
            "Folding WT..."
        )

        wt_folds.append(
            fold_sequence(
                model=model,
                sequence=wt_sequence,
                pdb_path=(
                    rep_dir
                    / "WT.pdb"
                ),
                num_steps=num_steps,
                rng_seed=rng_seed,
            )
        )


        for seed_idx, seed_mutant in enumerate(
            unique_seeds,
            start=1,
        ):
            print(
                f"Folding seed "
                f"{seed_idx}/"
                f"{len(unique_seeds)}"
            )

            seed_folds[
                seed_mutant
            ].append(
                fold_sequence(
                    model=model,
                    sequence=seed_lookup[
                        seed_mutant
                    ],
                    pdb_path=(
                        rep_dir
                        / (
                            "seed_"
                            + safe_name(
                                seed_mutant
                            )
                            + ".pdb"
                        )
                    ),
                    num_steps=num_steps,
                    rng_seed=rng_seed,
                )
            )


        for cand_idx, row in enumerate(
            shortlist.itertuples(
                index=False
            ),
            start=1,
        ):
            print(
                f"Folding candidate "
                f"{cand_idx}/"
                f"{len(shortlist)}: "
                f"{row.candidate_id}"
            )

            candidate_folds[
                row.candidate_id
            ].append(
                fold_sequence(
                    model=model,
                    sequence=str(
                        row.generated_sequence
                    ),
                    pdb_path=(
                        rep_dir
                        / (
                            safe_name(
                                row.candidate_id
                            )
                            + ".pdb"
                        )
                    ),
                    num_steps=num_steps,
                    rng_seed=rng_seed,
                )
            )


    # ========================================================
    # WT stochastic baseline
    # ========================================================

    wt_self_rmsds = (
        pairwise_self_rmsds(
            wt_folds
        )
    )

    wt_self_median = float(
        np.median(
            wt_self_rmsds
        )
    )

    wt_self_mean = float(
        wt_self_rmsds.mean()
    )

    wt_plddt_mean = finite_mean(
        [
            fold[
                "mean_plddt"
            ]
            for fold in wt_folds
        ]
    )

    wt_ptm_mean = finite_mean(
        [
            fold[
                "ptm"
            ]
            for fold in wt_folds
        ]
    )


    # ========================================================
    # Candidate matched comparisons
    # ========================================================

    replicate_rows = []

    aggregate_rows = []


    for row in shortlist.itertuples(
        index=False
    ):
        candidate_id = str(
            row.candidate_id
        )

        seed_mutant = str(
            row.seed_mutant
        )

        candidate_reps = (
            candidate_folds[
                candidate_id
            ]
        )

        seed_reps = (
            seed_folds[
                seed_mutant
            ]
        )

        seed_self = (
            pairwise_self_rmsds(
                seed_reps
            )
        )

        candidate_self = (
            pairwise_self_rmsds(
                candidate_reps
            )
        )

        seed_self_median = float(
            np.median(
                seed_self
            )
        )

        candidate_self_median = float(
            np.median(
                candidate_self
            )
        )

        matched_candidate_seed = []

        matched_candidate_wt = []

        matched_seed_wt = []

        matched_delta_to_wt = []

        matched_design_disp = []

        matched_mutation_mean_disp = []

        matched_mutation_max_disp = []


        mutation_positions = (
            parse_mutation_positions(
                row.generated_mutant
            )
        )

        design_idx = (
            int(
                row.design_position
            )
            - 1
        )


        for replicate_idx in range(
            args.replicates
        ):
            cand_fold = candidate_reps[
                replicate_idx
            ]

            seed_fold = seed_reps[
                replicate_idx
            ]

            wt_fold = wt_folds[
                replicate_idx
            ]

            cand_seed_rmsd, cand_seed_aligned = (
                aligned_rmsd(
                    cand_fold[
                        "ca"
                    ],
                    seed_fold[
                        "ca"
                    ],
                )
            )

            cand_wt_rmsd, cand_wt_aligned = (
                aligned_rmsd(
                    cand_fold[
                        "ca"
                    ],
                    wt_fold[
                        "ca"
                    ],
                )
            )

            seed_wt_rmsd, _ = (
                aligned_rmsd(
                    seed_fold[
                        "ca"
                    ],
                    wt_fold[
                        "ca"
                    ],
                )
            )

            design_disp = float(
                np.linalg.norm(
                    cand_seed_aligned[
                        design_idx
                    ]
                    - seed_fold[
                        "ca"
                    ][
                        design_idx
                    ]
                )
            )

            mutation_displacements = (
                np.linalg.norm(
                    cand_wt_aligned[
                        mutation_positions
                    ]
                    - wt_fold[
                        "ca"
                    ][
                        mutation_positions
                    ],
                    axis=1,
                )
            )

            delta_to_wt = float(
                cand_wt_rmsd
                - seed_wt_rmsd
            )

            matched_candidate_seed.append(
                cand_seed_rmsd
            )

            matched_candidate_wt.append(
                cand_wt_rmsd
            )

            matched_seed_wt.append(
                seed_wt_rmsd
            )

            matched_delta_to_wt.append(
                delta_to_wt
            )

            matched_design_disp.append(
                design_disp
            )

            matched_mutation_mean_disp.append(
                float(
                    mutation_displacements.mean()
                )
            )

            matched_mutation_max_disp.append(
                float(
                    mutation_displacements.max()
                )
            )

            replicate_rows.append(
                {
                    "replicate":
                        int(
                            replicate_idx
                        ),

                    "rng_seed":
                        int(
                            replicate_seeds[
                                replicate_idx
                            ]
                        ),

                    "candidate_id":
                        candidate_id,

                    "seed_mutant":
                        seed_mutant,

                    "generated_mutant":
                        row.generated_mutant,

                    "candidate_vs_seed_ca_rmsd":
                        cand_seed_rmsd,

                    "candidate_vs_wt_ca_rmsd":
                        cand_wt_rmsd,

                    "seed_vs_wt_ca_rmsd":
                        seed_wt_rmsd,

                    "delta_rmsd_to_wt_vs_seed":
                        delta_to_wt,

                    "design_position_ca_displacement_vs_seed":
                        design_disp,

                    "mutation_site_mean_ca_displacement_vs_wt":
                        float(
                            mutation_displacements.mean()
                        ),

                    "mutation_site_max_ca_displacement_vs_wt":
                        float(
                            mutation_displacements.max()
                        ),

                    "candidate_mean_plddt":
                        cand_fold[
                            "mean_plddt"
                        ],

                    "candidate_ptm":
                        cand_fold[
                            "ptm"
                        ],

                    "seed_mean_plddt":
                        seed_fold[
                            "mean_plddt"
                        ],

                    "seed_ptm":
                        seed_fold[
                            "ptm"
                        ],

                    "candidate_radius_of_gyration":
                        radius_of_gyration(
                            cand_fold[
                                "ca"
                            ]
                        ),

                    "seed_radius_of_gyration":
                        radius_of_gyration(
                            seed_fold[
                                "ca"
                            ]
                        ),

                    "wt_radius_of_gyration":
                        radius_of_gyration(
                            wt_fold[
                                "ca"
                            ]
                        ),
                }
            )


        matched_candidate_seed = np.asarray(
            matched_candidate_seed
        )

        matched_candidate_wt = np.asarray(
            matched_candidate_wt
        )

        matched_seed_wt = np.asarray(
            matched_seed_wt
        )

        matched_delta_to_wt = np.asarray(
            matched_delta_to_wt
        )

        matched_design_disp = np.asarray(
            matched_design_disp
        )

        self_baseline = max(
            seed_self_median,
            candidate_self_median,
        )

        excess_over_self = float(
            np.median(
                matched_candidate_seed
            )
            - self_baseline
        )

        candidate_plddt = [
            rep[
                "mean_plddt"
            ]
            for rep in candidate_reps
        ]

        candidate_ptm = [
            rep[
                "ptm"
            ]
            for rep in candidate_reps
        ]

        seed_plddt = [
            rep[
                "mean_plddt"
            ]
            for rep in seed_reps
        ]

        seed_ptm = [
            rep[
                "ptm"
            ]
            for rep in seed_reps
        ]


        aggregate_rows.append(
            {
                "shortlist_rank":
                    int(
                        row.shortlist_rank
                    ),

                "candidate_id":
                    candidate_id,

                "seed_mutant":
                    seed_mutant,

                "generated_mutant":
                    row.generated_mutant,

                "design_position":
                    int(
                        row.design_position
                    ),

                "predicted_fitness_mu":
                    float(
                        row.predicted_fitness_mu
                    ),

                "uncertainty_sigma":
                    float(
                        row.uncertainty_sigma
                    ),

                "conservative_score":
                    float(
                        row.conservative_score
                    ),

                "predicted_delta_mu_vs_seed":
                    float(
                        row.predicted_delta_mu_vs_seed
                    ),

                "delta_conservative_vs_seed":
                    float(
                        row.delta_conservative_vs_seed
                    ),

                # Matched sequence-comparison metrics
                "candidate_vs_seed_rmsd_mean":
                    float(
                        matched_candidate_seed.mean()
                    ),

                "candidate_vs_seed_rmsd_std":
                    float(
                        matched_candidate_seed.std(
                            ddof=1
                        )
                    ),

                "candidate_vs_seed_rmsd_median":
                    float(
                        np.median(
                            matched_candidate_seed
                        )
                    ),

                "candidate_vs_wt_rmsd_mean":
                    float(
                        matched_candidate_wt.mean()
                    ),

                "seed_vs_wt_rmsd_mean":
                    float(
                        matched_seed_wt.mean()
                    ),

                "delta_rmsd_to_wt_vs_seed_mean":
                    float(
                        matched_delta_to_wt.mean()
                    ),

                "delta_rmsd_to_wt_vs_seed_std":
                    float(
                        matched_delta_to_wt.std(
                            ddof=1
                        )
                    ),

                "design_position_displacement_mean":
                    float(
                        matched_design_disp.mean()
                    ),

                "design_position_displacement_std":
                    float(
                        matched_design_disp.std(
                            ddof=1
                        )
                    ),

                # Stochastic fold baselines
                "wt_self_rmsd_median":
                    wt_self_median,

                "seed_self_rmsd_median":
                    seed_self_median,

                "candidate_self_rmsd_median":
                    candidate_self_median,

                "candidate_seed_rmsd_excess_over_self_variability":
                    excess_over_self,

                # Confidence
                "candidate_mean_plddt_mean":
                    finite_mean(
                        candidate_plddt
                    ),

                "candidate_mean_plddt_std":
                    finite_std(
                        candidate_plddt
                    ),

                "candidate_ptm_mean":
                    finite_mean(
                        candidate_ptm
                    ),

                "candidate_ptm_std":
                    finite_std(
                        candidate_ptm
                    ),

                "seed_mean_plddt_mean":
                    finite_mean(
                        seed_plddt
                    ),

                "seed_ptm_mean":
                    finite_mean(
                        seed_ptm
                    ),
            }
        )


    replicate_df = pd.DataFrame(
        replicate_rows
    )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )


    # ========================================================
    # Keep prediction rank primary.
    # Structural metrics are diagnostics, not a new composite.
    # ========================================================

    aggregate_df = (
        aggregate_df.sort_values(
            "shortlist_rank"
        )
        .reset_index(
            drop=True
        )
    )


    replicate_df.to_csv(
        REPLICATE_METRICS_PATH,
        index=False,
    )

    aggregate_df.to_csv(
        AGGREGATE_METRICS_PATH,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "stage":
            "Step 10-v2",

        "model":
            model_name,

        "sequence_length":
            int(
                sequence_length
            ),

        "replicate_seeds":
            replicate_seeds,

        "n_replicates":
            int(
                args.replicates
            ),

        "structure_steps":
            int(
                num_steps
            ),

        "tensor_shapes": {
            "raw_coordinates_per_fold":
                [
                    238,
                    37,
                    3,
                ],

            "ca_coordinates_per_fold":
                [
                    238,
                    3,
                ],
        },

        "wt_stochastic_baseline": {
            "self_rmsd_mean":
                wt_self_mean,

            "self_rmsd_median":
                wt_self_median,

            "mean_plddt":
                wt_plddt_mean,

            "mean_ptm":
                wt_ptm_mean,
        },

        "interpretation": {
            "candidate_vs_seed":
                (
                    "Matched-seed candidate-vs-seed "
                    "structural difference."
                ),

            "self_variability":
                (
                    "Pairwise RMSD among repeated folds "
                    "of the same sequence."
                ),

            "excess_over_self_variability":
                (
                    "Median candidate-vs-seed RMSD minus "
                    "the larger of seed/candidate self-RMSD "
                    "medians. Positive values suggest that "
                    "between-sequence difference exceeds "
                    "the estimated stochastic folding baseline."
                ),

            "experimental_validation":
                False,
        },

        "artifacts": {
            "replicate_metrics":
                str(
                    REPLICATE_METRICS_PATH
                ),

            "aggregate_metrics":
                str(
                    AGGREGATE_METRICS_PATH
                ),

            "pdb_directory":
                str(
                    PDB_ROOT
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
    # Console
    # ========================================================

    print()
    print(
        "=" * 165
    )

    print(
        "Replicated GFP Structural Plausibility Summary"
    )

    print(
        "=" * 165
    )

    print()
    print(
        "WT stochastic folding baseline"
    )

    print(
        f"WT self-RMSD mean   : "
        f"{wt_self_mean:.4f} Å"
    )

    print(
        f"WT self-RMSD median : "
        f"{wt_self_median:.4f} Å"
    )

    print(
        f"WT mean pLDDT       : "
        f"{wt_plddt_mean:.4f}"
    )

    print(
        f"WT mean pTM         : "
        f"{wt_ptm_mean:.4f}"
    )


    display_columns = [
        "shortlist_rank",
        "candidate_id",
        "predicted_fitness_mu",
        "uncertainty_sigma",
        "conservative_score",
        "candidate_vs_seed_rmsd_mean",
        "candidate_vs_seed_rmsd_std",
        "seed_self_rmsd_median",
        "candidate_self_rmsd_median",
        "candidate_seed_rmsd_excess_over_self_variability",
        "delta_rmsd_to_wt_vs_seed_mean",
        "design_position_displacement_mean",
        "candidate_mean_plddt_mean",
        "candidate_ptm_mean",
    ]

    print()
    print(
        aggregate_df[
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
        REPLICATE_METRICS_PATH
    )

    print(
        AGGREGATE_METRICS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        PDB_ROOT
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do not interpret a large candidate-vs-seed RMSD "
        "without comparing it to seed/candidate self-RMSD."
    )

    print(
        "These are ESM3-predicted structural diagnostics, "
        "not experimental structure validation."
    )


if __name__ == "__main__":
    main()
