from pathlib import Path
import json
import random
import re

import numpy as np
import pandas as pd
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig


# ============================================================
# Step 10
# Structural plausibility audit for final ESM3 GFP shortlist
#
# Fold with ESM3:
#   WT
#   each unique seed
#   each shortlisted generated candidate
#
# Compare C-alpha geometry after Kabsch alignment:
#
# candidate vs WT
# seed      vs WT
# candidate vs seed
#
# Diagnostics:
#   - global CA RMSD
#   - designed-position CA displacement
#   - mutation-site mean CA displacement
#   - radius of gyration
#   - CA bond geometry
#   - non-local CA clash count
#   - ESM3 pLDDT / pTM if exposed by current esm version
#
# IMPORTANT
# ---------
# Structural predictions are computational plausibility checks,
# not experimental validation.
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
    / "structural_plausibility"
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

METRICS_PATH = (
    OUTPUT_ROOT
    / "structural_metrics.csv"
)

FINAL_REPORT_PATH = (
    OUTPUT_ROOT
    / "final_candidate_report.csv"
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
# Basic utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta(path):
    seq = "".join(
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if (
            line.strip()
            and not line.startswith(">")
        )
    )

    if not seq:
        raise ValueError(
            f"Empty FASTA: {path}"
        )

    if not set(seq).issubset(
        AA20
    ):
        raise ValueError(
            "WT contains non-standard amino acids."
        )

    return seq


def parse_mutation_positions(
    mutant_string,
):
    positions = []

    for token in str(
        mutant_string
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
            int(
                pos
            )
            - 1
        )

    return positions


def safe_name(text):
    text = str(
        text
    )

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    )

    return text[:180]


# ============================================================
# Load ESM3
# ============================================================

def load_esm3(device):
    errors = []

    for model_name in MODEL_NAMES:
        try:
            print(
                f"Trying ESM3 model: "
                f"{model_name}"
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
                f"Loaded: {model_name}"
            )

            return (
                model,
                model_name,
            )

        except Exception as exc:
            errors.append(
                (
                    model_name,
                    repr(
                        exc
                    ),
                )
            )

    raise RuntimeError(
        "Could not load ESM3:\n"
        + "\n".join(
            f"{name}: {err}"
            for name, err in errors
        )
    )


# ============================================================
# ESM3 structure generation
# ============================================================

def tensor_or_array_to_numpy(
    value,
):
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


def extract_ca_coordinates(
    folded_protein,
    sequence_length,
):
    coordinates = tensor_or_array_to_numpy(
        getattr(
            folded_protein,
            "coordinates",
            None,
        )
    )

    if coordinates is None:
        raise ValueError(
            "Folded ESMProtein has no coordinates."
        )

    # Remove accidental leading batch dimension.
    if (
        coordinates.ndim == 4
        and coordinates.shape[0] == 1
    ):
        coordinates = coordinates[
            0
        ]

    # Common ESMProtein layouts:
    #
    # [L, 37, 3]
    # [L, 14, 3]
    # [L,  3, 3]  (N, CA, C backbone)
    #
    # CA is atom index 1 in these layouts.
    if (
        coordinates.ndim == 3
        and coordinates.shape[-1] == 3
    ):
        if coordinates.shape[
            0
        ] != sequence_length:
            raise ValueError(
                "Unexpected coordinate residue dimension: "
                f"{coordinates.shape}"
            )

        if coordinates.shape[
            1
        ] < 2:
            raise ValueError(
                "Cannot locate CA atom: "
                f"{coordinates.shape}"
            )

        ca = coordinates[
            :,
            1,
            :
        ]

    elif (
        coordinates.ndim == 2
        and coordinates.shape
        == (
            sequence_length,
            3,
        )
    ):
        ca = coordinates

    else:
        raise ValueError(
            "Unsupported coordinate shape: "
            f"{coordinates.shape}"
        )

    if not np.isfinite(
        ca
    ).all():
        raise ValueError(
            "Non-finite CA coordinates found."
        )

    return (
        ca.astype(
            np.float64
        ),
        tuple(
            coordinates.shape
        ),
    )


def extract_optional_confidence(
    folded_protein,
):
    result = {
        "mean_plddt":
            np.nan,

        "ptm":
            np.nan,
    }

    plddt = getattr(
        folded_protein,
        "plddt",
        None,
    )

    if plddt is not None:
        arr = tensor_or_array_to_numpy(
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
        folded_protein,
        "ptm",
        None,
    )

    if ptm is not None:
        arr = tensor_or_array_to_numpy(
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
    seed,
):
    set_seed(
        seed
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
            "Unexpected ESM3 fold output: "
            f"{type(folded)}"
        )

    folded.to_pdb(
        str(
            pdb_path
        )
    )

    ca, raw_coord_shape = (
        extract_ca_coordinates(
            folded,
            len(
                sequence
            ),
        )
    )

    confidence = (
        extract_optional_confidence(
            folded
        )
    )

    return {
        "ca":
            ca,

        "raw_coordinate_shape":
            raw_coord_shape,

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
# Geometry metrics
# ============================================================

def kabsch_align(
    mobile,
    reference,
):
    if mobile.shape != reference.shape:
        raise ValueError(
            "Kabsch shape mismatch."
        )

    mobile_center = (
        mobile.mean(
            axis=0
        )
    )

    reference_center = (
        reference.mean(
            axis=0
        )
    )

    p = (
        mobile
        - mobile_center
    )

    q = (
        reference
        - reference_center
    )

    h = (
        p.T
        @ q
    )

    u, _, vt = np.linalg.svd(
        h
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


def rmsd(
    a,
    b,
):
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        a
                        - b
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )


def aligned_rmsd(
    mobile,
    reference,
):
    aligned = kabsch_align(
        mobile,
        reference,
    )

    return (
        rmsd(
            aligned,
            reference,
        ),
        aligned,
    )


def radius_of_gyration(
    ca,
):
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


def ca_step_stats(
    ca,
):
    distances = np.linalg.norm(
        ca[
            1:
        ]
        - ca[
            :-1
        ],
        axis=1,
    )

    return {
        "ca_step_mean":
            float(
                distances.mean()
            ),

        "ca_step_std":
            float(
                distances.std()
            ),

        "ca_step_min":
            float(
                distances.min()
            ),

        "ca_step_max":
            float(
                distances.max()
            ),

        "ca_step_outlier_count":
            int(
                (
                    (distances < 3.2)
                    |
                    (distances > 4.5)
                )
                .sum()
            ),
    }


def ca_clash_count(
    ca,
    threshold=3.0,
):
    n = len(
        ca
    )

    count = 0

    minimum = np.inf

    for i in range(
        n
    ):
        for j in range(
            i + 3,
            n,
        ):
            d = float(
                np.linalg.norm(
                    ca[
                        i
                    ]
                    - ca[
                        j
                    ]
                )
            )

            minimum = min(
                minimum,
                d,
            )

            if d < threshold:
                count += 1

    return (
        int(
            count
        ),
        float(
            minimum
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

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

    # Official ESM3 cookbook folding heuristic:
    # num_steps ~= L / 16
    num_steps = max(
        8,
        int(
            sequence_length
            / 16
        ),
    )

    print(
        "=" * 110
    )

    print(
        "Step 10: GFP Structural Plausibility Audit"
    )

    print(
        "=" * 110
    )

    print(
        f"Sequence length       : "
        f"{sequence_length}"
    )

    print(
        f"Structure steps       : "
        f"{num_steps}"
    )

    print(
        f"Device                : "
        f"{device}"
    )


    # ========================================================
    # Load shortlist + candidate sequences
    # ========================================================

    shortlist = pd.read_csv(
        SHORTLIST_PATH
    )

    # Step 9-3 currently preserves all columns from the
    # scored-candidate table, including generated_sequence.
    #
    # Therefore, only recover generated_sequence from the
    # Step 9-2 scored table when it is NOT already present.
    #
    # If we merge it unconditionally, pandas creates:
    #   generated_sequence_x
    #   generated_sequence_y
    # and the later lookup of "generated_sequence" fails.
    if "generated_sequence" not in shortlist.columns:

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

    if shortlist[
        "generated_sequence"
    ].isna().any():
        raise ValueError(
            "Could not recover generated sequences."
        )

    shortlist[
        "generated_sequence"
    ] = shortlist[
        "generated_sequence"
    ].astype(
        str
    )


    # ========================================================
    # Recover seed sequences WITHOUT labels
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
        .to_dict()
    )

    missing_seeds = [
        seed
        for seed in shortlist[
            "seed_mutant"
        ].unique()
        if seed not in seed_lookup
    ]

    if missing_seeds:
        raise ValueError(
            "Missing seed sequences: "
            f"{missing_seeds}"
        )


    # ========================================================
    # Load ESM3
    # ========================================================

    model, model_name = (
        load_esm3(
            device
        )
    )


    # ========================================================
    # Fold WT
    # ========================================================

    print()
    print(
        "Folding WT..."
    )

    wt_fold = fold_sequence(
        model=model,
        sequence=wt_sequence,
        pdb_path=(
            PDB_ROOT
            / "WT.pdb"
        ),
        num_steps=num_steps,
        seed=42,
    )

    wt_ca = wt_fold[
        "ca"
    ]

    wt_rg = radius_of_gyration(
        wt_ca
    )

    wt_clashes, wt_min_nonlocal = (
        ca_clash_count(
            wt_ca
        )
    )

    wt_step = ca_step_stats(
        wt_ca
    )


    print(
        f"WT coordinates        : "
        f"{wt_fold['raw_coordinate_shape']}"
    )

    print(
        f"WT CA                 : "
        f"{wt_ca.shape}"
    )

    print(
        f"WT mean pLDDT         : "
        f"{wt_fold['mean_plddt']}"
    )

    print(
        f"WT pTM                : "
        f"{wt_fold['ptm']}"
    )


    # ========================================================
    # Fold unique seeds
    # ========================================================

    seed_folds = {}

    unique_seeds = (
        shortlist[
            "seed_mutant"
        ]
        .drop_duplicates()
        .tolist()
    )

    for seed_idx, seed_mutant in enumerate(
        unique_seeds,
        start=1,
    ):

        print()
        print(
            f"Folding seed "
            f"{seed_idx}/{len(unique_seeds)}"
        )

        seed_sequence = str(
            seed_lookup[
                seed_mutant
            ]
        )

        seed_fold = fold_sequence(
            model=model,
            sequence=seed_sequence,
            pdb_path=(
                PDB_ROOT
                / (
                    "seed_"
                    + safe_name(
                        seed_mutant
                    )
                    + ".pdb"
                )
            ),
            num_steps=num_steps,
            seed=1000
            + seed_idx,
        )

        seed_folds[
            seed_mutant
        ] = seed_fold


    # ========================================================
    # Fold candidates and compute metrics
    # ========================================================

    rows = []

    for candidate_idx, row in enumerate(
        shortlist.itertuples(
            index=False
        ),
        start=1,
    ):

        print()
        print(
            f"Folding candidate "
            f"{candidate_idx}/{len(shortlist)}: "
            f"{row.candidate_id}"
        )

        candidate_sequence = str(
            row.generated_sequence
        )

        candidate_fold = fold_sequence(
            model=model,
            sequence=candidate_sequence,
            pdb_path=(
                PDB_ROOT
                / (
                    safe_name(
                        row.candidate_id
                    )
                    + ".pdb"
                )
            ),
            num_steps=num_steps,
            seed=2000
            + candidate_idx,
        )

        candidate_ca = (
            candidate_fold[
                "ca"
            ]
        )

        seed_fold = seed_folds[
            str(
                row.seed_mutant
            )
        ]

        seed_ca = seed_fold[
            "ca"
        ]


        # ----------------------------------------------------
        # Global RMSD
        # ----------------------------------------------------

        cand_vs_wt_rmsd, cand_wt_aligned = (
            aligned_rmsd(
                candidate_ca,
                wt_ca,
            )
        )

        seed_vs_wt_rmsd, seed_wt_aligned = (
            aligned_rmsd(
                seed_ca,
                wt_ca,
            )
        )

        cand_vs_seed_rmsd, cand_seed_aligned = (
            aligned_rmsd(
                candidate_ca,
                seed_ca,
            )
        )


        # ----------------------------------------------------
        # Designed-position displacement vs seed
        # after global candidate-to-seed alignment
        # ----------------------------------------------------

        design_idx = int(
            row.design_position
        ) - 1

        design_displacement_vs_seed = float(
            np.linalg.norm(
                cand_seed_aligned[
                    design_idx
                ]
                - seed_ca[
                    design_idx
                ]
            )
        )


        # ----------------------------------------------------
        # All mutation-site displacement vs WT
        # after candidate-to-WT alignment
        # ----------------------------------------------------

        mutation_positions = (
            parse_mutation_positions(
                row.generated_mutant
            )
        )

        mutation_site_displacements = (
            np.linalg.norm(
                cand_wt_aligned[
                    mutation_positions
                ]
                - wt_ca[
                    mutation_positions
                ],
                axis=1,
            )
        )

        mutation_site_mean_disp = float(
            mutation_site_displacements.mean()
        )

        mutation_site_max_disp = float(
            mutation_site_displacements.max()
        )


        # ----------------------------------------------------
        # Gross geometry diagnostics
        # ----------------------------------------------------

        candidate_rg = (
            radius_of_gyration(
                candidate_ca
            )
        )

        candidate_clashes, candidate_min_nonlocal = (
            ca_clash_count(
                candidate_ca
            )
        )

        candidate_step = (
            ca_step_stats(
                candidate_ca
            )
        )


        # ----------------------------------------------------
        # Heuristic structural-warning flag
        #
        # This is intentionally conservative and diagnostic.
        # It is NOT a learned quality score.
        # ----------------------------------------------------

        warnings = []

        if (
            cand_vs_seed_rmsd
            > 2.0
        ):
            warnings.append(
                "candidate_seed_rmsd_gt_2A"
            )

        if (
            design_displacement_vs_seed
            > 3.0
        ):
            warnings.append(
                "design_site_displacement_gt_3A"
            )

        if (
            candidate_clashes
            > wt_clashes
            + 2
        ):
            warnings.append(
                "extra_ca_clashes"
            )

        if (
            candidate_step[
                "ca_step_outlier_count"
            ]
            > wt_step[
                "ca_step_outlier_count"
            ]
            + 2
        ):
            warnings.append(
                "ca_geometry_outliers"
            )


        rows.append(
            {
                "shortlist_rank":
                    int(
                        row.shortlist_rank
                    ),

                "candidate_id":
                    row.candidate_id,

                "seed_mutant":
                    row.seed_mutant,

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

                "candidate_coordinate_shape":
                    str(
                        candidate_fold[
                            "raw_coordinate_shape"
                        ]
                    ),

                "candidate_mean_plddt":
                    candidate_fold[
                        "mean_plddt"
                    ],

                "candidate_ptm":
                    candidate_fold[
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

                "candidate_vs_wt_ca_rmsd":
                    cand_vs_wt_rmsd,

                "seed_vs_wt_ca_rmsd":
                    seed_vs_wt_rmsd,

                "candidate_vs_seed_ca_rmsd":
                    cand_vs_seed_rmsd,

                "delta_rmsd_to_wt_vs_seed":
                    float(
                        cand_vs_wt_rmsd
                        - seed_vs_wt_rmsd
                    ),

                "design_position_ca_displacement_vs_seed":
                    design_displacement_vs_seed,

                "mutation_site_mean_ca_displacement_vs_wt":
                    mutation_site_mean_disp,

                "mutation_site_max_ca_displacement_vs_wt":
                    mutation_site_max_disp,

                "candidate_radius_of_gyration":
                    candidate_rg,

                "wt_radius_of_gyration":
                    wt_rg,

                "delta_radius_of_gyration_vs_wt":
                    float(
                        candidate_rg
                        - wt_rg
                    ),

                "candidate_ca_clash_count_lt3A":
                    candidate_clashes,

                "wt_ca_clash_count_lt3A":
                    wt_clashes,

                "candidate_min_nonlocal_ca_distance":
                    candidate_min_nonlocal,

                "wt_min_nonlocal_ca_distance":
                    wt_min_nonlocal,

                **{
                    f"candidate_{key}":
                        value
                    for key, value
                    in candidate_step.items()
                },

                "structural_warning_count":
                    int(
                        len(
                            warnings
                        )
                    ),

                "structural_warnings":
                    (
                        ";".join(
                            warnings
                        )
                        if warnings
                        else ""
                    ),
            }
        )


    metrics_df = pd.DataFrame(
        rows
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )


    # ========================================================
    # Final report
    #
    # Keep original model-guided ranking as the primary rank.
    # Structural metrics are appended as plausibility evidence;
    # we do NOT create an arbitrary weighted composite score.
    # ========================================================

    final_report = (
        metrics_df.sort_values(
            [
                "structural_warning_count",
                "conservative_score",
                "candidate_vs_seed_ca_rmsd",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    final_report[
        "structural_review_rank"
    ] = (
        np.arange(
            len(
                final_report
            )
        )
        + 1
    )

    final_report.to_csv(
        FINAL_REPORT_PATH,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "stage":
            "Step 10",

        "model":
            model_name,

        "sequence_length":
            sequence_length,

        "structure_generation_steps":
            num_steps,

        "coordinate_reference":
            (
                "ESM3-predicted WT structure "
                "using the same model"
            ),

        "n_candidates":
            int(
                len(
                    metrics_df
                )
            ),

        "wt": {
            "coordinate_shape":
                list(
                    wt_fold[
                        "raw_coordinate_shape"
                    ]
                ),

            "ca_shape":
                list(
                    wt_ca.shape
                ),

            "mean_plddt":
                (
                    None
                    if np.isnan(
                        wt_fold[
                            "mean_plddt"
                        ]
                    )
                    else float(
                        wt_fold[
                            "mean_plddt"
                        ]
                    )
                ),

            "ptm":
                (
                    None
                    if np.isnan(
                        wt_fold[
                            "ptm"
                        ]
                    )
                    else float(
                        wt_fold[
                            "ptm"
                        ]
                    )
                ),

            "radius_of_gyration":
                wt_rg,

            "ca_clash_count_lt3A":
                wt_clashes,
        },

        "heuristic_warning_rules": {
            "candidate_vs_seed_ca_rmsd":
                "> 2.0 A",

            "design_position_ca_displacement":
                "> 3.0 A",

            "additional_ca_clashes_vs_wt":
                "> 2",

            "additional_ca_step_outliers_vs_wt":
                "> 2",
        },

        "interpretation":
            (
                "Structural metrics are computational "
                "plausibility diagnostics only. "
                "No experimental structural or fitness "
                "validation is implied."
            ),

        "artifacts": {
            "metrics":
                str(
                    METRICS_PATH
                ),

            "final_candidate_report":
                str(
                    FINAL_REPORT_PATH
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
    # Console report
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "GFP Structural Plausibility Summary"
    )

    print(
        "=" * 140
    )

    display_columns = [
        "structural_review_rank",
        "candidate_id",
        "generated_mutant",
        "predicted_fitness_mu",
        "uncertainty_sigma",
        "conservative_score",
        "candidate_mean_plddt",
        "candidate_ptm",
        "candidate_vs_wt_ca_rmsd",
        "candidate_vs_seed_ca_rmsd",
        "design_position_ca_displacement_vs_seed",
        "structural_warning_count",
        "structural_warnings",
    ]

    print(
        final_report[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "WT reference"
    )

    print(
        f"WT coordinate shape     : "
        f"{wt_fold['raw_coordinate_shape']}"
    )

    print(
        f"WT CA shape             : "
        f"{wt_ca.shape}"
    )

    print(
        f"WT mean pLDDT           : "
        f"{wt_fold['mean_plddt']}"
    )

    print(
        f"WT pTM                  : "
        f"{wt_fold['ptm']}"
    )

    print(
        f"WT radius of gyration   : "
        f"{wt_rg:.4f}"
    )

    print(
        f"WT CA clashes (<3A)     : "
        f"{wt_clashes}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        METRICS_PATH
    )

    print(
        FINAL_REPORT_PATH
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
        "The final structural_review_rank is "
        "a plausibility review ordering, not "
        "experimental evidence of higher GFP fitness."
    )

    print(
        "The original conservative fitness ranking "
        "is retained in the output and is not "
        "replaced by an arbitrary weighted score."
    )


if __name__ == "__main__":
    main()
