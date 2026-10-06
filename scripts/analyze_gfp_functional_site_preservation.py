#!/usr/bin/env python3
"""Step 10-3: GFP functional-site preservation analysis.

Goal
----
Evaluate whether ESM3-generated GFP candidates retain WT-like structural
geometry around the chromophore-forming motif and key surrounding residues
under independent ColabFold/AlphaFold2 predictions.

Important limitation
--------------------
AlphaFold2/ColabFold predicts the polypeptide structure but does not model
the post-translational chemical maturation of the GFP chromophore. Therefore
this script assesses *structural functional plausibility*, not experimental
fluorescence or chromophore maturation.

Default GFP positions (1-based)
-------------------------------
Chromophore-forming motif:
    65, 66, 67

Key surrounding residues:
    94, 96, 148, 203, 205, 222

The chromophore-neighborhood is additionally derived geometrically from WT
predictions as residues having any heavy atom within a configurable radius
(default 8 A) of any heavy atom in residues 65-67. A majority vote across WT
replicates makes this neighborhood less sensitive to a single prediction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_position_list(text: str) -> list[int]:
    vals = [x for x in re.split(r"[;,\s]+", str(text).strip()) if x]
    out = sorted({int(x) for x in vals})
    if not out or min(out) < 1:
        raise ValueError(f"Invalid residue position list: {text!r}")
    return out


def parse_tokens(filename: str) -> tuple[int | None, int | None, int | None]:
    rank_m = re.search(r"rank[_-]?(\d+)", filename)
    model_m = re.search(r"model[_-]?(\d+)", filename)
    seed_matches = re.findall(r"seed[_-]?(\d+)", filename)
    rank = int(rank_m.group(1)) if rank_m else None
    model = int(model_m.group(1)) if model_m else None
    seed = int(seed_matches[-1]) if seed_matches else None
    return rank, model, seed


def replicate_key(rank: int | None, model: int | None, seed: int | None) -> str:
    if model is not None and seed is not None:
        return f"model_{model}_seed_{seed}"
    if seed is not None:
        return f"seed_{seed}"
    if rank is not None:
        return f"rank_{rank}"
    return "replicate_unknown"


def sequence_id_from_filename(filename: str, sequence_ids: list[str]) -> str | None:
    for seq_id in sorted(sequence_ids, key=len, reverse=True):
        if filename == seq_id or filename.startswith(seq_id + "_"):
            return seq_id
    return None


def parse_pdb(path: Path) -> dict:
    """Parse first model of a PDB into position-indexed atom coordinates."""
    atoms_by_pos: dict[int, list[np.ndarray]] = {}
    ca_by_pos: dict[int, np.ndarray] = {}
    ca_bfactor_by_pos: dict[int, float] = {}

    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue

        alt = line[16:17]
        if alt not in (" ", "A"):
            continue

        atom_name = line[12:16].strip()
        try:
            pos = int(line[22:26])
            xyz = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ], dtype=float)
            bfac = float(line[60:66])
        except ValueError:
            continue

        atoms_by_pos.setdefault(pos, []).append(xyz)
        if atom_name == "CA" and pos not in ca_by_pos:
            ca_by_pos[pos] = xyz
            ca_bfactor_by_pos[pos] = bfac

    if not ca_by_pos:
        raise ValueError(f"No C-alpha atoms parsed from {path}")

    return {
        "atoms_by_pos": {
            pos: np.vstack(coords)
            for pos, coords in atoms_by_pos.items()
        },
        "ca_by_pos": ca_by_pos,
        "ca_bfactor_by_pos": ca_bfactor_by_pos,
    }


def ca_matrix(structure: dict, positions: list[int]) -> np.ndarray:
    missing = [p for p in positions if p not in structure["ca_by_pos"]]
    if missing:
        raise ValueError(f"Missing C-alpha atoms at positions: {missing}")
    return np.vstack([structure["ca_by_pos"][p] for p in positions])


def bfactor_vector(structure: dict, positions: list[int]) -> np.ndarray:
    missing = [p for p in positions if p not in structure["ca_bfactor_by_pos"]]
    if missing:
        raise ValueError(f"Missing C-alpha B-factors at positions: {missing}")
    return np.asarray(
        [structure["ca_bfactor_by_pos"][p] for p in positions],
        dtype=float,
    )


def kabsch_transform(reference: np.ndarray, mobile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation R and translation t so mobile @ R.T + t aligns to reference."""
    reference = np.asarray(reference, dtype=float)
    mobile = np.asarray(mobile, dtype=float)

    if (
        reference.shape != mobile.shape
        or reference.ndim != 2
        or reference.shape[1] != 3
    ):
        raise ValueError(
            f"Coordinate shape mismatch: {reference.shape} vs {mobile.shape}"
        )

    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)

    X = mobile - mob_center
    Y = reference - ref_center
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = ref_center - mob_center @ R.T
    return R, t


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=float) @ R.T + t


def rmsd_direct(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def aligned_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    R, t = kabsch_transform(reference, mobile)
    return rmsd_direct(reference, apply_transform(mobile, R, t))


def pairwise_distance_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if len(a) < 2:
        return 0.0

    da = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    db = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)
    iu = np.triu_indices(len(a), k=1)
    return float(np.sqrt(np.mean((da[iu] - db[iu]) ** 2)))


def minimum_atom_distance(
    structure: dict,
    positions_a: list[int],
    positions_b: list[int],
) -> float:
    a_atoms = [
        structure["atoms_by_pos"][p]
        for p in positions_a
        if p in structure["atoms_by_pos"]
    ]
    b_atoms = [
        structure["atoms_by_pos"][p]
        for p in positions_b
        if p in structure["atoms_by_pos"]
    ]
    if not a_atoms or not b_atoms:
        return float("nan")

    A = np.vstack(a_atoms)
    B = np.vstack(b_atoms)
    distances = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return float(np.min(distances))


def neighborhood_positions(
    structure: dict,
    chromophore_positions: list[int],
    radius: float,
) -> set[int]:
    chrom_atoms = [
        structure["atoms_by_pos"][p]
        for p in chromophore_positions
        if p in structure["atoms_by_pos"]
    ]
    if len(chrom_atoms) != len(chromophore_positions):
        missing = [
            p for p in chromophore_positions
            if p not in structure["atoms_by_pos"]
        ]
        raise ValueError(
            f"Chromophore positions missing from structure: {missing}"
        )

    chrom = np.vstack(chrom_atoms)
    selected = set()
    for pos, atoms in structure["atoms_by_pos"].items():
        d = np.linalg.norm(atoms[:, None, :] - chrom[None, :, :], axis=-1)
        if float(np.min(d)) <= radius:
            selected.add(int(pos))
    return selected


def derive_mutations(wt_seq: str, candidate_seq: str) -> tuple[list[int], list[str]]:
    if len(wt_seq) != len(candidate_seq):
        raise ValueError(
            f"WT/candidate sequence length mismatch: "
            f"{len(wt_seq)} vs {len(candidate_seq)}"
        )
    positions = []
    labels = []
    for i, (wt, mut) in enumerate(zip(wt_seq, candidate_seq), start=1):
        if wt != mut:
            positions.append(i)
            labels.append(f"{wt}{i}{mut}")
    return positions, labels


def median_or_nan(values) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else float("nan")


def mean_or_nan(values) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gfp/functional_site_preservation"),
    )
    parser.add_argument(
        "--chromophore-residues",
        default="65,66,67",
        help="1-based GFP chromophore-forming motif positions.",
    )
    parser.add_argument(
        "--environment-residues",
        default="94,96,148,203,205,222",
        help="1-based key chromophore-environment positions.",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=float,
        default=8.0,
        help="Heavy-atom radius in Angstrom for WT chromophore neighborhood.",
    )
    args = parser.parse_args()

    chrom_positions = parse_position_list(args.chromophore_residues)
    env_positions = parse_position_list(args.environment_residues)
    functional_positions = sorted(set(chrom_positions + env_positions))

    pred_dir = args.predictions_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_csv(args.pair_manifest, dtype=str)
    seq_df = pd.read_csv(args.sequence_manifest, dtype=str)

    required_seq_cols = {"sequence_id", "sequence_length", "sequence"}
    missing_cols = required_seq_cols - set(seq_df.columns)
    if missing_cols:
        raise ValueError(
            f"sequence_manifest missing required columns: {sorted(missing_cols)}"
        )

    seq_df["sequence_id"] = seq_df["sequence_id"].astype(str)
    seq_map = dict(zip(seq_df["sequence_id"], seq_df["sequence"].astype(str)))
    len_map = dict(
        zip(
            seq_df["sequence_id"],
            seq_df["sequence_length"].astype(int),
        )
    )
    sequence_ids = seq_df["sequence_id"].tolist()

    if "WT" not in seq_map:
        raise ValueError("sequence_manifest must contain sequence_id='WT'")

    wt_seq = seq_map["WT"]
    L = len(wt_seq)
    invalid_positions = [
        p for p in functional_positions
        if p < 1 or p > L
    ]
    if invalid_positions:
        raise ValueError(
            f"Functional residue positions outside sequence length {L}: "
            f"{invalid_positions}"
        )

    pdbs = sorted(pred_dir.glob("**/*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"No PDB files found under {pred_dir}")

    structure_map: dict[str, dict] = {}
    inventory_rows = []

    for pdb in pdbs:
        seq_id = sequence_id_from_filename(pdb.name, sequence_ids)
        if seq_id is None:
            continue

        structure = parse_pdb(pdb)
        positions = sorted(structure["ca_by_pos"])
        expected = int(len_map[seq_id])

        if len(positions) != expected:
            raise ValueError(
                f"{pdb.name}: parsed {len(positions)} C-alpha atoms, "
                f"expected sequence length {expected}"
            )
        if positions != list(range(1, expected + 1)):
            raise ValueError(
                f"{pdb.name}: residue numbering is not contiguous 1..{expected}"
            )

        rank, model, seed = parse_tokens(pdb.name)
        rep_key = replicate_key(rank, model, seed)
        map_key = f"{seq_id}|{rep_key}"

        if map_key in structure_map:
            raise ValueError(
                f"Duplicate structure for sequence/replicate: {map_key}"
            )
        structure_map[map_key] = structure

        all_ca_b = np.asarray(
            [structure["ca_bfactor_by_pos"][p] for p in range(1, expected + 1)],
            dtype=float,
        )
        inventory_rows.append({
            "sequence_id": seq_id,
            "replicate_key": rep_key,
            "structure_path": str(pdb),
            "sequence_length": expected,
            "mean_plddt": float(np.mean(all_ca_b)),
        })

    inventory = pd.DataFrame(inventory_rows)
    if inventory.empty:
        raise RuntimeError(
            "PDB files found, but none matched IDs from sequence_manifest"
        )
    inventory.to_csv(out_dir / "structure_inventory.csv", index=False)

    inv_by_id = {
        sid: grp.set_index("replicate_key", drop=False)
        for sid, grp in inventory.groupby("sequence_id")
    }
    if "WT" not in inv_by_id:
        raise RuntimeError("No WT ColabFold prediction was found")

    wt_rep_keys = sorted(inv_by_id["WT"].index.tolist())
    if len(wt_rep_keys) < 2:
        raise RuntimeError(
            "At least two WT replicates are required for a WT self-variability baseline"
        )

    # Canonical chromophore neighborhood: majority vote over WT replicates.
    neigh_counter = Counter()
    per_wt_neighborhood = {}
    for key in wt_rep_keys:
        st = structure_map[f"WT|{key}"]
        neigh = neighborhood_positions(
            st,
            chrom_positions,
            args.neighborhood_radius,
        )
        per_wt_neighborhood[key] = neigh
        neigh_counter.update(neigh)

    majority_needed = len(wt_rep_keys) // 2 + 1
    canonical_neighborhood = sorted(
        p for p, count in neigh_counter.items()
        if count >= majority_needed
    )
    if not canonical_neighborhood:
        raise RuntimeError("Derived chromophore neighborhood is empty")

    all_positions = list(range(1, L + 1))

    # WT self-variability baseline.
    wt_self_rows = []
    for key_a, key_b in itertools.combinations(wt_rep_keys, 2):
        a = structure_map[f"WT|{key_a}"]
        b = structure_map[f"WT|{key_b}"]

        a_all = ca_matrix(a, all_positions)
        b_all = ca_matrix(b, all_positions)
        global_rmsd = aligned_rmsd(a_all, b_all)

        a_func = ca_matrix(a, functional_positions)
        b_func = ca_matrix(b, functional_positions)
        a_pocket = ca_matrix(a, canonical_neighborhood)
        b_pocket = ca_matrix(b, canonical_neighborhood)

        wt_self_rows.append({
            "replicate_a": key_a,
            "replicate_b": key_b,
            "global_ca_rmsd": global_rmsd,
            "functional_site_internal_rmsd": aligned_rmsd(a_func, b_func),
            "functional_site_pairwise_distance_rmse": pairwise_distance_rmse(
                a_func, b_func
            ),
            "chromophore_neighborhood_internal_rmsd": aligned_rmsd(
                a_pocket, b_pocket
            ),
            "chromophore_neighborhood_pairwise_distance_rmse": (
                pairwise_distance_rmse(a_pocket, b_pocket)
            ),
        })

    wt_self_df = pd.DataFrame(wt_self_rows)
    wt_self_df.to_csv(
        out_dir / "wt_self_functional_baseline.csv",
        index=False,
    )

    baseline = {
        col: median_or_nan(wt_self_df[col])
        for col in [
            "global_ca_rmsd",
            "functional_site_internal_rmsd",
            "functional_site_pairwise_distance_rmse",
            "chromophore_neighborhood_internal_rmsd",
            "chromophore_neighborhood_pairwise_distance_rmse",
        ]
    }

    rep_rows = []
    summary_rows = []

    for _, pair in pair_df.iterrows():
        cand_id = str(pair["candidate_id"])
        pair_id = str(pair["pair_id"])
        source_candidate_id = str(pair.get("source_candidate_id", ""))

        if cand_id not in seq_map:
            raise ValueError(
                f"{cand_id} missing from sequence_manifest"
            )
        if cand_id not in inv_by_id:
            raise RuntimeError(
                f"No ColabFold predictions found for {cand_id}"
            )

        candidate_seq = seq_map[cand_id]
        mutation_positions, mutation_labels = derive_mutations(
            wt_seq,
            candidate_seq,
        )
        core_mutations = sorted(
            set(mutation_positions) & set(functional_positions)
        )
        chrom_mutations = sorted(
            set(mutation_positions) & set(chrom_positions)
        )
        env_mutations = sorted(
            set(mutation_positions) & set(env_positions)
        )
        pocket_mutations = sorted(
            set(mutation_positions) & set(canonical_neighborhood)
        )

        cand_rep_keys = set(inv_by_id[cand_id].index.tolist())
        common = sorted(set(wt_rep_keys) & cand_rep_keys)
        if not common:
            raise RuntimeError(
                f"No matched WT/candidate replicate keys for {cand_id}"
            )

        candidate_rep_rows = []

        for key in common:
            wt_st = structure_map[f"WT|{key}"]
            cand_st = structure_map[f"{cand_id}|{key}"]

            wt_all = ca_matrix(wt_st, all_positions)
            cand_all = ca_matrix(cand_st, all_positions)

            R, t = kabsch_transform(wt_all, cand_all)
            cand_all_aligned = apply_transform(cand_all, R, t)
            global_rmsd = rmsd_direct(wt_all, cand_all_aligned)

            # Functional-site geometry.
            wt_func = ca_matrix(wt_st, functional_positions)
            cand_func = ca_matrix(cand_st, functional_positions)
            cand_func_global_aligned = cand_all_aligned[
                np.asarray(functional_positions) - 1
            ]

            func_global_aligned_rmsd = rmsd_direct(
                wt_func,
                cand_func_global_aligned,
            )
            func_internal_rmsd = aligned_rmsd(wt_func, cand_func)
            func_dist_rmse = pairwise_distance_rmse(wt_func, cand_func)

            # Chromophore-neighborhood geometry.
            wt_pocket = ca_matrix(wt_st, canonical_neighborhood)
            cand_pocket = ca_matrix(cand_st, canonical_neighborhood)
            cand_pocket_global_aligned = cand_all_aligned[
                np.asarray(canonical_neighborhood) - 1
            ]

            pocket_global_aligned_rmsd = rmsd_direct(
                wt_pocket,
                cand_pocket_global_aligned,
            )
            pocket_internal_rmsd = aligned_rmsd(
                wt_pocket,
                cand_pocket,
            )
            pocket_dist_rmse = pairwise_distance_rmse(
                wt_pocket,
                cand_pocket,
            )

            # Confidence on candidate functional regions.
            func_plddt = bfactor_vector(
                cand_st,
                functional_positions,
            )
            chrom_plddt = bfactor_vector(
                cand_st,
                chrom_positions,
            )
            pocket_plddt = bfactor_vector(
                cand_st,
                canonical_neighborhood,
            )

            # Mutation proximity to chromophore-forming motif.
            min_mut_to_chrom_wt = minimum_atom_distance(
                wt_st,
                mutation_positions,
                chrom_positions,
            )
            min_mut_to_chrom_candidate = minimum_atom_distance(
                cand_st,
                mutation_positions,
                chrom_positions,
            )

            row = {
                "pair_id": pair_id,
                "source_candidate_id": source_candidate_id,
                "candidate_id": cand_id,
                "replicate_key": key,
                "wt_candidate_global_ca_rmsd": global_rmsd,
                "functional_site_global_aligned_ca_rmsd": (
                    func_global_aligned_rmsd
                ),
                "functional_site_internal_rmsd": func_internal_rmsd,
                "functional_site_pairwise_distance_rmse": func_dist_rmse,
                "chromophore_neighborhood_global_aligned_ca_rmsd": (
                    pocket_global_aligned_rmsd
                ),
                "chromophore_neighborhood_internal_rmsd": (
                    pocket_internal_rmsd
                ),
                "chromophore_neighborhood_pairwise_distance_rmse": (
                    pocket_dist_rmse
                ),
                "candidate_functional_plddt_mean": float(
                    np.mean(func_plddt)
                ),
                "candidate_functional_plddt_min": float(
                    np.min(func_plddt)
                ),
                "candidate_chromophore_motif_plddt_mean": float(
                    np.mean(chrom_plddt)
                ),
                "candidate_neighborhood_plddt_mean": float(
                    np.mean(pocket_plddt)
                ),
                "min_mutation_to_chromophore_heavy_atom_distance_wt": (
                    min_mut_to_chrom_wt
                ),
                "min_mutation_to_chromophore_heavy_atom_distance_candidate": (
                    min_mut_to_chrom_candidate
                ),
            }
            rep_rows.append(row)
            candidate_rep_rows.append(row)

        cdf = pd.DataFrame(candidate_rep_rows)

        func_internal_med = median_or_nan(
            cdf["functional_site_internal_rmsd"]
        )
        func_dist_med = median_or_nan(
            cdf["functional_site_pairwise_distance_rmse"]
        )
        pocket_internal_med = median_or_nan(
            cdf["chromophore_neighborhood_internal_rmsd"]
        )
        pocket_dist_med = median_or_nan(
            cdf["chromophore_neighborhood_pairwise_distance_rmse"]
        )

        summary_rows.append({
            "pair_id": pair_id,
            "source_candidate_id": source_candidate_id,
            "candidate_id": cand_id,
            "matched_replicates": len(common),
            "n_wt_candidate_mutations": len(mutation_positions),
            "wt_candidate_mutation_positions": ";".join(
                map(str, mutation_positions)
            ),
            "wt_candidate_mutations": ";".join(mutation_labels),
            "mutates_chromophore_motif": bool(chrom_mutations),
            "chromophore_motif_mutations": ";".join(
                map(str, chrom_mutations)
            ),
            "mutates_key_environment_residue": bool(env_mutations),
            "key_environment_mutations": ";".join(
                map(str, env_mutations)
            ),
            "mutates_any_functional_residue": bool(core_mutations),
            "functional_residue_mutations": ";".join(
                map(str, core_mutations)
            ),
            "n_mutations_in_chromophore_neighborhood": len(
                pocket_mutations
            ),
            "mutations_in_chromophore_neighborhood": ";".join(
                map(str, pocket_mutations)
            ),
            "wt_candidate_global_ca_rmsd_median": median_or_nan(
                cdf["wt_candidate_global_ca_rmsd"]
            ),
            "functional_site_global_aligned_ca_rmsd_median": (
                median_or_nan(
                    cdf["functional_site_global_aligned_ca_rmsd"]
                )
            ),
            "functional_site_internal_rmsd_median": func_internal_med,
            "functional_site_pairwise_distance_rmse_median": func_dist_med,
            "chromophore_neighborhood_global_aligned_ca_rmsd_median": (
                median_or_nan(
                    cdf[
                        "chromophore_neighborhood_global_aligned_ca_rmsd"
                    ]
                )
            ),
            "chromophore_neighborhood_internal_rmsd_median": (
                pocket_internal_med
            ),
            "chromophore_neighborhood_pairwise_distance_rmse_median": (
                pocket_dist_med
            ),
            "wt_self_functional_site_internal_rmsd_median": baseline[
                "functional_site_internal_rmsd"
            ],
            "functional_site_internal_rmsd_excess_over_wt_self": (
                func_internal_med
                - baseline["functional_site_internal_rmsd"]
            ),
            "wt_self_functional_site_pairwise_distance_rmse_median": (
                baseline["functional_site_pairwise_distance_rmse"]
            ),
            "functional_site_distance_rmse_excess_over_wt_self": (
                func_dist_med
                - baseline["functional_site_pairwise_distance_rmse"]
            ),
            "wt_self_neighborhood_internal_rmsd_median": baseline[
                "chromophore_neighborhood_internal_rmsd"
            ],
            "neighborhood_internal_rmsd_excess_over_wt_self": (
                pocket_internal_med
                - baseline[
                    "chromophore_neighborhood_internal_rmsd"
                ]
            ),
            "wt_self_neighborhood_pairwise_distance_rmse_median": (
                baseline[
                    "chromophore_neighborhood_pairwise_distance_rmse"
                ]
            ),
            "neighborhood_distance_rmse_excess_over_wt_self": (
                pocket_dist_med
                - baseline[
                    "chromophore_neighborhood_pairwise_distance_rmse"
                ]
            ),
            "candidate_functional_plddt_mean": mean_or_nan(
                cdf["candidate_functional_plddt_mean"]
            ),
            "candidate_functional_plddt_min_mean": mean_or_nan(
                cdf["candidate_functional_plddt_min"]
            ),
            "candidate_chromophore_motif_plddt_mean": mean_or_nan(
                cdf["candidate_chromophore_motif_plddt_mean"]
            ),
            "candidate_neighborhood_plddt_mean": mean_or_nan(
                cdf["candidate_neighborhood_plddt_mean"]
            ),
            "min_mutation_to_chromophore_heavy_atom_distance_wt_median": (
                median_or_nan(
                    cdf[
                        "min_mutation_to_chromophore_heavy_atom_distance_wt"
                    ]
                )
            ),
            "min_mutation_to_chromophore_heavy_atom_distance_candidate_median": (
                median_or_nan(
                    cdf[
                        "min_mutation_to_chromophore_heavy_atom_distance_candidate"
                    ]
                )
            ),
            "functional_geometry_within_wt_self_variability": bool(
                func_internal_med
                <= baseline["functional_site_internal_rmsd"]
            ),
            "neighborhood_geometry_within_wt_self_variability": bool(
                pocket_internal_med
                <= baseline[
                    "chromophore_neighborhood_internal_rmsd"
                ]
            ),
        })

    rep_df = pd.DataFrame(rep_rows)
    summary_df = pd.DataFrame(summary_rows)

    rep_df.to_csv(
        out_dir / "replicate_functional_metrics.csv",
        index=False,
    )
    summary_df.to_csv(
        out_dir / "candidate_functional_summary.csv",
        index=False,
    )

    # Figures are descriptive only; no hard pass/fail threshold.
    try:
        import matplotlib.pyplot as plt

        labels = summary_df["candidate_id"].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(labels))
        ax.bar(
            x,
            summary_df["functional_site_internal_rmsd_median"],
            label="Candidate vs WT",
        )
        ax.axhline(
            baseline["functional_site_internal_rmsd"],
            linestyle="--",
            label="WT self-variability median",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Functional-site Cα RMSD (Å)")
        ax.set_title("GFP functional-site geometry vs WT self-variability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            fig_dir / "functional_site_geometry_vs_wt_self.png",
            dpi=180,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(
            labels,
            summary_df[
                "min_mutation_to_chromophore_heavy_atom_distance_wt_median"
            ],
        )
        ax.set_ylabel("Minimum heavy-atom distance (Å)")
        ax.set_title("Nearest candidate mutation to GFP chromophore-forming motif")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "mutation_to_chromophore_distance.png",
            dpi=180,
        )
        plt.close(fig)
    except Exception as e:
        print(f"[warning] Figure generation skipped: {e}")

    summary_obj = {
        "n_candidates": int(len(summary_df)),
        "n_wt_replicates": int(len(wt_rep_keys)),
        "sequence_length": int(L),
        "chromophore_residues": chrom_positions,
        "environment_residues": env_positions,
        "functional_residues": functional_positions,
        "neighborhood_radius_angstrom": float(
            args.neighborhood_radius
        ),
        "canonical_chromophore_neighborhood": canonical_neighborhood,
        "canonical_chromophore_neighborhood_size": int(
            len(canonical_neighborhood)
        ),
        "wt_self_baseline_medians": baseline,
        "metric_definitions": {
            "global_ca_rmsd": (
                "WT-candidate C-alpha RMSD after global Kabsch alignment"
            ),
            "functional_site_internal_rmsd": (
                "C-alpha RMSD of predefined GFP functional residues after "
                "local Kabsch alignment; measures internal site geometry"
            ),
            "functional_site_pairwise_distance_rmse": (
                "RMSE between all pairwise C-alpha distances among "
                "predefined functional residues; alignment invariant"
            ),
            "chromophore_neighborhood": (
                "WT-derived majority-vote set of residues with any heavy "
                "atom within the configured radius of residues 65-67"
            ),
            "min_mutation_to_chromophore_distance": (
                "minimum heavy-atom distance between any WT-to-candidate "
                "mutated residue and residues 65-67"
            ),
            "plddt": (
                "AlphaFold/ColabFold C-alpha B-factor confidence on a "
                "0-100 scale"
            ),
        },
        "interpretation": (
            "This analysis assesses preservation of the predicted GFP "
            "chromophore-forming motif environment and key functional-site "
            "geometry. It does not model chromophore maturation, quantum "
            "yield, or experimentally demonstrate fluorescence."
        ),
        "outputs": {
            "inventory": str(out_dir / "structure_inventory.csv"),
            "wt_self_baseline": str(
                out_dir / "wt_self_functional_baseline.csv"
            ),
            "replicate_metrics": str(
                out_dir / "replicate_functional_metrics.csv"
            ),
            "candidate_summary": str(
                out_dir / "candidate_functional_summary.csv"
            ),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            summary_obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 132)
    print("Step 10-3: GFP Functional-Site Preservation Analysis")
    print("=" * 132)
    print(f"WT replicates                 : {len(wt_rep_keys)}")
    print(f"Candidate pairs               : {len(summary_df)}")
    print(f"GFP sequence length           : {L}")
    print(f"Chromophore-forming residues  : {chrom_positions}")
    print(f"Key environment residues      : {env_positions}")
    print(
        f"WT-derived {args.neighborhood_radius:.1f} Å neighborhood : "
        f"{len(canonical_neighborhood)} residues"
    )
    print(f"Output                        : {out_dir}")
    print()

    cols = [
        "candidate_id",
        "n_wt_candidate_mutations",
        "mutates_chromophore_motif",
        "mutates_key_environment_residue",
        "n_mutations_in_chromophore_neighborhood",
        "wt_candidate_global_ca_rmsd_median",
        "functional_site_internal_rmsd_median",
        "wt_self_functional_site_internal_rmsd_median",
        "functional_site_internal_rmsd_excess_over_wt_self",
        "chromophore_neighborhood_internal_rmsd_median",
        "wt_self_neighborhood_internal_rmsd_median",
        "neighborhood_internal_rmsd_excess_over_wt_self",
        "min_mutation_to_chromophore_heavy_atom_distance_wt_median",
        "candidate_functional_plddt_mean",
    ]
    print(summary_df[cols].to_string(index=False))
    print()
    print("Shape semantics")
    print("-" * 132)
    print(f"Full Cα coordinates            : [{L}, 3]")
    print(
        f"Functional-site Cα coordinates : "
        f"[{len(functional_positions)}, 3]"
    )
    print(
        f"Chromophore-neighborhood Cα    : "
        f"[{len(canonical_neighborhood)}, 3]"
    )
    print("Per matched WT/candidate pair  : coordinate sets -> scalar geometry metrics")
    print("Per candidate                  : 3 replicate metrics -> median/mean summary")
    print()
    print(
        "Interpretation: structural functional plausibility only; "
        "not experimental fluorescence validation."
    )


if __name__ == "__main__":
    main()
