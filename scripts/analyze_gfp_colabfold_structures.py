#!/usr/bin/env python3
"""Analyze ColabFold/AlphaFold2 structures for GFP seed/candidate pairs.

Outputs matched-replicate RMSDs, same-sequence self-RMSDs, confidence metrics,
and a per-candidate summary. Structures are aligned with Kabsch using all Cα atoms.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_pdb_ca(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, str]]]:
    coords = []
    bfac = []
    residues = []
    seen = set()
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        alt = line[16:17]
        if atom != "CA" or alt not in (" ", "A"):
            continue
        chain = line[21:22].strip() or "A"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        icode = line[26:27].strip()
        key = (chain, resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            b = float(line[60:66])
        except ValueError as e:
            raise ValueError(f"Could not parse PDB coordinates in {path}: {e}")
        coords.append([x, y, z])
        bfac.append(b)
        residues.append(key)
    if not coords:
        raise ValueError(f"No Cα atoms found in {path}")
    return np.asarray(coords, dtype=float), np.asarray(bfac, dtype=float), residues


def kabsch_align(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    mobile = np.asarray(mobile, dtype=float)
    if reference.shape != mobile.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError(f"Coordinate shape mismatch: reference={reference.shape}, mobile={mobile.shape}")
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
    return (R @ X.T).T + ref_center


def rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    aligned = kabsch_align(reference, mobile)
    return float(np.sqrt(np.mean(np.sum((reference - aligned) ** 2, axis=1))))


def parse_tokens(filename: str) -> tuple[int | None, int | None, int | None]:
    rank_m = re.search(r"rank[_-]?(\d+)", filename)
    model_m = re.search(r"model[_-]?(\d+)", filename)
    seed_matches = re.findall(r"seed[_-]?(\d+)", filename)
    rank = int(rank_m.group(1)) if rank_m else None
    model = int(model_m.group(1)) if model_m else None
    seed = int(seed_matches[-1]) if seed_matches else None
    return rank, model, seed


def sequence_id_from_filename(filename: str, sequence_ids: list[str]) -> str | None:
    for seq_id in sorted(sequence_ids, key=len, reverse=True):
        if filename == seq_id or filename.startswith(seq_id + "_"):
            return seq_id
    return None


def replicate_key(rank: int | None, model: int | None, seed: int | None) -> str:
    if model is not None and seed is not None:
        return f"model_{model}_seed_{seed}"
    if seed is not None:
        return f"seed_{seed}"
    if rank is not None:
        return f"rank_{rank}"
    return "replicate_unknown"


def read_score_jsons(pred_dir: Path) -> list[tuple[Path, dict]]:
    out = []
    for path in pred_dir.glob("**/*.json"):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append((path, obj))
    return out


def scalar_from_obj(obj: dict, aliases: list[str]) -> float:
    for key in aliases:
        if key in obj:
            val = obj[key]
            if np.isscalar(val):
                try:
                    return float(val)
                except Exception:
                    pass
    return float("nan")


def find_score_for_prediction(
    seq_id: str,
    rank: int | None,
    model: int | None,
    seed: int | None,
    score_jsons: list[tuple[Path, dict]],
) -> dict | None:
    matches = []
    for path, obj in score_jsons:
        name = path.name
        if not (name == seq_id or name.startswith(seq_id + "_")):
            continue
        r2, m2, s2 = parse_tokens(name)
        score = 0
        if rank is not None and r2 == rank:
            score += 1
        if model is not None and m2 == model:
            score += 2
        if seed is not None and s2 == seed:
            score += 4
        matches.append((score, path, obj))
    if not matches:
        return None
    matches.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return matches[0][2]


def parse_positions(value) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(x) for x in re.split(r"[;,\s]+", text) if x]


def finite_or_none(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def pairwise_self_rmsd(structures: pd.DataFrame, coord_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for seq_id, group in structures.groupby("sequence_id"):
        records = group.to_dict("records")
        for a, b in itertools.combinations(records, 2):
            ca = coord_map[a["structure_path"]]
            cb = coord_map[b["structure_path"]]
            if ca.shape != cb.shape:
                continue
            rows.append({
                "sequence_id": seq_id,
                "replicate_a": a["replicate_key"],
                "replicate_b": b["replicate_key"],
                "rmsd": rmsd(ca, cb),
                "structure_a": a["structure_path"],
                "structure_b": b["structure_path"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", type=Path, required=True)
    p.add_argument("--pair-manifest", type=Path, required=True)
    p.add_argument("--sequence-manifest", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gfp/colabfold_structural_validation"),
    )
    args = p.parse_args()

    pred_dir = args.predictions_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_csv(args.pair_manifest)
    seq_df = pd.read_csv(args.sequence_manifest)
    sequence_ids = seq_df["sequence_id"].astype(str).tolist()
    expected_lengths = dict(zip(seq_df["sequence_id"].astype(str), seq_df["sequence_length"].astype(int)))

    pdbs = sorted(pred_dir.glob("**/*.pdb"))
    if not pdbs:
        raise FileNotFoundError(
            f"No PDB files found under {pred_dir}. This analyzer expects ColabFold PDB output."
        )

    score_jsons = read_score_jsons(pred_dir)
    coord_map: dict[str, np.ndarray] = {}
    inventory_rows = []

    for pdb in pdbs:
        seq_id = sequence_id_from_filename(pdb.name, sequence_ids)
        if seq_id is None:
            continue
        coords, bfac, residue_keys = parse_pdb_ca(pdb)
        expected = expected_lengths[seq_id]
        if len(coords) != expected:
            raise ValueError(
                f"{pdb.name}: Cα count {len(coords)} does not match expected length {expected}"
            )
        rank, model, seed = parse_tokens(pdb.name)
        key = replicate_key(rank, model, seed)
        score_obj = find_score_for_prediction(seq_id, rank, model, seed, score_jsons)
        ptm = float("nan")
        json_plddt = float("nan")
        if score_obj is not None:
            ptm = scalar_from_obj(score_obj, ["ptm", "ptm_score", "predicted_tm_score"])
            plddt_val = score_obj.get("plddt", score_obj.get("plddts"))
            if isinstance(plddt_val, list) and plddt_val:
                json_plddt = float(np.mean(np.asarray(plddt_val, dtype=float)))
            elif np.isscalar(plddt_val) and plddt_val is not None:
                try:
                    json_plddt = float(plddt_val)
                except Exception:
                    pass
        mean_plddt = float(np.mean(bfac))
        if not np.isfinite(mean_plddt) and np.isfinite(json_plddt):
            mean_plddt = json_plddt
        coord_map[str(pdb)] = coords
        inventory_rows.append({
            "sequence_id": seq_id,
            "structure_path": str(pdb),
            "rank": rank,
            "model": model,
            "seed": seed,
            "replicate_key": key,
            "n_ca": len(coords),
            "mean_plddt": mean_plddt,
            "ptm": ptm,
        })

    inventory = pd.DataFrame(inventory_rows)
    if inventory.empty:
        raise RuntimeError("PDB files were found, but none matched sequence IDs in sequence_manifest.csv")
    inventory.to_csv(out_dir / "structure_inventory.csv", index=False)

    self_df = pairwise_self_rmsd(inventory, coord_map)
    self_df.to_csv(out_dir / "self_rmsd_metrics.csv", index=False)

    self_med = {}
    if not self_df.empty:
        self_med = self_df.groupby("sequence_id")["rmsd"].median().to_dict()
    wt_self = float(self_med.get("WT", float("nan")))

    inv_by_id = {
        sid: grp.set_index("replicate_key", drop=False)
        for sid, grp in inventory.groupby("sequence_id")
    }

    rep_rows = []
    summary_rows = []

    for _, pair in pair_df.iterrows():
        pair_id = str(pair["pair_id"])
        seed_id = str(pair["seed_id"])
        cand_id = str(pair["candidate_id"])
        positions = parse_positions(pair.get("design_positions", ""))

        if seed_id not in inv_by_id or cand_id not in inv_by_id:
            raise RuntimeError(f"Missing ColabFold predictions for {pair_id}: {seed_id}, {cand_id}")

        seed_g = inv_by_id[seed_id]
        cand_g = inv_by_id[cand_id]
        common = sorted(set(seed_g.index) & set(cand_g.index))
        if not common:
            raise RuntimeError(f"No matched replicate keys for {pair_id}")

        pair_rmsds = []
        local_disps = []
        for key in common:
            srow = seed_g.loc[key]
            crow = cand_g.loc[key]
            # Defensive handling in case replicate_key is duplicated.
            if isinstance(srow, pd.DataFrame):
                srow = srow.iloc[0]
            if isinstance(crow, pd.DataFrame):
                crow = crow.iloc[0]
            s_ca = coord_map[srow["structure_path"]]
            c_ca = coord_map[crow["structure_path"]]
            if s_ca.shape != c_ca.shape:
                raise ValueError(f"Shape mismatch for {pair_id}/{key}: {s_ca.shape} vs {c_ca.shape}")
            aligned_c = kabsch_align(s_ca, c_ca)
            sc_rmsd = float(np.sqrt(np.mean(np.sum((s_ca - aligned_c) ** 2, axis=1))))
            pair_rmsds.append(sc_rmsd)

            valid_idx = [pos - 1 for pos in positions if 1 <= pos <= len(s_ca)]
            local_disp = float("nan")
            if valid_idx:
                local_disp = float(np.mean(np.linalg.norm(s_ca[valid_idx] - aligned_c[valid_idx], axis=1)))
                local_disps.append(local_disp)

            wt_seed = float("nan")
            wt_cand = float("nan")
            if "WT" in inv_by_id and key in inv_by_id["WT"].index:
                wrow = inv_by_id["WT"].loc[key]
                if isinstance(wrow, pd.DataFrame):
                    wrow = wrow.iloc[0]
                w_ca = coord_map[wrow["structure_path"]]
                if w_ca.shape == s_ca.shape:
                    wt_seed = rmsd(w_ca, s_ca)
                    wt_cand = rmsd(w_ca, c_ca)

            rep_rows.append({
                "pair_id": pair_id,
                "seed_id": seed_id,
                "candidate_id": cand_id,
                "replicate_key": key,
                "seed_candidate_rmsd": sc_rmsd,
                "wt_seed_rmsd": wt_seed,
                "wt_candidate_rmsd": wt_cand,
                "design_position_displacement": local_disp,
                "seed_mean_plddt": float(srow["mean_plddt"]),
                "candidate_mean_plddt": float(crow["mean_plddt"]),
                "seed_ptm": float(srow["ptm"]),
                "candidate_ptm": float(crow["ptm"]),
            })

        seed_self = float(self_med.get(seed_id, float("nan")))
        cand_self = float(self_med.get(cand_id, float("nan")))
        baseline_vals = [x for x in (wt_self, seed_self, cand_self) if np.isfinite(x)]
        baseline = max(baseline_vals) if baseline_vals else float("nan")
        pair_med = float(np.median(pair_rmsds))
        excess = pair_med - baseline if np.isfinite(baseline) else float("nan")

        cand_inv = inventory[inventory["sequence_id"] == cand_id]
        seed_inv = inventory[inventory["sequence_id"] == seed_id]
        summary_rows.append({
            "pair_id": pair_id,
            "source_candidate_id": pair.get("source_candidate_id", ""),
            "seed_id": seed_id,
            "candidate_id": cand_id,
            "design_positions": pair.get("design_positions", ""),
            "matched_replicates": len(common),
            "candidate_seed_rmsd_mean": float(np.mean(pair_rmsds)),
            "candidate_seed_rmsd_std": float(np.std(pair_rmsds, ddof=1)) if len(pair_rmsds) > 1 else 0.0,
            "candidate_seed_rmsd_median": pair_med,
            "wt_self_rmsd_median": wt_self,
            "seed_self_rmsd_median": seed_self,
            "candidate_self_rmsd_median": cand_self,
            "self_variability_baseline": baseline,
            "excess_over_self_baseline": excess,
            "design_position_displacement_mean": float(np.mean(local_disps)) if local_disps else float("nan"),
            "seed_mean_plddt": float(seed_inv["mean_plddt"].mean()),
            "candidate_mean_plddt": float(cand_inv["mean_plddt"].mean()),
            "seed_mean_ptm": float(seed_inv["ptm"].mean()),
            "candidate_mean_ptm": float(cand_inv["ptm"].mean()),
            "within_self_variability": bool(pair_med <= baseline) if np.isfinite(baseline) else None,
        })

    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(out_dir / "replicate_metrics.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "structural_metrics.csv", index=False)

    # Simple figures; no hard filtering threshold is imposed.
    try:
        import matplotlib.pyplot as plt

        labels = summary_df["candidate_id"].astype(str).tolist()
        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, summary_df["candidate_seed_rmsd_median"], width, label="Candidate–seed RMSD")
        ax.bar(x + width / 2, summary_df["self_variability_baseline"], width, label="Self-variability baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Cα RMSD (Å)")
        ax.set_title("ColabFold structural difference vs self-variability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "candidate_seed_rmsd_vs_self_variability.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, summary_df["candidate_mean_plddt"])
        ax.set_ylabel("Mean pLDDT (0–100)")
        ax.set_title("Candidate ColabFold confidence")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(fig_dir / "candidate_mean_plddt.png", dpi=180)
        plt.close(fig)
    except Exception as e:
        print(f"[warning] Figure generation skipped: {e}")

    n_with_baseline = int(summary_df["self_variability_baseline"].notna().sum())
    n_within = int((summary_df["within_self_variability"] == True).sum())  # noqa: E712
    summary_obj = {
        "n_pairs": int(len(summary_df)),
        "n_structures": int(len(inventory)),
        "n_pairs_with_self_variability_baseline": n_with_baseline,
        "n_pairs_within_self_variability": n_within,
        "metric_definition": {
            "candidate_seed_rmsd": "C-alpha RMSD after global Kabsch alignment",
            "self_variability_baseline": "max(median WT self-RMSD, median seed self-RMSD, median candidate self-RMSD)",
            "excess_over_self_baseline": "median candidate-seed RMSD minus self-variability baseline",
            "design_position_displacement": "mean C-alpha displacement at seed-to-candidate changed positions after global alignment",
            "plddt_scale": "ColabFold/AlphaFold PDB B-factor confidence, 0-100",
        },
        "interpretation": (
            "This is a cross-model structural consistency / plausibility analysis. "
            "It is not experimental validation and does not impose a hard RMSD rejection threshold."
        ),
        "outputs": {
            "inventory": str(out_dir / "structure_inventory.csv"),
            "replicate_metrics": str(out_dir / "replicate_metrics.csv"),
            "self_rmsd_metrics": str(out_dir / "self_rmsd_metrics.csv"),
            "structural_metrics": str(out_dir / "structural_metrics.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2, allow_nan=False), encoding="utf-8")

    print("=" * 120)
    print("Step 10-2-2: ColabFold / AlphaFold2 cross-model structural analysis")
    print("=" * 120)
    print(f"Prediction PDBs parsed : {len(inventory)}")
    print(f"Candidate pairs        : {len(summary_df)}")
    print(f"Output                 : {out_dir}")
    print()
    cols = [
        "candidate_id",
        "matched_replicates",
        "candidate_seed_rmsd_median",
        "self_variability_baseline",
        "excess_over_self_baseline",
        "design_position_displacement_mean",
        "candidate_mean_plddt",
        "candidate_mean_ptm",
        "within_self_variability",
    ]
    print(summary_df[cols].to_string(index=False))
    print()
    print("Shape semantics:")
    print("  per structure: Cα coordinates [L, 3]")
    print("  per matched seed/candidate replicate: two [L,3] arrays -> scalar RMSD")
    print("  per candidate: replicate scalars -> mean/median/std summary")


if __name__ == "__main__":
    main()
