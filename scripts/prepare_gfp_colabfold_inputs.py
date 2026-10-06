#!/usr/bin/env python3
"""Prepare GFP WT / ESM3 seed / candidate sequences for ColabFold.

Supports the actual project layout:

1) Final shortlist
   results/gfp/esm3_shortlist/esm3_candidate_shortlist.csv
   - candidate_id
   - seed_mutant
   - generated_sequence
   - ...

2) Selected seed variants
   results/gfp/esm3_generation/selected_seed_variants.csv
   - mutant
   - mutated_sequence
   - ...

The seed full sequence is recovered by joining:

    shortlist.seed_mutant == selected_seed_variants.mutant

The script also supports a shortlist that already contains seed_sequence.

No DMS_score or experimental multi-mutant label is loaded.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

SEED_SEQUENCE_ALIASES = [
    "seed_sequence",
    "parent_sequence",
    "source_sequence",
    "seed_seq",
]

CANDIDATE_SEQUENCE_ALIASES = [
    "generated_sequence",
    "candidate_sequence",
    "generated_seq",
    "candidate_seq",
]

SOURCE_CANDIDATE_ID_ALIASES = [
    "candidate_id",
    "generated_candidate_id",
    "generated_id",
    "sequence_id",
    "id",
]

SEED_MUTANT_ALIASES = [
    "seed_mutant",
    "parent_mutant",
    "source_mutant",
]

SEED_TABLE_MUTANT_ALIASES = [
    "mutant",
    "seed_mutant",
]

SEED_TABLE_SEQUENCE_ALIASES = [
    "mutated_sequence",
    "seed_sequence",
    "sequence",
]


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalized_column_map(df: pd.DataFrame) -> dict[str, str]:
    return {normalize_column_name(c): c for c in df.columns}


def resolve_column(
    df: pd.DataFrame,
    aliases: list[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    cmap = normalized_column_map(df)
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in cmap:
            return cmap[key]
    if required:
        raise ValueError(
            f"Could not identify {label} column. "
            f"Expected one of {aliases}; actual columns={list(df.columns)}"
        )
    return None


def clean_sequence(value, label: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{label}: sequence is missing")
    seq = re.sub(r"\s+", "", str(value)).upper()
    if not seq:
        raise ValueError(f"{label}: sequence is empty")
    invalid = sorted(set(seq) - VALID_AA)
    if invalid:
        raise ValueError(
            f"{label}: non-canonical amino-acid symbols found: {invalid}"
        )
    return seq


def read_single_fasta(path: Path) -> tuple[str, str]:
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip() or "sequence"
            chunks = []
        else:
            if header is None:
                raise ValueError(f"{path}: FASTA sequence appears before a header")
            chunks.append(line)

    if header is not None:
        records.append((header, "".join(chunks)))

    if len(records) != 1:
        raise ValueError(
            f"{path}: expected exactly one WT FASTA record, found {len(records)}"
        )

    name, seq = records[0]
    return name, clean_sequence(seq, f"WT FASTA {path}")


def find_wt(project_root: Path) -> Path:
    preferred = [
        project_root / "data/processed/gfp_wt.fasta",
        project_root / "data/processed/gfp/gfp_wt.fasta",
    ]
    for p in preferred:
        if p.is_file():
            return p.resolve()

    matches = sorted(
        p.resolve()
        for p in project_root.glob("data/processed/**/gfp_wt.fasta")
        if p.is_file()
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-detect gfp_wt.fasta under data/processed/. "
            "Pass --wt-fasta explicitly."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple gfp_wt.fasta files found. Pass --wt-fasta explicitly:\n"
            + "\n".join(f"  - {p}" for p in matches)
        )
    return matches[0]


def find_shortlist(project_root: Path) -> Path:
    preferred = (
        project_root
        / "results/gfp/esm3_shortlist/esm3_candidate_shortlist.csv"
    )
    if preferred.is_file():
        return preferred.resolve()

    matches = sorted(
        p.resolve()
        for p in project_root.glob("results/gfp/**/*.csv")
        if p.is_file() and "shortlist" in p.name.lower()
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-detect the final GFP shortlist CSV. "
            "Pass --shortlist-csv explicitly."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple shortlist-like CSV files found. "
            "Pass --shortlist-csv explicitly:\n"
            + "\n".join(f"  - {p}" for p in matches)
        )
    return matches[0]


def find_seed_variants(project_root: Path) -> Path:
    preferred = (
        project_root
        / "results/gfp/esm3_generation/selected_seed_variants.csv"
    )
    if preferred.is_file():
        return preferred.resolve()

    matches = sorted(
        p.resolve()
        for p in project_root.glob("results/gfp/**/*.csv")
        if p.is_file() and p.name == "selected_seed_variants.csv"
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-detect selected_seed_variants.csv. "
            "Pass --seed-variants-csv explicitly."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple selected_seed_variants.csv files found. "
            "Pass --seed-variants-csv explicitly:\n"
            + "\n".join(f"  - {p}" for p in matches)
        )
    return matches[0]


def changed_positions(seed: str, candidate: str) -> list[int]:
    if len(seed) != len(candidate):
        raise ValueError(
            f"Seed/candidate length mismatch: {len(seed)} vs {len(candidate)}"
        )
    return [
        i + 1
        for i, (a, b) in enumerate(zip(seed, candidate))
        if a != b
    ]


def mutation_labels(seed: str, candidate: str, positions: list[int]) -> list[str]:
    return [f"{seed[p-1]}{p}{candidate[p-1]}" for p in positions]


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for seq_id, seq in records:
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def recover_seed_sequences(
    shortlist: pd.DataFrame,
    shortlist_path: Path,
    project_root: Path,
    seed_variants_path: Optional[Path],
) -> tuple[pd.Series, str, Optional[Path], Optional[str]]:
    """Return seed sequence series aligned to shortlist rows."""
    direct_col = resolve_column(
        shortlist,
        SEED_SEQUENCE_ALIASES,
        "seed sequence",
        required=False,
    )
    if direct_col is not None:
        return shortlist[direct_col], "direct_shortlist_column", None, direct_col

    seed_mutant_col = resolve_column(
        shortlist,
        SEED_MUTANT_ALIASES,
        "seed mutant identifier",
        required=False,
    )
    if seed_mutant_col is None:
        raise ValueError(
            f"{shortlist_path} has neither a seed sequence column "
            f"{SEED_SEQUENCE_ALIASES} nor seed mutant identifier "
            f"{SEED_MUTANT_ALIASES}."
        )

    seed_variants_path = (
        seed_variants_path.resolve()
        if seed_variants_path is not None
        else find_seed_variants(project_root)
    )
    seed_df = pd.read_csv(seed_variants_path, dtype=str)

    seed_key_col = resolve_column(
        seed_df,
        SEED_TABLE_MUTANT_ALIASES,
        "seed-table mutant identifier",
    )
    seed_seq_col = resolve_column(
        seed_df,
        SEED_TABLE_SEQUENCE_ALIASES,
        "seed-table full sequence",
    )

    if seed_df[seed_key_col].duplicated().any():
        dupes = (
            seed_df.loc[seed_df[seed_key_col].duplicated(keep=False), seed_key_col]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{seed_variants_path}: mutant identifiers are not unique: {dupes[:10]}"
        )

    mapping = dict(zip(seed_df[seed_key_col], seed_df[seed_seq_col]))
    recovered = shortlist[seed_mutant_col].map(mapping)

    missing_mask = recovered.isna()
    if missing_mask.any():
        missing = shortlist.loc[missing_mask, seed_mutant_col].tolist()
        raise ValueError(
            "Failed to recover full sequence for some shortlist seeds. "
            f"Missing seed_mutant values in {seed_variants_path}: {missing}"
        )

    return (
        recovered,
        "joined_selected_seed_variants",
        seed_variants_path,
        seed_mutant_col,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare GFP WT / ESM3 seed / candidate sequences "
            "for independent ColabFold structure prediction."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root used for auto-detection.",
    )
    parser.add_argument(
        "--wt-fasta",
        type=Path,
        default=None,
        help="WT GFP FASTA. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--shortlist-csv",
        type=Path,
        default=None,
        help=(
            "Final ESM3 shortlist CSV. Defaults to "
            "results/gfp/esm3_shortlist/esm3_candidate_shortlist.csv"
        ),
    )
    parser.add_argument(
        "--seed-variants-csv",
        type=Path,
        default=None,
        help=(
            "Selected seed variants CSV used when the shortlist does not "
            "contain full seed_sequence. Defaults to "
            "results/gfp/esm3_generation/selected_seed_variants.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gfp_colabfold"),
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    wt_path = args.wt_fasta.resolve() if args.wt_fasta else find_wt(project_root)
    shortlist_path = (
        args.shortlist_csv.resolve()
        if args.shortlist_csv
        else find_shortlist(project_root)
    )
    seed_variants_path = (
        args.seed_variants_csv.resolve()
        if args.seed_variants_csv
        else None
    )
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _, wt_seq = read_single_fasta(wt_path)

    # dtype=str is important: candidate_id like 00033 must remain 00033.
    shortlist = pd.read_csv(shortlist_path, dtype=str)
    if shortlist.empty:
        raise ValueError(f"{shortlist_path}: shortlist CSV is empty")

    candidate_seq_col = resolve_column(
        shortlist,
        CANDIDATE_SEQUENCE_ALIASES,
        "candidate/generated full sequence",
    )
    source_id_col = resolve_column(
        shortlist,
        SOURCE_CANDIDATE_ID_ALIASES,
        "candidate ID",
        required=False,
    )
    seed_mutant_col = resolve_column(
        shortlist,
        SEED_MUTANT_ALIASES,
        "seed mutant identifier",
        required=False,
    )

    seed_series, seed_source, joined_seed_path, direct_seed_info = (
        recover_seed_sequences(
            shortlist=shortlist,
            shortlist_path=shortlist_path,
            project_root=project_root,
            seed_variants_path=seed_variants_path,
        )
    )

    pair_rows = []
    sequence_rows = [{
        "sequence_id": "WT",
        "role": "wt",
        "pair_id": "",
        "source_candidate_id": "",
        "sequence_length": len(wt_seq),
        "sequence": wt_seq,
    }]
    fasta_records = [("WT", wt_seq)]

    for ordinal, (row_idx, row) in enumerate(shortlist.iterrows(), start=1):
        pair_id = f"pair_{ordinal:02d}"
        seed_id = f"seed_{ordinal:02d}"
        candidate_id = f"candidate_{ordinal:02d}"

        source_candidate_id = (
            str(row[source_id_col])
            if source_id_col is not None and pd.notna(row[source_id_col])
            else str(row_idx)
        )
        seed_mutant = (
            str(row[seed_mutant_col])
            if seed_mutant_col is not None and pd.notna(row[seed_mutant_col])
            else ""
        )

        seed_seq = clean_sequence(seed_series.loc[row_idx], f"{pair_id} seed")
        candidate_seq = clean_sequence(
            row[candidate_seq_col],
            f"{pair_id} candidate",
        )

        if len(seed_seq) != len(wt_seq):
            raise ValueError(
                f"{pair_id}: seed length {len(seed_seq)} != WT length {len(wt_seq)}"
            )
        if len(candidate_seq) != len(wt_seq):
            raise ValueError(
                f"{pair_id}: candidate length {len(candidate_seq)} "
                f"!= WT length {len(wt_seq)}"
            )

        positions = changed_positions(seed_seq, candidate_seq)
        if not positions:
            raise ValueError(
                f"{pair_id}: seed and candidate are identical. "
                "Expected at least one ESM3 design substitution."
            )

        mutations = mutation_labels(seed_seq, candidate_seq, positions)

        # Preserve original design_position for provenance when available,
        # but derive positions independently from full sequences.
        original_design_position = (
            str(row["design_position"])
            if "design_position" in shortlist.columns
            and pd.notna(row["design_position"])
            else ""
        )

        if original_design_position:
            expected = ";".join(map(str, positions))
            if original_design_position != expected:
                raise ValueError(
                    f"{pair_id}: shortlist design_position="
                    f"{original_design_position} but sequence-derived "
                    f"position(s)={expected}"
                )

        pair_rows.append({
            "pair_id": pair_id,
            "source_candidate_id": source_candidate_id,
            "seed_mutant": seed_mutant,
            "seed_id": seed_id,
            "candidate_id": candidate_id,
            "design_positions": ";".join(map(str, positions)),
            "n_design_positions": str(len(positions)),
            "seed_to_candidate_mutations": ";".join(mutations),
            "sequence_length": str(len(wt_seq)),
            "seed_sequence": seed_seq,
            "candidate_sequence": candidate_seq,
        })

        sequence_rows.extend([
            {
                "sequence_id": seed_id,
                "role": "seed",
                "pair_id": pair_id,
                "source_candidate_id": source_candidate_id,
                "sequence_length": len(seed_seq),
                "sequence": seed_seq,
            },
            {
                "sequence_id": candidate_id,
                "role": "candidate",
                "pair_id": pair_id,
                "source_candidate_id": source_candidate_id,
                "sequence_length": len(candidate_seq),
                "sequence": candidate_seq,
            },
        ])
        fasta_records.extend([
            (seed_id, seed_seq),
            (candidate_id, candidate_seq),
        ])

    pair_df = pd.DataFrame(pair_rows)
    seq_df = pd.DataFrame(sequence_rows)

    fasta_path = out_dir / "colabfold_input.fasta"
    pair_path = out_dir / "pair_manifest.csv"
    seq_path = out_dir / "sequence_manifest.csv"
    summary_path = out_dir / "summary.json"

    write_fasta(fasta_records, fasta_path)
    pair_df.to_csv(pair_path, index=False)
    seq_df.to_csv(seq_path, index=False)

    summary = {
        "wt_fasta": str(wt_path),
        "shortlist_csv": str(shortlist_path),
        "seed_variants_csv": (
            str(joined_seed_path) if joined_seed_path is not None else None
        ),
        "seed_sequence_source": seed_source,
        "wt_length": len(wt_seq),
        "n_pairs": int(len(pair_df)),
        "n_fasta_records": int(len(seq_df)),
        "candidate_sequence_column": candidate_seq_col,
        "seed_mutant_column": seed_mutant_col,
        "uses_experimental_labels": False,
        "design_position_definition": (
            "1-based positions derived directly by comparing "
            "seed_sequence and candidate_sequence"
        ),
        "outputs": {
            "colabfold_input_fasta": str(fasta_path),
            "pair_manifest": str(pair_path),
            "sequence_manifest": str(seq_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 118)
    print("Step 10-2-1: Prepare GFP ColabFold inputs")
    print("=" * 118)
    print(f"WT FASTA             : {wt_path}")
    print(f"Shortlist CSV        : {shortlist_path}")
    if joined_seed_path is not None:
        print(f"Seed variants CSV    : {joined_seed_path}")
    print(f"Seed sequence source : {seed_source}")
    print(f"WT length            : {len(wt_seq)} aa")
    print(f"Candidate pairs      : {len(pair_df)}")
    print(f"FASTA records        : {len(seq_df)}")
    print(f"Output directory     : {out_dir}")
    print()
    print("Pair mapping")
    print("-" * 118)
    show_cols = [
        "pair_id",
        "source_candidate_id",
        "seed_mutant",
        "seed_id",
        "candidate_id",
        "design_positions",
        "seed_to_candidate_mutations",
    ]
    print(pair_df[show_cols].to_string(index=False))
    print()
    print("Shape semantics")
    print("-" * 118)
    print(f"Each GFP sequence       : [{len(wt_seq)} aa]")
    print(f"ColabFold FASTA records : [{len(seq_df)} sequences]")
    print(f"Expected Cα/model       : [{len(wt_seq)}, 3]")
    print()
    print("No DMS_score or experimental multi-mutant label was loaded.")


if __name__ == "__main__":
    main()
