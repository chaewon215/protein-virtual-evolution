from pathlib import Path
import re

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

REFERENCE_PATH = RAW_ROOT / "reference_files" / "DMS_substitutions.csv"

ASSAY_ID = "BLAT_ECOLX_Stiffler_2015"


# ============================================================
# Utility functions
# ============================================================

def parse_single_mutation(mutation: str):
    """
    Example:
        H24C -> ("H", 24, "C")
    """
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation)

    if match is None:
        raise ValueError(f"Invalid single mutation: {mutation}")

    wt_aa, position, mut_aa = match.groups()

    return wt_aa, int(position), mut_aa


def find_cv_file():
    candidates = list(
        RAW_ROOT.rglob(f"{ASSAY_ID}.csv")
    )

    candidates = [
        path
        for path in candidates
        if "cv_folds_singles_substitutions" in str(path)
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one CV file, found {len(candidates)}:\n"
            + "\n".join(map(str, candidates))
        )

    return candidates[0]


# ============================================================
# Load assay metadata
# ============================================================

reference_df = pd.read_csv(REFERENCE_PATH)

metadata = reference_df.loc[
    reference_df["DMS_id"] == ASSAY_ID
]

if len(metadata) != 1:
    raise RuntimeError(
        f"Expected one metadata row for {ASSAY_ID}, "
        f"found {len(metadata)}."
    )

metadata = metadata.iloc[0]

wt_sequence = metadata["target_seq"]
expected_length = int(metadata["seq_len"])

assert len(wt_sequence) == expected_length

print("=== Assay Metadata ===")
print(f"DMS ID          : {metadata['DMS_id']}")
print(f"Protein         : {metadata['molecule_name']}")
print(f"Sequence length : {expected_length}")
print(f"Mutated region  : {metadata['region_mutated']}")
print(
    f"Single mutants  : "
    f"{metadata['DMS_number_single_mutants']}"
)


# ============================================================
# Load official CV dataset
# ============================================================

cv_path = find_cv_file()

df = pd.read_csv(cv_path)

print("\n=== CV Dataset ===")
print(f"Path  : {cv_path}")
print(f"Shape : {df.shape}")

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")


required_columns = {
    "mutant",
    "mutated_sequence",
    "DMS_score",
    "fold_random_5",
    "fold_modulo_5",
    "fold_contiguous_5",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}"
    )


# ============================================================
# Parse mutation labels
# ============================================================

parsed = df["mutant"].apply(parse_single_mutation)

df[["wt_aa", "position", "mut_aa"]] = pd.DataFrame(
    parsed.tolist(),
    index=df.index,
)


# ============================================================
# Validate mutations against WT sequence
# ============================================================

for row in df.itertuples():

    idx = row.position - 1

    # Position must exist
    assert 0 <= idx < len(wt_sequence), row.mutant

    # WT amino acid must match
    assert wt_sequence[idx] == row.wt_aa, (
        f"{row.mutant}: "
        f"WT sequence has {wt_sequence[idx]} "
        f"at position {row.position}"
    )

    # Mutated sequence length must not change
    assert len(row.mutated_sequence) == len(wt_sequence)

    # Mutated amino acid must match label
    assert row.mutated_sequence[idx] == row.mut_aa

    # Single mutant means exactly one residue differs
    differences = [
        i
        for i, (wt, mut) in enumerate(
            zip(wt_sequence, row.mutated_sequence)
        )
        if wt != mut
    ]

    assert differences == [idx], (
        f"{row.mutant}: unexpected differences "
        f"at {differences}"
    )


print("\nMutation validation passed.")


# ============================================================
# Validate CV folds
# ============================================================

fold_columns = [
    "fold_random_5",
    "fold_modulo_5",
    "fold_contiguous_5",
]

print("\n=== Fold Distribution ===")

for column in fold_columns:

    assert not df[column].isna().any()

    folds = sorted(df[column].unique().tolist())

    assert folds == [0, 1, 2, 3, 4], (
        f"{column}: unexpected folds {folds}"
    )

    print(f"\n{column}")
    print(df[column].value_counts().sort_index())


# ============================================================
# Check position isolation
# ============================================================

print("\n=== Position Leakage Check ===")

for column in [
    "fold_modulo_5",
    "fold_contiguous_5",
]:

    folds_per_position = (
        df.groupby("position")[column]
        .nunique()
    )

    max_folds = folds_per_position.max()

    assert max_folds == 1

    print(
        f"{column}: "
        "each position belongs to exactly one fold"
    )


random_folds_per_position = (
    df.groupby("position")["fold_random_5"]
    .nunique()
)

print(
    "fold_random_5: maximum number of folds "
    f"for one position = "
    f"{random_folds_per_position.max()}"
)


# ============================================================
# Save processed dataset
# ============================================================

PROCESSED_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

processed_path = (
    PROCESSED_ROOT
    / "tem1_single_mutants.csv"
)

df.to_csv(
    processed_path,
    index=False,
)


fasta_path = (
    PROCESSED_ROOT
    / "tem1_wt.fasta"
)

with open(fasta_path, "w") as f:
    f.write(">BLAT_ECOLX_Stiffler_2015\n")
    f.write(wt_sequence + "\n")


print("\n=== Saved ===")
print(processed_path)
print(fasta_path)