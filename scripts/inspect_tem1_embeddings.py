from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tem1_single_mutants.csv"
)

FASTA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tem1_wt.fasta"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load WT sequence
# ============================================================

with open(FASTA_PATH) as f:
    lines = [
        line.strip()
        for line in f
        if not line.startswith(">")
    ]

wt_sequence = "".join(lines)

print("WT length:", len(wt_sequence))


# ============================================================
# Load mutation dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Number of mutations:", len(df))


# Use one example
row = df.iloc[0]

mutation = row["mutant"]
mut_sequence = row["mutated_sequence"]

position = int(row["position"])
wt_aa = row["wt_aa"]
mut_aa = row["mut_aa"]

print()
print("Example mutation:", mutation)
print("Position:", position)
print("WT amino acid:", wt_aa)
print("Mutant amino acid:", mut_aa)
print("DMS score:", row["DMS_score"])


# ============================================================
# Sanity check
# ============================================================

sequence_idx = position - 1

assert wt_sequence[sequence_idx] == wt_aa
assert mut_sequence[sequence_idx] == mut_aa


# ============================================================
# Load ESM-C
# ============================================================

print()
print("Loading ESM-C 300M...")

model = ESMC.from_pretrained("esmc_300m").to(device)

model.eval()


# ============================================================
# Embedding function
# ============================================================

def get_embeddings(sequence: str):

    protein = ESMProtein(
        sequence=sequence
    )

    protein_tensor = model.encode(
        protein
    )

    output = model.logits(
        protein_tensor,
        LogitsConfig(
            sequence=True,
            return_embeddings=True,
        ),
    )

    return output.embeddings


# ============================================================
# WT / mutant embeddings
# ============================================================

with torch.inference_mode():

    wt_embeddings = get_embeddings(
        wt_sequence
    )

    mut_embeddings = get_embeddings(
        mut_sequence
    )


print()
print("WT embedding shape :", wt_embeddings.shape)
print("Mut embedding shape:", mut_embeddings.shape)


# Remove <CLS> and <EOS>
wt_residue_embeddings = wt_embeddings[:, 1:-1, :]
mut_residue_embeddings = mut_embeddings[:, 1:-1, :]

print()
print(
    "Residue embedding shape:",
    wt_residue_embeddings.shape,
)


# Protein positions are 1-based
idx = position - 1


wt_local = wt_residue_embeddings[
    0,
    idx,
    :
]

mut_local = mut_residue_embeddings[
    0,
    idx,
    :
]

delta_local = (
    mut_local
    - wt_local
)


print()
print("Local WT shape    :", wt_local.shape)
print("Local mutant shape:", mut_local.shape)
print("Local delta shape :", delta_local.shape)


cosine_similarity = F.cosine_similarity(
    wt_local.unsqueeze(0),
    mut_local.unsqueeze(0),
).item()

l2_distance = torch.norm(
    delta_local,
    p=2,
).item()


print()
print(
    "Local cosine similarity:",
    f"{cosine_similarity:.6f}",
)

print(
    "Local L2 distance:",
    f"{l2_distance:.6f}",
)


wt_global = (
    wt_residue_embeddings[0]
    .mean(dim=0)
)

mut_global = (
    mut_residue_embeddings[0]
    .mean(dim=0)
)

delta_global = (
    mut_global
    - wt_global
)


print()
print("Global WT shape    :", wt_global.shape)
print("Global mutant shape:", mut_global.shape)
print("Global delta shape :", delta_global.shape)