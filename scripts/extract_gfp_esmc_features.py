from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from esm.models.esmc import EsmcModel, EsmcTokenizer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "biohub/ESMC-300M"

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_all_variants.csv"
)

WT_FASTA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfp_wt.fasta"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esmc300m_gfp"
)

CHUNK_ROOT = (
    OUTPUT_ROOT
    / "chunks"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHUNK_ROOT.mkdir(parents=True, exist_ok=True)

WT_OUTPUT_PATH = OUTPUT_ROOT / "gfp_wt_embeddings.pt"
FINAL_OUTPUT_PATH = OUTPUT_ROOT / "gfp_variant_features.pt"
METADATA_OUTPUT_PATH = OUTPUT_ROOT / "gfp_variant_feature_metadata.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_ROOT / "feature_summary.json"

BATCH_SIZE = 16
CHUNK_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Helpers
# ============================================================

def read_fasta_sequence(path: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    sequence = "".join(lines).upper()
    if not sequence:
        raise ValueError(f"Empty FASTA sequence: {path}")
    return sequence


def parse_positions(value) -> list[int]:
    if pd.isna(value):
        raise ValueError("Missing mutation_positions.")

    positions = [
        int(x)
        for x in str(value).split(",")
        if x != ""
    ]

    if not positions:
        raise ValueError(f"Could not parse mutation positions: {value}")

    return positions


# ============================================================
# Load data
#
# IMPORTANT:
# DMS_score is deliberately NOT loaded here.
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

required_feature_columns = [
    "mutant",
    "mutated_sequence",
    "mutation_count",
    "mutation_positions",
]

header_columns = pd.read_csv(INPUT_PATH, nrows=0).columns.tolist()

missing = set(required_feature_columns) - set(header_columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = pd.read_csv(
    INPUT_PATH,
    usecols=required_feature_columns,
    low_memory=False,
)

wt_sequence = read_fasta_sequence(WT_FASTA_PATH)
sequence_length = len(wt_sequence)

print("=" * 80)
print("GFP ESM-C Multi-Mutant Feature Extraction")
print("=" * 80)
print(f"Device          : {DEVICE}")
print(f"Model           : {MODEL_NAME}")
print(f"WT length       : {sequence_length}")
print(f"Variants        : {len(df)}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Chunk size      : {CHUNK_SIZE}")


# ============================================================
# Validate mutation metadata
# ============================================================

parsed_positions = []

for i, row in df.iterrows():
    positions = parse_positions(row["mutation_positions"])
    mutation_count = int(row["mutation_count"])

    if len(positions) != mutation_count:
        raise ValueError(
            f"Mutation count mismatch at row {i}: "
            f"{row['mutant']} | count={mutation_count}, "
            f"positions={positions}"
        )

    if any(
        position < 1 or position > sequence_length
        for position in positions
    ):
        raise ValueError(
            f"Out-of-range mutation position at row {i}: {positions}"
        )

    parsed_positions.append(positions)

print("Mutation metadata : OK")


# ============================================================
# Load ESM-C
# ============================================================

print("\nLoading ESM-C...")

tokenizer = EsmcTokenizer.from_pretrained(MODEL_NAME)
model = EsmcModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# ============================================================
# Encode WT once
#
# Raw WT sequence       [238 aa]
# Tokenized             [1, 240]
# ESM-C output          [1, 240, 960]
# Remove BOS/EOS        [238, 960]
# Mean pool             [960]
# ============================================================

@torch.inference_mode()
def encode_wt():
    inputs = tokenizer(
        [wt_sequence],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.autocast(
        device_type="cuda" if DEVICE == "cuda" else "cpu",
        dtype=torch.float16,
        enabled=(DEVICE == "cuda"),
    ):
        output = model(**inputs)

    embeddings = output.last_hidden_state[0].float().cpu()

    residue_embeddings = embeddings[1:sequence_length + 1]

    if residue_embeddings.shape != (sequence_length, 960):
        raise ValueError(
            "Unexpected WT residue embedding shape: "
            f"{tuple(residue_embeddings.shape)}"
        )

    global_embedding = residue_embeddings.mean(dim=0)

    return residue_embeddings, global_embedding


if WT_OUTPUT_PATH.exists():
    print(f"Loading cached WT embeddings: {WT_OUTPUT_PATH}")
    wt_cache = torch.load(WT_OUTPUT_PATH, map_location="cpu")
    wt_residue_embeddings = wt_cache["wt_residue_embeddings"]
    wt_global = wt_cache["wt_global"]
else:
    print("Encoding WT sequence...")
    wt_residue_embeddings, wt_global = encode_wt()

    torch.save(
        {
            "model_name": MODEL_NAME,
            "sequence_length": sequence_length,
            "wt_residue_embeddings": wt_residue_embeddings,
            "wt_global": wt_global,
        },
        WT_OUTPUT_PATH,
    )

print("WT residue embeddings:", tuple(wt_residue_embeddings.shape))
print("WT global embedding :", tuple(wt_global.shape))


# ============================================================
# Main fixed-size multi-mutant representation
#
# Variant with K mutations at p1...pK
#
# WT local embeddings       [K, 960]
# Mutant local embeddings   [K, 960]
# Local delta               [K, 960]
#
# WT context mean           [960]
#   mean_i h_WT(pi)
#
# Local perturbation sum    [960]
#   sum_i (h_MUT(pi) - h_WT(pi))
#
# Concatenate:
#   [WT context mean, local perturbation sum]
#   -> [1920]
#
# For K=1 this reduces EXACTLY to:
#   [WT local, Delta local]
# which matches the TEM-1 representation.
#
# Additional cached diagnostics:
#   mutant_global           [960]
#   delta_global            [960]
# ============================================================

@torch.inference_mode()
def extract_batch_features(
    sequences: list[str],
    positions_batch: list[list[int]],
):
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.autocast(
        device_type="cuda" if DEVICE == "cuda" else "cpu",
        dtype=torch.float16,
        enabled=(DEVICE == "cuda"),
    ):
        output = model(**inputs)

    embeddings = output.last_hidden_state.float().cpu()

    expected_shape = (
        len(sequences),
        sequence_length + 2,
        960,
    )

    if tuple(embeddings.shape) != expected_shape:
        raise ValueError(
            "Unexpected ESM-C output shape: "
            f"{tuple(embeddings.shape)} | expected={expected_shape}"
        )

    batch_features = []
    batch_wt_context_mean = []
    batch_delta_local_sum = []
    batch_mutant_global = []
    batch_delta_global = []

    for batch_index, positions in enumerate(positions_batch):
        mutant_residue_embeddings = embeddings[
            batch_index,
            1:sequence_length + 1,
            :
        ]  # [238, 960]

        residue_indices = torch.tensor(
            [position - 1 for position in positions],
            dtype=torch.long,
        )

        wt_local = wt_residue_embeddings[residue_indices]        # [K, 960]
        mutant_local = mutant_residue_embeddings[residue_indices]  # [K, 960]
        delta_local = mutant_local - wt_local                    # [K, 960]

        wt_context_mean = wt_local.mean(dim=0)                   # [960]
        delta_local_sum = delta_local.sum(dim=0)                 # [960]

        feature = torch.cat(
            [wt_context_mean, delta_local_sum],
            dim=0,
        )                                                        # [1920]

        mutant_global = mutant_residue_embeddings.mean(dim=0)    # [960]
        delta_global = mutant_global - wt_global                 # [960]

        batch_features.append(feature)
        batch_wt_context_mean.append(wt_context_mean)
        batch_delta_local_sum.append(delta_local_sum)
        batch_mutant_global.append(mutant_global)
        batch_delta_global.append(delta_global)

    return {
        "features": torch.stack(batch_features),
        "wt_context_mean": torch.stack(batch_wt_context_mean),
        "delta_local_sum": torch.stack(batch_delta_local_sum),
        "mutant_global": torch.stack(batch_mutant_global),
        "delta_global": torch.stack(batch_delta_global),
    }


# ============================================================
# Chunked extraction with restartability
# ============================================================

n_variants = len(df)
n_chunks = math.ceil(n_variants / CHUNK_SIZE)

print(f"\nTotal chunks     : {n_chunks}")

for chunk_index in range(n_chunks):
    start = chunk_index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_variants)

    chunk_path = CHUNK_ROOT / f"chunk_{chunk_index:04d}.pt"

    if chunk_path.exists():
        print(
            f"[skip] chunk {chunk_index + 1}/{n_chunks} "
            f"rows {start}:{end}"
        )
        continue

    print(
        f"\n[extract] chunk {chunk_index + 1}/{n_chunks} "
        f"rows {start}:{end}"
    )

    chunk_df = df.iloc[start:end]
    chunk_positions = parsed_positions[start:end]

    chunk_outputs = {
        "features": [],
        "wt_context_mean": [],
        "delta_local_sum": [],
        "mutant_global": [],
        "delta_global": [],
    }

    for local_start in tqdm(
        range(0, len(chunk_df), BATCH_SIZE),
        desc=f"chunk {chunk_index:04d}",
    ):
        local_end = min(
            local_start + BATCH_SIZE,
            len(chunk_df),
        )

        batch_df = chunk_df.iloc[local_start:local_end]

        sequences = (
            batch_df["mutated_sequence"]
            .astype(str)
            .tolist()
        )

        positions_batch = chunk_positions[local_start:local_end]

        outputs = extract_batch_features(
            sequences,
            positions_batch,
        )

        for key in chunk_outputs:
            chunk_outputs[key].append(outputs[key])

    saved_chunk = {
        "start": start,
        "end": end,
        "features": torch.cat(chunk_outputs["features"], dim=0),
        "wt_context_mean": torch.cat(
            chunk_outputs["wt_context_mean"], dim=0
        ),
        "delta_local_sum": torch.cat(
            chunk_outputs["delta_local_sum"], dim=0
        ),
        "mutant_global": torch.cat(
            chunk_outputs["mutant_global"], dim=0
        ),
        "delta_global": torch.cat(
            chunk_outputs["delta_global"], dim=0
        ),
    }

    torch.save(saved_chunk, chunk_path)


# ============================================================
# Merge chunks
# ============================================================

print("\nMerging chunks...")

merged = {
    "features": [],
    "wt_context_mean": [],
    "delta_local_sum": [],
    "mutant_global": [],
    "delta_global": [],
}

expected_start = 0

for chunk_index in range(n_chunks):
    chunk_path = CHUNK_ROOT / f"chunk_{chunk_index:04d}.pt"

    if not chunk_path.exists():
        raise FileNotFoundError(f"Missing chunk: {chunk_path}")

    chunk = torch.load(chunk_path, map_location="cpu")

    if chunk["start"] != expected_start:
        raise ValueError(
            f"Chunk alignment failure: "
            f"expected start={expected_start}, "
            f"found={chunk['start']}"
        )

    expected_start = chunk["end"]

    for key in merged:
        merged[key].append(chunk[key])

if expected_start != n_variants:
    raise ValueError(
        "Merged chunks do not cover all GFP variants."
    )

for key in merged:
    merged[key] = torch.cat(merged[key], dim=0)


# ============================================================
# Final validation
# ============================================================

expected_shapes = {
    "features": (n_variants, 1920),
    "wt_context_mean": (n_variants, 960),
    "delta_local_sum": (n_variants, 960),
    "mutant_global": (n_variants, 960),
    "delta_global": (n_variants, 960),
}

for key, expected_shape in expected_shapes.items():
    actual_shape = tuple(merged[key].shape)

    if actual_shape != expected_shape:
        raise ValueError(
            f"{key} shape mismatch: "
            f"{actual_shape} != {expected_shape}"
        )

    if not torch.isfinite(merged[key]).all():
        raise ValueError(
            f"Non-finite values found in {key}."
        )

print("Feature shape validation: OK")


# ============================================================
# Save label-free feature cache
# ============================================================

torch.save(
    {
        "model_name": MODEL_NAME,
        "feature_definition": (
            "[mean WT local over mutated positions, "
            "sum local delta over mutated positions]"
        ),
        "features": merged["features"],
        "wt_context_mean": merged["wt_context_mean"],
        "delta_local_sum": merged["delta_local_sum"],
        "mutant_global": merged["mutant_global"],
        "delta_global": merged["delta_global"],
    },
    FINAL_OUTPUT_PATH,
)


# ============================================================
# Save label-free alignment metadata
# ============================================================

metadata_df = df[
    [
        "mutant",
        "mutation_count",
        "mutation_positions",
    ]
].copy()

metadata_df["feature_index"] = np.arange(n_variants)

metadata_df.to_csv(
    METADATA_OUTPUT_PATH,
    index=False,
)


# ============================================================
# Embedding-only diagnostics by mutation count
# No DMS labels are used.
# ============================================================

feature_np = merged["features"].numpy()
delta_sum_np = merged["delta_local_sum"].numpy()
delta_global_np = merged["delta_global"].numpy()

mutation_counts = df["mutation_count"].to_numpy(dtype=np.int64)

diagnostics = []

for k in sorted(np.unique(mutation_counts)):
    mask = mutation_counts == k

    diagnostics.append(
        {
            "mutation_count": int(k),
            "n_variants": int(mask.sum()),
            "mean_feature_l2": float(
                np.linalg.norm(
                    feature_np[mask],
                    axis=1,
                ).mean()
            ),
            "mean_delta_local_sum_l2": float(
                np.linalg.norm(
                    delta_sum_np[mask],
                    axis=1,
                ).mean()
            ),
            "mean_delta_global_l2": float(
                np.linalg.norm(
                    delta_global_np[mask],
                    axis=1,
                ).mean()
            ),
        }
    )


summary = {
    "model_name": MODEL_NAME,
    "n_variants": int(n_variants),
    "sequence_length": int(sequence_length),
    "feature_shape": [int(n_variants), 1920],
    "feature_definition": {
        "wt_context_mean": (
            "mean WT ESM-C local embedding "
            "over all mutated positions"
        ),
        "wt_context_mean_shape": [960],
        "delta_local_sum": (
            "sum over mutated positions of "
            "(mutant local embedding - WT local embedding)"
        ),
        "delta_local_sum_shape": [960],
        "concatenated_feature_shape": [1920],
    },
    "label_usage": (
        "No DMS_score column is loaded during feature extraction."
    ),
    "diagnostics_by_mutation_count": diagnostics,
}

with open(
    SUMMARY_OUTPUT_PATH,
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
# Final report
# ============================================================

print("\n" + "=" * 80)
print("Extraction Complete")
print("=" * 80)

print("Main feature       :", tuple(merged["features"].shape))
print(
    "WT context mean    :",
    tuple(merged["wt_context_mean"].shape),
)
print(
    "Delta local sum    :",
    tuple(merged["delta_local_sum"].shape),
)
print(
    "Mutant global      :",
    tuple(merged["mutant_global"].shape),
)
print(
    "Delta global       :",
    tuple(merged["delta_global"].shape),
)

print("\nSaved:")
print(WT_OUTPUT_PATH)
print(FINAL_OUTPUT_PATH)
print(METADATA_OUTPUT_PATH)
print(SUMMARY_OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "This extraction stage is label-free. "
    "Multi-mutant DMS_score was not loaded."
)
