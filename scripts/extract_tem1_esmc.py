from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from esm.models.esmc import EsmcModel, EsmcTokenizer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DATA_PATH = (
    DATA_ROOT
    / "tem1_single_mutants.csv"
)

FASTA_PATH = (
    DATA_ROOT
    / "tem1_wt.fasta"
)

CACHE_ROOT = (
    DATA_ROOT
    / "esmc300m"
)

CHUNK_ROOT = (
    CACHE_ROOT
    / "chunks"
)

WT_CACHE_PATH = (
    CACHE_ROOT
    / "wt_embeddings.pt"
)

FINAL_CACHE_PATH = (
    CACHE_ROOT
    / "tem1_mutant_embeddings.pt"
)


MODEL_NAME = "biohub/ESMC-300M"

# 수정 가능
BATCH_SIZE = 16

# 한 checkpoint에 저장할 mutation 수
CHUNK_SIZE = 256


# ============================================================
# Setup
# ============================================================

CACHE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Mutations:", len(df))


with open(FASTA_PATH) as f:

    wt_sequence = "".join(
        line.strip()
        for line in f
        if not line.startswith(">")
    )


print("WT length:", len(wt_sequence))

assert len(wt_sequence) == 286

assert (
    df["mutated_sequence"]
    .str.len()
    .eq(len(wt_sequence))
    .all()
)


# ============================================================
# Load ESM-C
# ============================================================

print("\nLoading ESM-C 300M...")

tokenizer = EsmcTokenizer.from_pretrained(
    MODEL_NAME
)

model = EsmcModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# Inference function
# ============================================================

def get_embeddings(sequences):

    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    autocast_context = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), autocast_context:

        output = model(**inputs)

        embeddings = output.last_hidden_state

    return embeddings


# ============================================================
# WT embedding
# ============================================================

if WT_CACHE_PATH.exists():

    print("\nLoading cached WT embeddings...")

    wt_cache = torch.load(
        WT_CACHE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    wt_residue = wt_cache["residue_embeddings"]
    wt_global = wt_cache["global_embedding"]

else:

    print("\nExtracting WT embeddings...")

    wt_embeddings = get_embeddings(
        [wt_sequence]
    )

    assert wt_embeddings.shape == (
        1,
        288,
        960,
    )

    # Remove BOS / EOS
    wt_residue = (
        wt_embeddings[
            0,
            1:-1,
            :
        ]
        .float()
        .cpu()
    )

    assert wt_residue.shape == (
        286,
        960,
    )

    wt_global = (
        wt_residue
        .mean(dim=0)
    )

    torch.save(
        {
            "model_name": MODEL_NAME,
            "sequence": wt_sequence,
            "residue_embeddings": wt_residue,
            "global_embedding": wt_global,
        },
        WT_CACHE_PATH,
    )

    print(
        "Saved WT cache:",
        WT_CACHE_PATH,
    )


# ============================================================
# Mutant extraction
# ============================================================

n_samples = len(df)

print("\nExtracting mutant embeddings...")


for chunk_start in range(
    0,
    n_samples,
    CHUNK_SIZE,
):

    chunk_end = min(
        chunk_start + CHUNK_SIZE,
        n_samples,
    )

    chunk_path = (
        CHUNK_ROOT
        / f"chunk_{chunk_start:05d}_{chunk_end:05d}.pt"
    )


    # ----------------------------------------
    # Resume support
    # ----------------------------------------

    if chunk_path.exists():

        print(
            f"Skipping existing "
            f"{chunk_path.name}"
        )

        continue


    chunk_df = df.iloc[
        chunk_start:chunk_end
    ]


    local_parts = []
    global_parts = []


    for batch_start in tqdm(
        range(
            0,
            len(chunk_df),
            BATCH_SIZE,
        ),
        desc=(
            f"{chunk_start:05d}"
            f"-{chunk_end:05d}"
        ),
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            len(chunk_df),
        )

        batch_df = chunk_df.iloc[
            batch_start:batch_end
        ]


        sequences = (
            batch_df[
                "mutated_sequence"
            ]
            .tolist()
        )


        positions = torch.tensor(
            batch_df[
                "position"
            ].to_numpy(),
            dtype=torch.long,
            device=device,
        )


        embeddings = get_embeddings(
            sequences
        )


        batch_size_actual = len(
            batch_df
        )


        assert embeddings.shape == (
            batch_size_actual,
            288,
            960,
        )


        # ------------------------------------
        # Local embedding
        #
        # token 0 = BOS
        # protein position 1 = token 1
        #
        # Therefore position can be used
        # directly as token index.
        # ------------------------------------

        batch_indices = torch.arange(
            batch_size_actual,
            device=device,
        )


        mutant_local = embeddings[
            batch_indices,
            positions,
            :
        ]


        # ------------------------------------
        # Global embedding
        #
        # Remove BOS / EOS first.
        # ------------------------------------

        mutant_global = (
            embeddings[
                :,
                1:-1,
                :
            ]
            .mean(dim=1)
        )


        local_parts.append(
            mutant_local
            .float()
            .cpu()
        )

        global_parts.append(
            mutant_global
            .float()
            .cpu()
        )


        del embeddings
        del mutant_local
        del mutant_global


    # ========================================================
    # Save chunk
    # ========================================================

    chunk_local = torch.cat(
        local_parts,
        dim=0,
    )

    chunk_global = torch.cat(
        global_parts,
        dim=0,
    )


    torch.save(
        {
            "start": chunk_start,
            "end": chunk_end,

            "mutants": (
                chunk_df[
                    "mutant"
                ]
                .tolist()
            ),

            "positions": torch.tensor(
                chunk_df[
                    "position"
                ].to_numpy(),
                dtype=torch.long,
            ),

            "mutant_local": chunk_local,
            "mutant_global": chunk_global,
        },
        chunk_path,
    )


    print(
        f"Saved {chunk_path.name}"
    )


# ============================================================
# Merge chunks
# ============================================================

print("\nMerging chunks...")


chunk_files = sorted(
    CHUNK_ROOT.glob(
        "chunk_*.pt"
    )
)


all_mutants = []
all_positions = []
all_local = []
all_global = []


for path in chunk_files:

    chunk = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    all_mutants.extend(
        chunk["mutants"]
    )

    all_positions.append(
        chunk["positions"]
    )

    all_local.append(
        chunk["mutant_local"]
    )

    all_global.append(
        chunk["mutant_global"]
    )


positions = torch.cat(
    all_positions,
    dim=0,
)

mutant_local = torch.cat(
    all_local,
    dim=0,
)

mutant_global = torch.cat(
    all_global,
    dim=0,
)


# ============================================================
# Final validation
# ============================================================

assert len(all_mutants) == n_samples

assert (
    all_mutants
    ==
    df["mutant"].tolist()
)

assert positions.shape == (
    n_samples,
)

assert mutant_local.shape == (
    n_samples,
    960,
)

assert mutant_global.shape == (
    n_samples,
    960,
)


assert torch.isfinite(
    mutant_local
).all()

assert torch.isfinite(
    mutant_global
).all()


# ============================================================
# Save final cache
# ============================================================

torch.save(
    {
        "model_name": MODEL_NAME,

        "mutants": all_mutants,

        "positions": positions,

        "mutant_local": mutant_local,

        "mutant_global": mutant_global,
    },
    FINAL_CACHE_PATH,
)


print("\n=== Finished ===")

print(
    "Mutant local :",
    mutant_local.shape,
)

print(
    "Mutant global:",
    mutant_global.shape,
)

print(
    "Saved:",
    FINAL_CACHE_PATH,
)
