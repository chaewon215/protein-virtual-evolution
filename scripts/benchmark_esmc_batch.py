from contextlib import nullcontext
from pathlib import Path
import time
import statistics

import pandas as pd
import torch

from esm.models.esmc import EsmcModel, EsmcTokenizer


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

MODEL_NAME = "biohub/ESMC-300M"

BATCH_SIZES = [8, 16, 32, 64, 128, 256]
N_WARMUP = 2
N_RUNS = 5


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Data
# ============================================================

df = pd.read_csv(DATA_PATH)

sequences = df["mutated_sequence"].tolist()

print("Number of sequences:", len(sequences))
print("Sequence length:", len(sequences[0]))


# ============================================================
# Model
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
# Forward function
# ============================================================

def forward_batch(batch_sequences):

    inputs = tokenizer(
        batch_sequences,
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
# Benchmark
# ============================================================

print("\n=== Batch Benchmark ===")

for batch_size in BATCH_SIZES:

    batch = sequences[:batch_size]

    try:

        torch.cuda.empty_cache()

        # -----------------------------------------
        # Warm-up
        # -----------------------------------------

        for _ in range(N_WARMUP):
            embeddings = forward_batch(batch)

        torch.cuda.synchronize()

        del embeddings
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        # -----------------------------------------
        # Benchmark
        # -----------------------------------------

        times = []

        for _ in range(N_RUNS):

            torch.cuda.synchronize()

            start = time.perf_counter()

            embeddings = forward_batch(batch)

            torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            times.append(elapsed)

        median_time = statistics.median(times)

        throughput = batch_size / median_time

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        print(
            f"Batch {batch_size:>3} | "
            f"shape={tuple(embeddings.shape)} | "
            f"time={median_time:.3f}s | "
            f"throughput={throughput:.1f} seq/s | "
            f"allocated={peak_allocated:.2f} GB | "
            f"reserved={peak_reserved:.2f} GB"
        )

        del embeddings

    except torch.cuda.OutOfMemoryError:

        print(
            f"Batch {batch_size:>3} | OOM"
        )

        torch.cuda.empty_cache()

        break