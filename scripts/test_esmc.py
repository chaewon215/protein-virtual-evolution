import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# ============================================================
# Load ESM-C
# ============================================================

print("Loading ESM-C 300M...")

model = ESMC.from_pretrained("esmc_300m").to(device)

model.eval()

print("Model loaded.")


# ============================================================
# Test sequence
# ============================================================

sequence = "ACDEFGHIKLMNPQRSTVWY"

protein = ESMProtein(sequence=sequence)


# ============================================================
# Encode
# ============================================================

protein_tensor = model.encode(protein)

output = model.logits(
    protein_tensor,
    LogitsConfig(
        sequence=True,
        return_embeddings=True,
    ),
)


print()
print("Sequence length:", len(sequence))
print("Embedding shape:", output.embeddings.shape)
print("Logits shape:", output.logits.sequence.shape)