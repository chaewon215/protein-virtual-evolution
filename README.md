# Protein Virtual Evolution

Uncertainty-aware protein fitness prediction and virtual directed evolution using protein language models.

## Project Overview

This project aims to predict the functional effects of protein mutations using protein language models such as ESM-C and to explore high-fitness protein variants through uncertainty-aware virtual directed evolution.

## Planned Pipeline

1. ProteinGym DMS dataset preprocessing
2. ESM-C protein representation extraction
3. Protein fitness prediction
4. Generalization evaluation
5. Uncertainty estimation
6. Virtual directed evolution
7. ESM3-based candidate generation

## Project Structure

```text
protein-virtual-evolution/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
├── results/
├── scripts/
└── src/
    ├── data/
    ├── evaluation/
    └── models/
```

## Dataset

ProteinGym v1.3

Initial assay:

BLAT_ECOLX_Stiffler_2015