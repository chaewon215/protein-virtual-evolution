# Step 11-5. Cross-Protein Validation

## 1. Purpose

This step evaluates whether the protein-fitness modeling framework transfers across different proteins, phenotypes, and distribution-shift settings.

The final comparison includes:

| Protein | Primary role | Evaluation setting |
|---|---|---|
| TEM-1 | Generalization under structured distribution shift | Random, unseen-position, unseen-region |
| GFP | Uncertainty-aware candidate prioritization | Random single-mutant OOF |
| TPMT | External protein validation without TPMT-specific tuning | Random, unseen-position |

All values in this document are derived from the canonical OOF prediction files under `results/` and recomputed by:

```bash
python scripts/build_cross_protein_summary.py
```

The generated canonical artifacts are:

```text
results/cross_protein/
├── cross_protein_prediction_summary.csv
├── cross_protein_risk_coverage_summary.csv
├── source_manifest.csv
└── summary.json
```

No cross-protein metric value is hard-coded in the summary script.

---

## 2. Shared modeling framework

### ESM-C representation

For GFP and TPMT, the final fixed representation was:

```text
WT local / context   [960]
Delta local          [960]
Delta global         [960]

        ↓ concat

Input                [2880]
```

### Branch Fusion predictor

```text
WT branch
[B,960] → [B,64]

Delta-local branch
[B,960] → [B,64]

Delta-global branch
[B,960] → [B,64]

concat
[B,192]

head
[B,192] → [B,64] → [B,1]

final prediction
[B]
```

The TPMT experiment reused the GFP-selected representation, architecture, ensemble size, and 24-epoch training protocol without TPMT-specific hyperparameter tuning.

---

## 3. Cross-protein prediction performance

| Protein | Split | N | Spearman | RMSE | MAE | R² | Mean uncertainty | ρ(σ, \|error\|) | High-error AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TEM-1 | Random | 4,996 | **0.9167** | 0.4427 | 0.2975 | **0.8522** | 0.1177 | **0.5071** | 0.6864 |
| TEM-1 | Modulo / unseen-position | 4,996 | 0.7798 | 0.7915 | 0.5728 | 0.5275 | 0.1289 | 0.4249 | 0.5954 |
| TEM-1 | Contiguous / unseen-region | 4,996 | 0.7666 | 0.8241 | 0.6099 | 0.4878 | 0.1369 | 0.3836 | 0.5712 |
| GFP | Random single-mutant OOF | 1,084 | 0.5921 | 0.4458 | 0.2306 | 0.4862 | 0.0725 | 0.4151 | **0.8429** |
| TPMT | Random | 3,648 | 0.5755 | 0.2645 | 0.1979 | 0.4458 | 0.0626 | 0.0822 | 0.5593 |
| TPMT | Position-unseen | 3,648 | 0.5111 | 0.2885 | 0.2216 | 0.3403 | 0.0662 | 0.0999 | 0.5684 |

### Important comparison rule

Raw `DMS_score` scales differ across assays. Therefore, absolute RMSE and MAE should **not** be compared directly across TEM-1, GFP, and TPMT.

Cross-protein interpretation should instead focus on:

- Spearman rank correlation,
- within-assay R²,
- robustness under distribution shift,
- uncertainty-error association,
- selective-risk improvement.

---

## 4. TEM-1: generalization under structured shift

TEM-1 showed the strongest overall predictive performance.

```text
Random
Spearman = 0.9167
R²       = 0.8522

Modulo / unseen-position
Spearman = 0.7798
R²       = 0.5275

Contiguous / unseen-region
Spearman = 0.7666
R²       = 0.4878
```

Prediction performance decreased under structured shift, but substantial ranking signal remained.

The Deep Ensemble uncertainty signal also weakened gradually under stronger shift:

```text
Random
ρ(σ, |error|) = 0.5071

Modulo
ρ(σ, |error|) = 0.4249

Contiguous
ρ(σ, |error|) = 0.3836
```

This indicates that uncertainty remained informative even when the test distribution became more difficult.

---

## 5. GFP: uncertainty-aware candidate prioritization

GFP provided the clearest evidence that uncertainty could be used operationally rather than only diagnostically.

The 5-member Deep Ensemble achieved:

```text
Spearman            = 0.5921
RMSE                = 0.4458
R²                  = 0.4862
ρ(σ, |error|)       = 0.4151
High-error AUROC    = 0.8429
```

The uncertainty signal strongly separated lower-risk from higher-risk subsets.

Risk-coverage results:

| Coverage | Low-uncertainty RMSE | Random RMSE | Reduction vs random |
|---:|---:|---:|---:|
| 20% | 0.2058 | 0.4409 | **53.3%** |
| 40% | 0.1677 | 0.4455 | **62.4%** |
| 60% | 0.1918 | 0.4445 | **56.9%** |
| 80% | 0.3009 | 0.4465 | **32.6%** |

The validated uncertainty signal was subsequently used in the retrospective virtual-evolution experiment over 50,630 GFP multi-mutant candidates.

With an acquisition budget of 100 candidates, model-based policies strongly enriched high-fitness variants relative to random selection.

---

## 6. TPMT: external protein validation

TPMT tested whether the frozen GFP modeling design transferred to a different human protein and a different phenotype.

No TPMT-specific feature selection, architecture search, or epoch tuning was performed.

### Prediction transfer

Deep Ensemble performance:

```text
Random
Spearman = 0.5755
R²       = 0.4458

Position-unseen
Spearman = 0.5111
R²       = 0.3403
```

Thus, useful ranking signal remained even when the mutation positions in the test fold were not observed during training.

### Uncertainty transfer

TPMT uncertainty was substantially weaker than TEM-1 or GFP as a sample-level error-ranking signal:

```text
Random
ρ(σ, |error|)    = 0.0822
High-error AUROC = 0.5593

Position-unseen
ρ(σ, |error|)    = 0.0999
High-error AUROC = 0.5684
```

However, low-uncertainty selection still provided modest selective-risk benefits.

At 20% coverage:

```text
Random split
RMSE: 0.2356 vs 0.2647 random
Reduction = 11.0%

Position-unseen split
RMSE: 0.2626 vs 0.2883 random
Reduction = 8.9%
```

Therefore, TPMT supports **prediction transfer**, while uncertainty transfer should be characterized as **weaker but still selectively useful**.

---

## 7. Cross-protein selective-risk comparison

RMSE reduction of the lowest-uncertainty subset relative to equally sized random subsets:

| Protein / split | 20% | 40% | 60% | 80% |
|---|---:|---:|---:|---:|
| TEM-1 Random | **65.2%** | 34.2% | 15.9% | 6.3% |
| TEM-1 Modulo | 29.0% | 16.2% | 5.7% | 0.2% |
| TEM-1 Contiguous | 28.4% | 14.2% | 4.2% | 0.4% |
| GFP Random OOF | 53.3% | **62.4%** | **56.9%** | **32.6%** |
| TPMT Random | 11.0% | 9.0% | 6.2% | 5.3% |
| TPMT Position-unseen | 8.9% | 7.8% | 4.2% | 2.5% |

The benefit of uncertainty-aware filtering differs substantially across proteins and assay settings.

A robust interpretation is therefore:

> Deep Ensemble disagreement can provide a useful selective-prediction signal, but its strength is protein- and assay-dependent and should be validated empirically before use.

---

## 8. Cross-protein interpretation

The three proteins play complementary methodological roles.

### TEM-1

Shows that both prediction and uncertainty retain useful signal under unseen-position and unseen-region distribution shifts.

### GFP

Shows that uncertainty can be used downstream for candidate prioritization, virtual directed evolution, and novel sequence design.

### TPMT

Shows that the frozen modeling framework transfers to a third protein and an abundance-based phenotype without TPMT-specific tuning.

Together, the project supports the following workflow:

```text
ProteinGym DMS
      ↓
ESM-C representation
      ↓
Fitness prediction
      ↓
Generalization evaluation
      ↓
Deep Ensemble uncertainty
      ↓
Selective prediction / candidate prioritization
      ↓
ESM3 sequence generation
      ↓
Structural plausibility review
```

---

## 9. Main conclusion

Across TEM-1, GFP, and TPMT, ESM-C representations combined with lightweight supervised fitness predictors consistently learned useful variant-ranking signals.

The strongest supported conclusion is:

> **Protein language model representations combined with lightweight supervised predictors can transfer across proteins and phenotypes, while Deep Ensemble uncertainty provides an additional reliability signal whose usefulness varies substantially across assays and must be validated empirically.**

The project does **not** claim universally calibrated uncertainty.

Instead, it demonstrates three levels of evidence:

1. **Prediction transfer** — observed across all three proteins.
2. **Generalization under shift** — retained on TEM-1 and TPMT structured splits.
3. **Uncertainty-aware decision making** — strongest on GFP and TEM-1; weaker but still selectively useful on TPMT.

---

## 10. Reproducibility

Canonical cross-protein numbers should always be regenerated from the raw OOF prediction artifacts:

```bash
python scripts/build_cross_protein_summary.py --print-sources-only
python scripts/build_cross_protein_summary.py
```

The exact source file used for every protein/split is recorded in:

```text
results/cross_protein/source_manifest.csv
```

This avoids manual transcription of experimental metrics and keeps the README/documentation traceable to the saved model outputs.
