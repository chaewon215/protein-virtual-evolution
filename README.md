# Protein Virtual Evolution

[한국어 README](README_ko.md)

**Uncertainty-aware protein fitness prediction, virtual directed evolution, and structure-informed candidate validation using protein language models**

This project predicts the functional effects of protein mutations with protein language models and uses uncertainty-aware acquisition to prioritize high-fitness variants under a limited experimental budget. The final workflow combines **ESM-C representations**, supervised fitness prediction, **Deep Ensemble uncertainty**, retrospective virtual directed evolution, **ESM3-based sequence generation**, independent **ColabFold/AlphaFold2 structural validation**, GFP functional-site preservation analysis, and cross-protein external validation.

---

## Key Results

| Protein | Main role | Key result |
|---|---|---|
| **TEM-1** | Generalization under distribution shift | Spearman **0.9167** on random OOF, **0.7798** on unseen-position, **0.7666** on unseen-region |
| **GFP** | Uncertainty-aware virtual evolution | Conservative acquisition achieved **9.3× top-1% enrichment** while screening only ~**0.20%** of the candidate pool |
| **TPMT** | External protein validation | Spearman **0.5755** on random OOF and **0.5111** on unseen-position, without TPMT-specific tuning |
| **ESM3 design** | Novel GFP candidate generation | **47** novel sequences generated, **29** improved both predicted mean and conservative score vs. seed, **6** retained after diversity-aware shortlisting |
| **Structural validation** | Cross-model GFP candidate assessment | All 6 shortlisted candidates retained seed-like AlphaFold2 folds; chromophore-neighborhood RMSD **0.050–0.097 Å**, below WT self-variability (**0.131 Å**) |

The project treats ensemble disagreement as a **relative empirical reliability signal**, not as a universally calibrated confidence interval.

---

## Final Pipeline

```text
ProteinGym DMS
      ↓
Data preprocessing
      ↓
ESM-C protein representation
      ↓
Fitness prediction
      ↓
Generalization evaluation
      ↓
Deep Ensemble uncertainty
      ↓
Selective prediction / candidate prioritization
      ↓
Retrospective virtual directed evolution
      ↓
ESM3 novel candidate generation
      ↓
ColabFold / AlphaFold2 cross-model validation
      ↓
GFP functional-site preservation
      ↓
Cross-protein external validation
```

---

## Datasets

ProteinGym v1.3 assays were used for three complementary roles.

| Protein | Assay | Variants used | Main role |
|---|---|---:|---|
| **TEM-1** | `BLAT_ECOLX_Stiffler_2015` | 4,996 single mutants | Generalization + uncertainty |
| **GFP** | `GFP_AEQVI_Sarkisyan_2016` | 1,084 single + 50,630 multi-mutants | Virtual evolution + generation |
| **TPMT** | `TPMT_HUMAN_Matreyek_2018` | 3,648 single mutants | External validation |

`DMS_score` is assay-specific, so absolute RMSE/MAE values should not be compared directly across proteins.

---

# 1. TEM-1 Fitness Prediction, Generalization, and Uncertainty

TEM-1 was used to establish the core protein-fitness prediction and uncertainty framework.

## Representation

ESM-C 300M was used to encode the WT and mutant sequences.

For the selected TEM-1 representation:

```text
WT local embedding      [960]
Delta local embedding   [960]

        ↓ concatenate

Final input             [1920]
```

The selected predictor was:

```text
[1920] → [512] → [128] → [1]
```

A 5-member Deep Ensemble was trained from independent random initializations.

```text
Input
[B,1920]

Each member
[B,1920] → [B,512] → [B,128] → [B,1]

Stacked member predictions
[B,5]

Ensemble mean
μ [B]

Ensemble standard deviation
σ [B]
```

The ensemble mean is used as the predicted DMS score and the ensemble standard deviation is used as a relative epistemic uncertainty score.

## Generalization Results

| Split | Ensemble Spearman | RMSE | R² | Uncertainty-error ρ | High-error AUROC |
|---|---:|---:|---:|---:|---:|
| Random | **0.9167** | **0.4427** | **0.8522** | **0.5071** | **0.6864** |
| Modulo / unseen-position | 0.7798 | 0.7915 | 0.5275 | 0.4249 | 0.5954 |
| Contiguous / unseen-region | 0.7666 | 0.8241 | 0.4878 | 0.3836 | 0.5712 |

<p align="center">
  <img src="results/tem1/deep_ensemble_uncertainty/figures/risk_coverage_curve.png" width="780">
</p>

**Risk-Coverage Curve.** Samples were ranked from lowest to highest ensemble uncertainty. Retaining only low-uncertainty predictions produced substantially lower observed prediction error.

At 20% coverage, RMSE reduction relative to equally sized random subsets was:

```text
Random       65.2%
Modulo       29.0%
Contiguous   28.4%
```

The uncertainty signal weakened under stronger distribution shift, so ensemble standard deviation is treated as a **relative ranking/filtering signal**, not as a calibrated confidence interval.

Detailed analysis: [`docs/uncertainty_results.md`](docs/uncertainty_results.md)

---

# 2. GFP Retrospective Virtual Evolution

GFP was used to test whether **ESM-C representations + Deep Ensemble uncertainty** can prioritize high-fitness multi-mutants under a limited experimental budget.

## Experimental Setup

ProteinGym assay:

```text
GFP_AEQVI_Sarkisyan_2016
```

Pipeline:

```text
1,084 labeled single mutants
          ↓
ESM-C 300M representations
          ↓
Branch Fusion fitness predictor
          ↓
5-member Deep Ensemble
          ↓
50,630 hidden multi-mutant candidates
          ↓
Random / Greedy / UCB / Conservative acquisition
```

## Multi-Mutant Representation

For a variant with \(K\) mutations:

```text
WT context
[K,960] → mean → [960]

Local perturbation
[K,960] → sum  → [960]

Global perturbation
[960]

        ↓ concatenate

Final input
[2880]
```

## Branch Fusion Predictor

The final Branch Fusion model independently projects each feature group before fusion.

```text
WT context     [B,960] → [B,64]
Delta local    [B,960] → [B,64]
Delta global   [B,960] → [B,64]

                ↓ concatenate

               [B,192]
                  ↓
               [B,64]
                  ↓
               [B,1]

Output
[B] predicted GFP DMS_score
```

Architecture comparison:

| Model | Parameters | OOF Spearman | RMSE | R² |
|---|---:|---:|---:|---:|
| **Branch Fusion** | **196,929** | **0.5585** | 0.4594 | 0.4543 |
| Gated Fusion | 326,017 | 0.5252 | **0.4501** | **0.4761** |
| Compact MLP | 372,929 | 0.5197 | 0.4628 | 0.4463 |

Branch Fusion achieved the **highest OOF Spearman while using the fewest parameters**, whereas Gated Fusion achieved the best RMSE and R².

## Deep Ensemble Uncertainty

A 5-member Deep Ensemble improved predictive performance:

```text
OOF Spearman          = 0.5921
RMSE                  = 0.4458
R²                    = 0.4862
Uncertainty-error ρ   = 0.4151
High-error AUROC      = 0.8429
```

![GFP uncertainty vs error](results/gfp/deep_ensemble_uncertainty/figures/uncertainty_vs_error.png)

Low-uncertainty subsets were substantially more accurate than equally sized random subsets.

![GFP risk coverage](results/gfp/deep_ensemble_uncertainty/figures/risk_coverage_baselines.png)

| Coverage | Low-uncertainty RMSE | Random RMSE | Reduction vs random |
|---:|---:|---:|---:|
| 20% | 0.2058 | 0.4409 | **53.3%** |
| 40% | 0.1677 | 0.4455 | **62.4%** |
| 60% | 0.1918 | 0.4445 | **56.9%** |
| 80% | 0.3009 | 0.4465 | **32.6%** |

---

## Why Pool-Based Virtual Evolution?

We first tested whether the measured GFP variants formed a sufficiently connected one-mutation-at-a-time graph.

```text
K=2 reachable : 99.88%
K=3 reachable : 29.82%
K=4 reachable :  0.17%
K≥5 reachable :  0%
```

Only **32.5% of all measured multi-mutants** were reachable from observed single mutants through complete one-substitution paths.

Therefore, strict path-constrained evolution would mainly measure benchmark sparsity rather than acquisition quality. The final experiment was formulated as **retrospective pool-based active learning** over the measured multi-mutant pool.

![GFP graph reachability](results/gfp/mutation_graph_audit/figures/reachable_fraction_by_mutation_count.png)

---

## Retrospective Acquisition

Initial labeled set:

```text
1,084 single mutants
```

Hidden candidate pool:

```text
50,630 multi-mutants
```

Acquisition budget:

```text
100 candidates
≈ 0.20% of the full multi-mutant pool
```

Policies were fixed before multi-mutant labels were revealed:

\[
\text{Greedy}(x)=\mu(x)
\]

\[
\text{UCB}(x)=\mu(x)+\sigma(x)
\]

\[
\text{Conservative}(x)=\mu(x)-\sigma(x)
\]

with \(\beta=1\) for uncertainty-aware policies.

At each round, selected candidates were revealed through a hidden oracle and added to the labeled set before retraining the ensemble.

## Acquisition Results

Final results over **10 simulation seeds**:

| Policy | Best acquired multi | Mean acquired fitness | Top-5 mean | Top-1% hits / 100 | Top-100 recovered |
|---|---:|---:|---:|---:|---:|
| Random | 3.8908 ± 0.0666 | 2.6563 | 3.8099 | 0.7 | 0.2 |
| Greedy | 4.0869 ± 0.0031 | 3.4227 | 3.9737 | 8.9 | **2.2** |
| UCB | 4.0759 ± 0.0323 | 3.3125 | 3.9654 | 8.1 | 2.1 |
| Conservative | **4.0879 ± 0.0041** | **3.4870** | **3.9800** | **9.3** | 2.1 |

The global top 1% represents 1% of the candidate pool, while Conservative acquisition achieved a **9.3% top-1% hit rate**, corresponding to approximately **9.3× enrichment over candidate-pool prevalence**.

![GFP top-1% hit rate](results/gfp/virtual_evolution/figures/top1pct_hit_rate_vs_budget.png)

The main result is therefore not recovery of a new global optimum, but **sample-efficient enrichment of high-fitness multi-mutants under a constrained screening budget**.

> Initial best single mutant: `4.113576`<br>
> Global best multi-mutant: `4.123109`<br>
> Best acquired multi-mutant within budget 100: ~`4.088`

The initial single-mutant set already contained a variant very close to the global maximum, so improvement over the initial best was unusually difficult.

Detailed results: [`docs/gfp_virtual_evolution_results.md`](docs/gfp_virtual_evolution_results.md)

---

# 3. ESM3 Novel Candidate Generation

After retrospective validation of the fitness predictor and uncertainty signal, ESM3 was used to generate **novel GFP sequences not present in the measured ProteinGym pool**.

## Generation Strategy

High-confidence seed variants were ranked using the conservative acquisition score:

\[
\text{Conservative}(x)=\mu(x)-\sigma(x)
\]

For each selected seed:

1. design positions were chosen using mutation-position frequencies from the high-ranked candidate pool,
2. exactly one residue was masked,
3. ESM3 generated candidate substitutions,
4. duplicate, unchanged, invalid, and already observed ProteinGym sequences were rejected.

## Generation Results

```text
Novel generated candidates     : 47
μ improved vs seed             : 36 / 47 (76.6%)
μ−σ improved vs seed           : 29 / 47 (61.7%)
Diversity-aware shortlist      : 6
```

Candidates were rescored using the frozen ESM-C + 5-member Deep Ensemble pipeline.

For \(N=47\) generated candidates:

```text
Generated sequences         [47,238 aa]

ESM-C residue embeddings    [47,238,960]

WT context                  [47,960]
Delta local                 [47,960]
Delta global                [47,960]

Final feature X             [47,2880]

Member predictions          [47,5]
Predicted mean μ            [47]
Uncertainty σ               [47]
```

The generated candidates are **computationally ranked designs**, not experimentally validated improvements.

A representative high-ranked candidate was:

```text
Y39N:K52M:P75S:E132G:K158G:I229V

Predicted μ            = 4.2198
Uncertainty σ          = 0.0952
Conservative score     = 4.1247
Δμ vs seed             = +0.2280
Δ(μ−σ) vs seed         = +0.2004
```

---

# 4. Structural and Functional-Site Plausibility Validation

The final six ESM3-generated GFP candidates were evaluated in three stages: an ESM3 structural sanity check, independent ColabFold/AlphaFold2 structure prediction, and GFP functional-site preservation analysis.

## 4.1 ESM3 Structural Sanity Check

Each candidate, its seed, and WT GFP were folded repeatedly with ESM3.

```text
Raw ESM3 coordinates
[238,37,3]

Cα coordinates
[238,3]
```

However, repeated predictions of the **same WT sequence** showed substantial stochastic variability:

```text
WT self-RMSD mean    = 6.2410 Å
WT self-RMSD median  = 6.5275 Å
WT mean pLDDT        = 0.7620
WT mean pTM          = 0.7461
```

Because the model's same-sequence structural variability was large, ESM3 structure prediction was retained only as a **qualitative sanity check**, not as a hard candidate filter.

## 4.2 ColabFold / AlphaFold2 Cross-Model Validation

To reduce same-model validation bias, WT, the six seed variants, and the six generated candidates were independently predicted with ColabFold/AlphaFold2 using three random seeds.

```text
13 sequences × 3 replicates = 39 predicted structures

Per structure
Cα coordinates [238,3]
```

All six generated candidates retained seed-like global folds:

| Candidate | Seed–candidate RMSD median | Self-variability baseline | Excess over self | Design-site displacement | Mean pLDDT | Mean pTM |
|---|---:|---:|---:|---:|---:|---:|
| candidate_01 | 1.226 Å | 1.486 Å | -0.261 Å | 0.473 Å | 94.58 | 0.890 |
| candidate_02 | 1.653 Å | 1.778 Å | -0.125 Å | 0.626 Å | 95.04 | 0.897 |
| candidate_03 | 1.418 Å | 1.465 Å | -0.047 Å | 0.400 Å | 95.23 | 0.900 |
| candidate_04 | 1.576 Å | 1.507 Å | +0.069 Å | 0.769 Å | 94.84 | 0.893 |
| candidate_05 | 0.485 Å | 1.033 Å | -0.548 Å | 0.107 Å | 95.02 | 0.897 |
| candidate_06 | 1.500 Å | 1.559 Å | -0.058 Å | 0.591 Å | 95.04 | 0.897 |

Five of six candidates remained within their same-sequence self-variability baseline. The remaining candidate exceeded the baseline by only `0.069 Å`.

These results support **cross-model structural plausibility**, but do not establish experimental stability or function.

## 4.3 GFP Functional-Site Preservation

Global fold preservation alone is not sufficient to argue that GFP function is retained. We therefore evaluated the local structural environment surrounding the GFP chromophore-forming motif.

```text
Chromophore-forming motif
Ser65 – Tyr66 – Gly67

Predefined key environment residues
94, 96, 148, 203, 205, 222
```

For the nine predefined functional residues:

```text
WT functional-site Cα
[9,3]

Candidate functional-site Cα
[9,3]

        ↓

functional-site internal RMSD
scalar
```

An additional **8 Å chromophore neighborhood** was derived from the WT AlphaFold2 structures using a majority vote across the three WT replicates.

```text
WT-derived chromophore neighborhood
60 residues

Cα coordinates
[60,3]
```

None of the six candidates directly mutated the chromophore-forming motif or the predefined key environment residues.

The chromophore-neighborhood RMSDs were:

```text
WT self-variability baseline : 0.13066 Å

candidate_01   0.09651 Å
candidate_02   0.09308 Å
candidate_03   0.09400 Å
candidate_04   0.08445 Å
candidate_05   0.05027 Å
candidate_06   0.07547 Å
```

**All six candidates remained below the WT self-variability baseline.**

Functional-site AlphaFold2 confidence also remained high:

```text
candidate functional-site pLDDT
≈ 92.7–96.4
```

The nearest WT-relative mutation to the chromophore-forming motif ranged from approximately `3.43 Å` to `12.88 Å`. Even candidate_03, which contained the closest mutation, retained chromophore-neighborhood geometry below the WT self-variability baseline.

> **The generated GFP candidates retained seed-like global folds under an independent AlphaFold2-based predictor and preserved WT-like local geometry around the chromophore-forming motif. This supports computational structural and functional-site plausibility, but does not constitute experimental validation of chromophore maturation, fluorescence, quantum yield, or thermodynamic stability.**

Detailed analysis: [`docs/gfp_structural_functional_validation.md`](docs/gfp_structural_functional_validation.md)

### Future Work

Physics-based stability estimation such as **FoldX ΔΔG** could be added as a complementary filter for future wet-lab candidate selection.

---

# 5. Cross-Protein Validation

The final framework was evaluated on **TEM-1, GFP, and TPMT**, covering different proteins, phenotypes, and generalization settings.

| Protein | Main role | Random Spearman | Structured / external result |
|---|---|---:|---|
| TEM-1 | Generalization under distribution shift | **0.917** | 0.780 unseen-position / 0.767 unseen-region |
| GFP | Uncertainty-aware virtual evolution | 0.592 | 53–62% RMSE reduction at 20–60% low-uncertainty coverage |
| TPMT | External protein validation | 0.575 | **0.511 unseen-position**, without TPMT-specific tuning |

## Uncertainty Across Proteins

```text
TEM-1
ρ(σ, |error|) = 0.507 random
               = 0.425 unseen-position
               = 0.384 unseen-region

GFP
ρ(σ, |error|) = 0.415
High-error AUROC = 0.843

TPMT
ρ(σ, |error|) = 0.082 random
               = 0.100 unseen-position
```

TPMT showed weaker sample-level error discrimination, but low-uncertainty selection still reduced RMSE relative to random subsets.

At 20% coverage:

```text
TPMT random
0.2356 vs 0.2647
→ 11.0% reduction

TPMT position-unseen
0.2626 vs 0.2883
→ 8.9% reduction
```

This supports a deliberately conservative interpretation:

> **ESM-C representations + lightweight fitness predictors transferred across proteins and phenotypes, while ensemble disagreement acted as an empirical reliability signal whose strength varied across assays.**

Detailed analysis: [`docs/cross_protein_validation_final.md`](docs/cross_protein_validation_final.md)

---

# 6. Reproducible Result Aggregation

Cross-protein values are **not manually entered** in the canonical result tables.

They are recomputed from saved OOF predictions:

```bash
python scripts/build_cross_protein_summary.py --print-sources-only
python scripts/build_cross_protein_summary.py
```

Canonical outputs:

```text
results/cross_protein/
├── cross_protein_prediction_summary.csv
├── cross_protein_risk_coverage_summary.csv
├── source_manifest.csv
└── summary.json
```

`source_manifest.csv` records the exact OOF prediction file used for every protein/split.

This prevents manual metric transcription and keeps summary results traceable to the saved experimental outputs.

---

# 7. Project Structure

```text
protein-virtual-evolution/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/
├── notebooks/
├── results/
│   ├── tem1/
│   ├── gfp/
│   ├── tpmt/
│   └── cross_protein/
├── scripts/
└── src/
    ├── data/
    ├── evaluation/
    └── models/
```

---

# 8. Environment

```text
OS      : Ubuntu 24.04 LTS
GPU     : NVIDIA Tesla T4
VRAM    : 16 GB
PyTorch : 2.11.0+cu130
CUDA    : 13.0 runtime
ESM-C   : 300M
```

Attention:

```text
PyTorch scaled_dot_product_attention
```

Optional acceleration packages were not required:

```text
Transformer Engine : disabled
flash-attn          : disabled
xformers            : disabled
```

---

# Final Takeaway

This project separates **model development**, **generalization testing**, **uncertainty validation**, **candidate prioritization**, **novel sequence generation**, **cross-model structural validation**, **functional-site preservation analysis**, and **external replication**.

The strongest supported conclusion is:

> **Protein language model representations combined with lightweight supervised predictors can transfer across proteins and phenotypes, while Deep Ensemble uncertainty provides an additional empirical reliability signal whose usefulness varies substantially across assays and should be validated before downstream use.**

The project does **not** claim:

- universally calibrated uncertainty,
- experimentally validated superiority of ESM3-generated GFP sequences,
- or recovery of a new GFP global optimum.

Instead, it demonstrates a reproducible workflow that moves from mutation-effect prediction to uncertainty-aware candidate prioritization, novel sequence generation, cross-model structural assessment, functional-site plausibility analysis, and external protein validation.
