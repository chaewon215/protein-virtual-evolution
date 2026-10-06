# Step 8. GFP Retrospective Virtual Directed Evolution

## 1. Objective

The goal of Step 8 was to evaluate whether a protein language model–based fitness predictor, combined with Deep Ensemble uncertainty, can efficiently prioritize high-fitness GFP multi-mutants under a limited experimental budget.

The experiment was designed as a **retrospective pool-based active-learning simulation** using the ProteinGym assay `GFP_AEQVI_Sarkisyan_2016`.

A strict one-mutation-at-a-time directed-evolution simulation was also examined, but the observed benchmark graph was too sparse beyond low mutation counts. Therefore, the final simulation used the full measured multi-mutant pool while keeping hidden labels unavailable to the acquisition policy until selection.

---

## 2. Dataset

The GFP assay contains a 238-aa wild-type sequence and 51,714 measured variants.

| Variant type | Count |
|---|---:|
| Single mutants | 1,084 |
| Multi-mutants | 50,630 |
| Total | 51,714 |
| Maximum mutation count | 15 |

The 1,084 single mutants were used as the initial labeled set. The 50,630 multi-mutants were treated as a hidden candidate pool.

For leakage control, multi-mutant `DMS_score` values were not used during representation selection, model selection, uncertainty validation, or final candidate prediction. They were first accessed only through the hidden oracle in the retrospective acquisition simulation.

---

## 3. ESM-C Representation for Variable-Length Multi-Mutants

ESM-C 300M produces 960-dimensional residue embeddings.

For a variant with mutation positions \(p_1,\dots,p_K\), three fixed-dimensional representations were constructed:

$$
h_{\mathrm{WT-context}}
=
\frac{1}{K}
\sum_{i=1}^{K}
h_{\mathrm{WT}}(p_i)
\in \mathbb{R}^{960}
$$

$$
\Delta h_{\mathrm{local}}
=
\sum_{i=1}^{K}
\left(
h_{\mathrm{MUT}}(p_i)
-
h_{\mathrm{WT}}(p_i)
\right)
\in \mathbb{R}^{960}
$$

$$
\Delta h_{\mathrm{global}}
=
h_{\mathrm{MUT,global}}
-
h_{\mathrm{WT,global}}
\in \mathbb{R}^{960}
$$

The final input representation was:

$$
x =
[
h_{\mathrm{WT-context}},
\Delta h_{\mathrm{local}},
\Delta h_{\mathrm{global}}
]
\in \mathbb{R}^{2880}
$$

For a batch of size \(B\):

```text
WT context mean   [B, 960]
Delta local sum   [B, 960]
Delta global      [B, 960]
        ↓ concat
Input             [B, 2880]
```

For single mutants (\(K=1\)), this representation reduces naturally to a single mutation-site WT embedding, a local embedding difference, and a global embedding difference.

---

## 4. Single-Mutant Representation Ablation

All representation and model-selection experiments used only the 1,084 single-mutant labels.

The strongest representation for ranking was:

```text
[WT context, Delta local, Delta global]
Input dimension = 2,880
```

Using the compact MLP:

| Representation | Model | Global OOF Spearman | RMSE | R² |
|---|---|---:|---:|---:|
| WT + Δlocal + Δglobal | Compact MLP | **0.5245** | 0.4657 | 0.4392 |
| WT + Δlocal | Compact MLP | 0.4924 | **0.4589** | **0.4555** |
| Simple mutation one-hot | Compact MLP | 0.4723 | 0.5118 | 0.3227 |

The global perturbation feature was weak on its own but improved ranking when combined with local mutation-site information, suggesting complementary sequence-level information.

---

## 5. Architecture Ablation

Four predictor architectures were evaluated using the same 2,880-dimensional input and shared nested cross-validation protocol.

| Model | Parameters | Global OOF Spearman | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| **Branch Fusion** | **196,929** | **0.5585** | 0.4594 | **0.2389** | 0.4543 |
| Gated Fusion | 326,017 | 0.5252 | **0.4501** | 0.2441 | **0.4761** |
| Compact MLP | 372,929 | 0.5197 | 0.4628 | 0.2500 | 0.4463 |
| Residual MLP | 375,810 | 0.5027 | 0.4786 | 0.3127 | 0.4078 |

Because the downstream task requires candidate ranking, Branch Fusion was selected as the final backbone.

### Final Branch Fusion architecture

```text
WT context
[B,960] → Linear 960→64 → z_WT [B,64]

Delta local
[B,960] → Linear 960→64 → z_local [B,64]

Delta global
[B,960] → Linear 960→64 → z_global [B,64]

concat
[B,192]

→ Linear 192→64
→ ReLU
→ Dropout
→ Linear 64→1

Output
[B,1] predicted GFP DMS_score
```

This architecture explicitly separates the three biologically distinct feature groups before fusion.

---

## 6. Deep Ensemble and Uncertainty Validation

A five-member Deep Ensemble of Branch Fusion models was evaluated with 5-fold OOF prediction.

Each member had the same architecture and training protocol but independent initialization and training stochasticity.

For a batch of \(B\) variants:

```text
Member predictions
[B,5]

mean(axis=1)
→ μ [B]
→ predicted fitness

std(axis=1, ddof=1)
→ σ [B]
→ relative epistemic disagreement
```

### OOF prediction performance

| Metric | Value |
|---|---:|
| Spearman | **0.5921** |
| RMSE | **0.4458** |
| MAE | **0.2306** |
| R² | **0.4862** |

The ensemble improved over the single Branch Fusion model, whose OOF Spearman was 0.5585.

### Uncertainty quality

| Metric | Value |
|---|---:|
| Mean ensemble uncertainty | 0.0725 |
| Spearman(\(\sigma\), absolute error) | **0.4151** |
| High-error AUROC | **0.8429** |

The ensemble disagreement therefore contained useful information about prediction reliability.

### Error by uncertainty quintile

| Quintile | Mean uncertainty | MAE | RMSE |
|---:|---:|---:|---:|
| Q1 | 0.0208 | 0.1086 | 0.2053 |
| Q2 | 0.0345 | 0.0869 | 0.1180 |
| Q3 | 0.0477 | 0.1157 | 0.2325 |
| Q4 | 0.0711 | 0.2398 | 0.5019 |
| Q5 | 0.1890 | 0.6038 | 0.7959 |

The relationship was not perfectly monotonic at the lowest uncertainty levels, but the high-uncertainty region showed substantially larger errors.

### Risk-coverage analysis

| Coverage | Uncertainty-selected RMSE | Random RMSE | RMSE reduction vs random |
|---:|---:|---:|---:|
| 20% | 0.2058 | 0.4409 | **53.3%** |
| 40% | 0.1677 | 0.4455 | **62.4%** |
| 60% | 0.1918 | 0.4445 | **56.9%** |
| 80% | 0.3009 | 0.4465 | **32.6%** |
| 100% | 0.4458 | 0.4458 | 0% |

At 20–80% coverage, none of 1,000 random subsets achieved lower RMSE than the uncertainty-selected subset.

The ensemble standard deviation should be interpreted as a **relative epistemic uncertainty signal**, not as a calibrated probability or confidence interval.

---

## 7. Final Multi-Mutant Inference

The final five-member ensemble was trained on all 1,084 labeled single mutants for 24 epochs per member. The epoch count was fixed as the median of the OOF-selected epochs:

```text
[24, 9, 7, 48, 34]
median = 24
```

The 50,630 hidden multi-mutants were then scored without loading their labels.

```text
Multi-mutant input
[50630, 2880]

↓ 5-member ensemble

Member predictions
[50630, 5]

↓
μ [50630]
σ [50630]
```

Prediction statistics:

| Statistic | μ | σ |
|---|---:|---:|
| Minimum | -2.8745 | 0.0027 |
| Median | 3.5804 | 0.0844 |
| Mean | 3.3022 | 0.1555 |
| Maximum | 4.1091 | 1.5041 |

The mean uncertainty for hidden multi-mutants (0.1555) was higher than the single-mutant OOF mean uncertainty (0.0725), consistent with the model encountering a stronger distribution shift when extrapolating from single to multi-mutants.

This comparison is descriptive; the ensemble standard deviation is not calibrated across distributions.

---

## 8. Mutation-Graph Accessibility Audit

A strict stepwise mutation graph was constructed without using labels.

A directed edge was defined as:

```text
parent → child
```

when the child preserved all parent substitutions and added exactly one additional WT-relative substitution.

### Accessibility results

| Mutation count | Observed variants | Reachable from singles | Fraction reachable |
|---:|---:|---:|---:|
| 2 | 12,777 | 12,762 | **99.88%** |
| 3 | 12,336 | 3,678 | **29.82%** |
| 4 | 9,387 | 16 | **0.17%** |
| ≥5 | 15,431 | 0 | **0%** |

Overall:

```text
Reachable multi-mutants = 16,456 / 50,630
Reachable fraction      = 32.50%
```

Furthermore, none of the top-10 predicted candidates and only 2% of the top-100 predicted candidates were reachable through a complete observed one-substitution path.

Therefore, the benchmark was too sparse to support strict path-constrained directed evolution as the primary retrospective experiment.

This does **not** imply that unreachable variants are biologically impossible to construct. It only indicates that the necessary intermediate measurements are absent from the benchmark.

For this reason, the final experiment was formulated as **retrospective pool-based active learning over measured multi-mutants**.

---

## 9. Retrospective Virtual Evolution Protocol

### Initial state

```text
Initial labeled set
= 1,084 single mutants

Hidden candidate pool
= 50,630 multi-mutants
```

The initial single-mutant set did not count toward the acquisition budget.

### Acquisition protocol

```text
Budget      = 100 multi-mutants
Batch size  = 20
Rounds      = 5
Simulations = 10 seeds
```

At every round, model-based policies retrained a five-member Branch Fusion ensemble using all labels revealed so far.

The acquisition policies were frozen before multi-mutant labels were examined:

$$
\text{Random}: \text{uniform random}
$$

$$
\text{Greedy}: a(x)=\mu(x)
$$

$$
\text{UCB}: a(x)=\mu(x)+\sigma(x)
$$

$$
\text{Conservative}: a(x)=\mu(x)-\sigma(x)
$$

with \(\beta=1\) fixed for UCB and Conservative acquisition.

The acquisition function could access only sequence-derived metadata, model predictions, and uncertainty. A hidden oracle returned the true `DMS_score` only after a candidate had been selected.

---

## 10. Virtual Evolution Results

The experimental budget of 100 corresponds to only:

$$
\frac{100}{50630}\times 100
\approx 0.20\%
$$

of the full multi-mutant candidate pool.

### Final performance after budget = 100

| Policy | Best acquired multi | Mean acquired fitness | Top-5 mean | Top-1% hits | Top-100 recovered | Regret to global best |
|---|---:|---:|---:|---:|---:|---:|
| Random | 3.8908 ± 0.0666 | 2.6563 | 3.8099 | 0.7 | 0.2 | 0.2323 |
| Greedy | 4.0869 ± 0.0031 | 3.4227 | 3.9737 | 8.9 | **2.2** | 0.0362 |
| UCB | 4.0759 ± 0.0323 | 3.3125 | 3.9654 | 8.1 | 2.1 | 0.0472 |
| Conservative | **4.0879 ± 0.0041** | **3.4870** | **3.9800** | **9.3** | 2.1 | **0.0352** |

### High-fitness enrichment

The global top 1% consists of 507 multi-mutants.

With a budget of 100:

```text
Random        : 0.7 top-1% hits
Greedy        : 8.9 top-1% hits
UCB           : 8.1 top-1% hits
Conservative  : 9.3 top-1% hits
```

Relative to the 1% prevalence in the full candidate pool, the observed hit rates correspond to:

```text
Greedy        : 8.9× enrichment
UCB           : 8.1× enrichment
Conservative  : 9.3× enrichment
```

Relative to the empirical Random baseline in these 10 simulations, Conservative produced approximately:

$$
9.3 / 0.7 \approx 13.3
$$

times as many top-1% hits.

The prevalence-based enrichment is the more stable primary interpretation because the empirical Random estimate is based on only 10 simulation seeds.

---

## 11. Interpretation of Acquisition Policies

All three model-based policies substantially outperformed Random in prioritizing high-fitness multi-mutants.

Greedy and Conservative were particularly effective at exploiting high-confidence high-fitness predictions.

Conservative acquisition,

$$
a(x)=\mu(x)-\sigma(x),
$$

produced the highest mean acquired fitness, highest top-5 acquired fitness, and highest top-1% hit count in the 10-seed experiment.

UCB,

$$
a(x)=\mu(x)+\sigma(x),
$$

showed larger variability in the best acquired fitness:

```text
Greedy std        = 0.0031
UCB std           = 0.0323
Conservative std  = 0.0041
```

This is consistent with UCB explicitly rewarding uncertain candidates, which increases exploration but can reduce short-horizon exploitation performance.

The small differences among Greedy, UCB, and Conservative should not be interpreted as establishing a universally superior acquisition policy. The robust result is that all model-based policies strongly improved high-fitness enrichment relative to Random.

---

## 12. Initial-Best Limitation

The initial labeled set already contained a very high-fitness single mutant:

```text
Initial best single = 4.113576
Global best multi   = 4.123109
Difference          = 0.009533
```

The best multi-mutants discovered within budget 100 did not exceed the initial best single mutant.

Therefore:

```text
improvement_over_initial_best_single = 0
```

for all acquisition policies.

This limits the usefulness of "best overall fitness" as the primary metric for this assay.

The main conclusion should instead focus on **sample-efficient enrichment and discovery of high-fitness multi-mutants under a constrained experimental budget**.

---

## 13. Main Conclusion

Step 8 demonstrates that an ESM-C–based Branch Fusion predictor combined with Deep Ensemble uncertainty can support sample-efficient prioritization of GFP multi-mutants.

The five-member ensemble achieved:

```text
Single-mutant OOF Spearman = 0.592
Uncertainty-error rho      = 0.415
High-error AUROC           = 0.843
```

Under a budget of only 100 measurements, corresponding to approximately 0.20% of the 50,630-candidate pool, model-based acquisition policies strongly enriched high-fitness variants relative to random sampling.

In particular, Conservative acquisition achieved a 9.3% top-1% hit rate, corresponding to approximately 9.3× enrichment over the prevalence of top-1% candidates in the full pool.

The results support the use of uncertainty-aware protein fitness modeling as a prioritization strategy for limited-budget protein variant screening.

---

## 14. Important Limitations

1. The experiment is retrospective. All candidate fitness values were previously measured in ProteinGym, even though they were hidden from the acquisition policy until simulated reveal.

2. The final simulation is pool-based active learning, not strict one-substitution-path directed evolution. The mutation-graph audit showed that the measured GFP benchmark lacks sufficient intermediate variants for a strict path-constrained simulation.

3. The model was initially trained on single mutants and extrapolated to multi-mutants. Selected multi-mutant labels were incorporated only after acquisition.

4. Deep Ensemble standard deviation is used as a relative epistemic disagreement signal. It is not a calibrated confidence interval or probability of correctness.

5. The initial single-mutant set already contained a variant very close to the global maximum of the multi-mutant pool, making further improvement over the initial best unusually difficult.

6. Differences among Greedy, UCB, and Conservative policies are smaller than the difference between model-based acquisition and Random and should therefore be interpreted cautiously.

---

## 15. Key Artifacts

```text
results/gfp/single_feature_ablation/
results/gfp/architecture_ablation/
results/gfp/deep_ensemble_uncertainty/
results/gfp/final_ensemble/
results/gfp/mutation_graph_audit/
results/gfp/virtual_evolution/
```

Recommended primary figures:

```text
results/gfp/deep_ensemble_uncertainty/figures/
    uncertainty_vs_error.png
    uncertainty_quintile_vs_mae.png
    risk_coverage_baselines.png

results/gfp/mutation_graph_audit/figures/
    reachable_fraction_by_mutation_count.png
    observed_vs_reachable_counts.png

results/gfp/virtual_evolution/figures/
    best_acquired_multi_vs_budget.png
    top5_acquired_mean_vs_budget.png
    top100_recovery_vs_budget.png
    top1pct_hit_rate_vs_budget.png
```
