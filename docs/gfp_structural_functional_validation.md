# GFP Structural and Functional Plausibility Validation

## Overview

ESM3로 생성한 GFP 후보 서열에 대해, 단순한 fitness prediction을 넘어 구조적·기능적 plausibility를 단계적으로 검토하였다.

최종 validation 흐름은 다음과 같다.

```text
ESM3 candidate generation
        ↓
ESM-C fitness + uncertainty scoring
        ↓
ESM3 structural sanity check
        ↓
ColabFold / AlphaFold2 cross-model validation
        ↓
GFP functional-site preservation analysis
```

이 단계들의 목적은 **실험적 기능 검증을 대체하는 것**이 아니라, 생성 후보가 계산적으로 보았을 때 기존 GFP fold와 주요 기능 부위를 보존하는지 확인하는 것이다.

---

## Step 10-1. ESM3 Structural Sanity Check

ESM3로 생성된 최종 shortlist 후보 6개에 대해 ESM3 structure generation을 수행하였다.

각 구조에서:

```text
Raw coordinates
[238,37,3]

Cα coordinates
[238,3]
```

를 추출해 WT, seed, candidate 사이의 RMSD를 비교하였다.

그러나 동일한 WT sequence를 반복 folding했을 때도 큰 structural variability가 관찰되었다.

```text
WT self-RMSD mean    = 6.2410 Å
WT self-RMSD median  = 6.5275 Å
WT mean pLDDT        = 0.7620
WT mean pTM          = 0.7461
```

즉 ESM3 structure prediction 자체의 stochasticity가 커서, 단일 candidate–seed RMSD를 mutation-induced structural change로 직접 해석하기 어려웠다.

따라서 ESM3 구조 예측은 **hard filter가 아니라 qualitative sanity check**로만 사용하였다.

---

## Step 10-2. ColabFold / AlphaFold2 Cross-Model Structural Validation

같은 모델로 생성과 검증을 모두 수행하는 편향을 줄이기 위해, shortlist 후보를 독립적인 AlphaFold2 계열 모델인 ColabFold로 다시 예측하였다.

### Prediction setup

```text
WT           1 sequence
Seed         6 sequences
Candidate    6 sequences

Total       13 sequences
```

각 sequence를 3개의 random seed로 반복 예측하였다.

```text
13 sequences × 3 replicates = 39 structures
```

각 structure:

```text
Cα coordinates
[L,3] = [238,3]
```

### Results

| Candidate | Seed–candidate RMSD median | Self-variability baseline | Excess over self | Design-site displacement | Mean pLDDT | Mean pTM |
|---|---:|---:|---:|---:|---:|---:|
| candidate_01 | 1.226 Å | 1.486 Å | -0.261 Å | 0.473 Å | 94.58 | 0.890 |
| candidate_02 | 1.653 Å | 1.778 Å | -0.125 Å | 0.626 Å | 95.04 | 0.897 |
| candidate_03 | 1.418 Å | 1.465 Å | -0.047 Å | 0.400 Å | 95.23 | 0.900 |
| candidate_04 | 1.576 Å | 1.507 Å | +0.069 Å | 0.769 Å | 94.84 | 0.893 |
| candidate_05 | 0.485 Å | 1.033 Å | -0.548 Å | 0.107 Å | 95.02 | 0.897 |
| candidate_06 | 1.500 Å | 1.559 Å | -0.058 Å | 0.591 Å | 95.04 | 0.897 |

6개 중 5개 candidate는 seed–candidate RMSD가 동일 sequence 반복 예측에서 관찰된 self-variability 범위 이내였다.

candidate_04만 baseline을 초과했지만, 차이는 약 `0.069 Å`로 매우 작았다.

따라서 모든 candidate는 독립적인 AlphaFold2 기반 prediction에서도 **seed-like global fold**를 유지하는 것으로 나타났다.

이 결과는 **cross-model structural plausibility**를 뒷받침하지만, 실제 stability 또는 function의 실험적 검증을 의미하지 않는다.

---

## Step 10-3. GFP Functional-Site Preservation Analysis

global fold가 유지되는 것만으로 GFP 기능이 보존된다고 결론 내릴 수 없기 때문에, fluorescence에 직접 관련된 구조적 환경을 추가로 분석하였다.

### Functional residues

Chromophore-forming motif:

```text
Ser65 – Tyr66 – Gly67
```

Key chromophore-environment residues:

```text
94, 96, 148, 203, 205, 222
```

총 9개 residue에 대해 functional-site geometry를 비교하였다.

```text
WT functional-site Cα
[9,3]

Candidate functional-site Cα
[9,3]

        ↓

functional-site internal RMSD
```

또한 WT AlphaFold2 구조에서 residues 65–67의 heavy atom으로부터 8 Å 이내에 있는 residue를 구해 chromophore neighborhood를 정의하였다.

3개의 WT replicate에서 majority vote로 선택된 neighborhood는 총:

```text
60 residues
```

였다.

### Mutation overlap

모든 후보에서:

```text
mutates_chromophore_motif       = False
mutates_key_environment_residue = False
```

즉 6개 candidate 모두 chromophore-forming motif와 지정한 핵심 주변 residue를 직접 변이시키지 않았다.

### Functional-site geometry

WT self-variability baseline:

```text
functional-site internal RMSD median
= 0.06773 Å
```

Candidate:

```text
candidate_01   0.06394 Å
candidate_02   0.10635 Å
candidate_03   0.10863 Å
candidate_04   0.07006 Å
candidate_05   0.05561 Å
candidate_06   0.07638 Å
```

candidate_01과 candidate_05는 WT self-variability보다 낮았고, 나머지 후보들도 절대값 기준으로 매우 작은 structural difference만 보였다.

### Chromophore-neighborhood geometry

WT self-variability baseline:

```text
0.13066 Å
```

Candidate:

```text
candidate_01   0.09651 Å
candidate_02   0.09308 Å
candidate_03   0.09400 Å
candidate_04   0.08445 Å
candidate_05   0.05027 Å
candidate_06   0.07547 Å
```

**6개 candidate 모두 WT self-variability baseline보다 낮았다.**

즉 chromophore 주변 local geometry가 WT 반복 예측 변동 범위보다도 더 작은 수준으로 유지되었다.

### Mutation proximity to chromophore

가장 가까운 WT-relative mutation과 chromophore-forming motif 사이의 최소 heavy-atom distance:

```text
candidate_01   12.88 Å
candidate_02    5.86 Å
candidate_03    3.43 Å
candidate_04    5.69 Å
candidate_05    8.81 Å
candidate_06    9.13 Å
```

candidate_03이 chromophore에 가장 가까운 mutation을 포함했지만, chromophore-neighborhood RMSD는 여전히 WT self baseline보다 낮았다.

### Functional-site confidence

```text
candidate_01   92.65
candidate_02   96.25
candidate_03   96.39
candidate_04   92.66
candidate_05   95.27
candidate_06   93.76
```

functional-site pLDDT는 모든 후보에서 높게 유지되었다.

---

## Final Interpretation

현재 결과가 뒷받침하는 가장 강한 결론은 다음과 같다.

> **ESM3로 생성한 GFP 후보들은 독립적인 AlphaFold2/ColabFold 구조 예측에서도 seed-like global fold를 유지했으며, GFP chromophore-forming motif와 주요 주변 residue를 직접 변이시키지 않았다. 또한 chromophore neighborhood의 local geometry가 모든 후보에서 WT self-variability 범위 이내로 유지되어, 계산적인 수준에서 functional-site preservation을 지지하였다.**

다만 다음은 주장하지 않는다.

- 실제 chromophore maturation이 성공한다는 것
- 실제 fluorescence가 유지되거나 증가한다는 것
- 실제 quantum yield가 보존된다는 것
- 실제 thermodynamic stability가 보존된다는 것
- 실험적으로 GFP 기능이 검증되었다는 것

따라서 본 분석은 **structural and functional plausibility validation**으로 해석한다.

---

## Why We Stop Here

추가적으로 FoldX/Rosetta 기반 ΔΔG 분석을 통해 stability prediction을 수행할 수 있지만, 본 프로젝트의 주된 질문은 다음과 같다.

```text
Can PLM-based fitness prediction and uncertainty guide
protein variant prioritization and candidate generation?
```

현재까지:

```text
Fitness prediction
        ↓
Uncertainty-aware prioritization
        ↓
Novel sequence generation
        ↓
Cross-model fold preservation
        ↓
Functional-site preservation
```

까지 연결되었기 때문에, 포트폴리오 프로젝트의 핵심 서사는 충분히 완결되었다.

FoldX 기반 stability analysis는 향후 실제 wet-lab candidate selection 또는 연구 확장 단계의 **Future Work**로 남긴다.

---

## Recommended README Claim

> **All six ESM3-generated GFP candidates retained seed-like global folds under independent AlphaFold2/ColabFold prediction. None directly mutated the chromophore-forming motif or predefined key environment residues, and chromophore-neighborhood Cα RMSDs remained below the WT self-variability baseline for all candidates. These results support cross-model structural and functional-site plausibility, but do not constitute experimental validation of GFP fluorescence.**
