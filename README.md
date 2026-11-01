# GreenFormula AI

## What this project does
GreenFormula AI uses machine learning to find sustainable cosmetic ingredients before they reach the lab.

1. **Predicts biodegradability.** A model trained on 2,830 lab-tested chemicals predicts from a molecule's structure whether it is readily biodegradable.
2. **Screens bio-sourced replacements for mineral oil.** It scores 9 plant-based oils against mineral oil on four checks:
   - **Biodegradable**: ML model prediction
   - **Oxidation stability**: counts C=C bonds and bis-allylic sites, which make oils go rancid
   - **pH 2–12 stability**: flags ester bonds, which break apart in strong acid or alkali
   - **Texture match**: closeness in molecular weight to mineral oil
3. **Ranks ingredient swaps** for a silicone (D5), a preservative (BHT) and a synthetic musk (galaxolide).

## Results

### Biodegradability model
| Test | Accuracy | ROC-AUC |
|---|---|---|
| 5-fold cross-validation (2,830 molecules) | 85.1% | 0.931 |
| Random 20% hold-out (566 molecules) | 84.3% | 0.940 |
| Scaffold hold-out, unseen ring structures (566 molecules) | 84.1% | 0.816 |

On the hold-out set, the model screened out 59% of candidates before lab testing, and 89% of those rejections were correct.

### Mineral oil replacement
| Oil | Biodegradable | Oxidation stable | pH 2–12 stable | Score |
|---|---|---|---|---|
| **C15–19 alkane** | yes | yes | yes | **0.95** |
| Undecane | yes | yes | yes | 0.89 |
| Hemisqualane | borderline | yes | yes | 0.77 |
| Isoamyl laurate | yes | yes | no | 0.73 |
| Coco-caprylate | yes | yes | no | 0.71 |
| Squalane | no | yes | yes | 0.63 |
| Caprylic/capric triglyceride | yes | yes | no | 0.59 |
| Olive oil | yes | partly | no | 0.40 |
| Sunflower oil | yes | no | no | 0.24 |

**Best candidate: the C15–19 plant-based alkane.** It has the same saturated-hydrocarbon chemistry as mineral oil, but comes from plants.

### Ingredient swaps
| Current ingredient | P(biodegradable) | Recommended swap | P(biodegradable) |
|---|---|---|---|
| Cyclopentasiloxane (D5) | 20% | Isoamyl laurate | 100% |
| BHT | 1% | Ferulic acid (from rice bran) | 59% |
| Galaxolide | 3% | Ethylene brassylate | 95% |

Squalane and tocopherol are bio-based but were flagged as not readily biodegradable.

## Tech stack
| Part | Used |
|---|---|
| Language | Python 3.12 |
| ML model | Random Forest classifier (500 trees, balanced class weights), scikit-learn 1.9 |
| Molecular features | RDKit 2026.09: 2,048 Morgan fingerprint counts (radius 2) + 217 molecular descriptors |
| Evaluation | Stratified 5-fold cross-validation, random and scaffold hold-out splits; accuracy, balanced accuracy, F1, ROC-AUC |
| Data handling | pandas, NumPy |
| Charts | Matplotlib |
| Dataset | "All-Public" ready-biodegradability dataset (Lunghini et al., *SAR and QSAR in Environmental Research*, 2020), via [mhlee216/Biodegradability_Prediction_QSAR_GCN](https://github.com/mhlee216/Biodegradability_Prediction_QSAR_GCN) |
