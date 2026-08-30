# _*Dataset employed*_
The dataset contained 471 compounds all of them previously assayed against FP2, the vast majority of them were form the public database ChEMBL and some retrieved from scientific literature. Then we transformed the compounds SMILES to canonical SMILES to ensure there are not any duplicates. The dataset was labeled following the next criteria: compounds were considered as active (1) or inactive (0) according to their reported inhibitory data. With IC50 ≤ 5 uM it was labeled as active ; otherwise, it would be considered as inactive (0). Once all the compounds were labeled the dataset was formed by 146 active compounds and 301 inactive compounds.

# _*Scripts*_

## _Machine Learning FP2 (Mordred descriptors)_
Calculates ~1600 2D physicochemical descriptors (Mordred) per SMILES, cleans empty/NaN/zero-variance columns, and imputes with the median. Filters redundancy using _Spearman_ correlation (threshold 0.7). Splits train/test with StratifiedGroupKFold (grouping near-identical compounds) and trains 14 ML algorithms via GridSearchCV. Selects the 3 best models by score_balanced (test ROC-AUC penalized for overfitting). Computes the applicability domain with kNN-AD (distance to the 5 nearest neighbors in the training set, threshold = mean + 2·SD) and the interpretability of the 3 models with permutation importance (magnitude + direction via Pearson correlation with the predicted probability).

## _Machine Learning FP2 (MACCS fingerprints)_
Converts each compound's canonical SMILES into a MACCS fingerprint (166 binary descriptors). Removes compounds with an identical MACCS vector but contradictory labels (non-modelable noise) and collapses redundant exact duplicates. Splits train/test with StratifiedGroupKFold grouped by MACCS vector. Trains the same 14 algorithms (Pipeline with an added variance filter) via grouped GridSearchCV, and selects the 3 best models using the same score_balanced criterion. Computes the applicability domain (kNN-AD, same method as Mordred) and the permutation-based interpretability, where each "variable" is the presence/absence of a specific structural fragment.

## _Analisis FP2_PF (Discrepancy Analysis)_
Script/notebook that computes molecular descriptors for compounds with tested activity against Falcipain-2 that also have recorded activity against _Plasmodium_, and compares — via PCA, t-SNE/UMAP, univariate statistical tests (Mann-Whitney, Fisher with FDR correction), and a Lasso model with Leave-One-Out validation — which characteristics distinguish compounds active against the falcipain-2 enzyme that do translate that activity to the whole organism (_P. falciparum_) from those that don't. Includes a confounding control to rule out that the difference is simply due to unequal enzymatic potency between groups. Output: tables of significant/consensus descriptors and plots (PCA, volcano plot, boxplots, t-SNE/UMAP).

# _*Website*_

## _Joblib files_
Files generated once from the original datasets; they are not executed on the website, only loaded. Each bundle contains the 3 best already-trained models (scikit-learn Pipelines with imputation + scaling + classifier), the list of descriptor columns they expect, the kNN-AD applicability domain parameters (scaler, nearest-neighbors model, threshold), a sample of the training set (for SHAP interpretability), and the label meaning. They must be copied alongside inference_pipeline.py, inside an artifacts/ folder, on the server running the website's backend.

## _Inference pipeline_
Loads the already-trained models (mordred_bundle.joblib, maccs_bundle.joblib) in seconds to predict without retraining anything. Each step is a function: it validates SMILES with RDKit, computes Mordred/MACCS descriptors, aligns them with the columns each bundle expects, checks the applicability domain (kNN-AD), and runs the 3 best saved models, returning the prediction (Active/Inactive), mean probability, model consensus, and per-molecule SHAP interpretability
