# Octwave 2.0 — Machine Learning Competition

🥈 **Runner-up overall** — Octwave 2.0, organized by the IEEE Student Branch Chapter (2025).

Two-round ML competition hosted on Kaggle. Round 1 was a tabular classification problem (obesity risk prediction); Round 2 was a computer vision change-detection problem (spot the difference between image pairs).

> **Note:** Competition datasets are Kaggle-hosted and not redistributed here — this repo contains the modeling notebooks only. See the linked competition pages to access the data.

---

## Round 1 — Obesity Risk Prediction

[Kaggle Competition Page](https://www.kaggle.com/competitions/obesity-risk-prediction-challenge)

Multi-class classification (7 weight categories) on a tabular dataset of lifestyle and demographic features.

**Approach**
- Cleaned inconsistent categorical labels (typos, capitalization) and imputed missing values (mode for low-missing columns, an explicit "Unknown" category where >70% of a column was missing).
- Feature engineering: derived BMI from height/weight, added a squared BMI term (`BMI_Sq`) to better separate extreme categories, age binning into ordinal groups, and a `Sedentary_Diet_Risk` interaction feature (screen time × meal frequency).
- Preprocessing pipeline: `ColumnTransformer` with standard scaling for numeric features, one-hot encoding for nominal categoricals, and ordinal encoding for ordered categoricals (e.g. snack frequency, activity level).
- Model: LightGBM classifier, tuned via `RandomizedSearchCV` (n_estimators, learning_rate, max_depth, num_leaves) with GPU acceleration.

**Results**
- Baseline 5-fold CV accuracy: **0.760**
- Public leaderboard score: **0.77128**
- Private leaderboard score: **0.78125**

**Notebook:** [`round-1-obesity-risk/octwave-2.ipynb`](./round-1-obesity-risk/octwave-2.ipynb)

---

## Round 2 — Spot the Difference (Change Detection)

[Kaggle Competition Page](https://www.kaggle.com/competitions/spot-the-difference-challenge)

Multi-label classification: given a pair of images of the same scene, identify objects that were **added**, **removed**, or **changed** between them, across 51 candidate object classes.

**Approach**
- Cleaned label noise via semantic class merging (e.g. "man", "boy", "individual", "guy" → a single `person` class), addressing severe class imbalance and redundant categories.
- Model: two-stream Siamese-style CNN with a shared pre-trained ResNet-50 backbone. Features from both images are concatenated and passed through a classification head (`Linear → ReLU → Dropout(0.7) → Linear`) producing `3 × 51` logits (one set per change category).
- Trained with `BCEWithLogitsLoss` for multi-label classification, Adam optimizer with weight decay, and high dropout to control overfitting.
- Post-training threshold search over the validation set to find the probability cutoff that maximizes Mean F1.

**Results**
- Best validation Mean F1: **0.643**
- **Ranked 1st on both the public and private leaderboards**
- Public leaderboard F1: **0.594**
- Private leaderboard F1: **0.587**

**Notebook:** [`round-2-change-detection/octwave-final.ipynb`](./round-2-change-detection/octwave-final.ipynb)

---

## Tech Stack
`Python` · `Pandas` / `NumPy` · `scikit-learn` · `LightGBM` · `PyTorch` / `TorchVision` (ResNet-50) · Kaggle Notebooks (GPU)
