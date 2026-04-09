# MMA Fight Outcome Predictor

Dual-model ML system predicting UFC fight outcomes built on a dataset of 92,716 real fights.

## Models
- Model 1: Predicts fight winner — 74% accuracy
- Model 2: Predicts method of victory (KO, Submission, Decision) — 50% F1-Macro

## Technologies
Python, Pandas, Scikit-learn, XGBoost, LightGBM, CatBoost, Neural Networks, Matplotlib

## Key Features
- Solved a Data Leakage problem with cumulative statistics per fight date
- Custom Quality Score feature engineering
- Neural Network Data Augmentation (150 synthetic fights)
- Soft Voting and Stacking ensemble methods
