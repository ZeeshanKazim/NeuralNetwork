# Airline Passenger Satisfaction — Accuracy Track (CatBoost)

Strong tabular baseline with:
- 5-fold Stratified CV (ROC-AUC)
- Robust header normalization & imputations
- Probability calibration (isotonic)
- Cost-aware threshold selection (FP vs FN)
- Clean inference script for test.csv

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Put your CSVs here (not committed):
# data/train.csv, data/test.csv

# Train (post-flight includes ArrivalDelay; use 'pre' to exclude it)
python train_cv.py --train data/train.csv --use-case post --outdir models --cost-fn 5 --cost-fp 1

# Predict on test.csv
python predict.py --model models/airline_catboost_calibrated.joblib --test data/test.csv --outdir outputs
