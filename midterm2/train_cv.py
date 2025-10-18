
---

# 4) `train_cv.py`
```python
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier, Pool


def header_unify(df: pd.DataFrame) -> pd.DataFrame:
    # Map common header variants to a stable schema
    alias = {
        # core cats
        'Gender':'Gender', 'sex':'Gender',
        'Customer Type':'CustomerType','customer_type':'CustomerType','customertype':'CustomerType',
        'Type of Travel':'TypeOfTravel','type_of_travel':'TypeOfTravel','typeoftravel':'TypeOfTravel',
        'Class':'Class','class':'Class',

        # target
        'Satisfaction':'Satisfaction','satisfaction':'Satisfaction',

        # numerics
        'Age':'Age','age':'Age',
        'Flight Distance':'FlightDistance','flight_distance':'FlightDistance','flightdistance':'FlightDistance',
        'DepartureDelayInMinutes':'DepartureDelay','Departure Delay in Minutes':'DepartureDelay',
        'departuredelayinminutes':'DepartureDelay','departuredelay':'DepartureDelay',
        'ArrivalDelayInMinutes':'ArrivalDelay','Arrival Delay in Minutes':'ArrivalDelay',
        'arrivaldelayinminutes':'ArrivalDelay','arrivaldelay':'ArrivalDelay',

        # ratings (keep numeric 0-5)
        'Inflight wifi service':'InflightWifiService','inflightwifiservice':'InflightWifiService',
        'Departure/Arrival time convenient':'TimeConvenient','departurearrivaltimeconvenient':'TimeConvenient',
        'Ease of Online booking':'EaseOnline','easeofonlinebooking':'EaseOnline',
        'Gate location':'GateLocation','gatelocation':'GateLocation',
        'Food and drink':'FoodDrink','foodanddrink':'FoodDrink',
        'Online boarding':'OnlineBoarding','onlineboarding':'OnlineBoarding',
        'Seat comfort':'SeatComfort','seatcomfort':'SeatComfort',
        'Inflight entertainment':'InflightEntertainment','inflightentertainment':'InflightEntertainment',
        'On-board service':'OnBoardService','onboardservice':'OnBoardService',
        'Leg room service':'LegRoomService','legroomservice':'LegRoomService',
        'Baggage handling':'BaggageHandling','baggagehandling':'BaggageHandling',
        'Checkin service':'CheckinService','checkinservice':'CheckinService',
        'Inflight service':'InflightService','inflightservice':'InflightService',
        'Cleanliness':'Cleanliness','cleanliness':'Cleanliness',
        # ids
        'PassengerId':'ID','Passenger ID':'ID','id':'ID','Id':'ID'
    }
    # rename all matching keys
    ren = {k:v for k,v in alias.items() if k in df.columns}
    df = df.rename(columns=ren)

    # normalize target
    if 'Satisfaction' in df.columns:
        if df['Satisfaction'].dtype == object:
            s = df['Satisfaction'].str.lower().fillna('')
            df['Satisfaction'] = np.where((s.str.contains('satisfied') & ~s.str.contains('neutral')), 1, 0).astype(int)
        else:
            df['Satisfaction'] = df['Satisfaction'].astype(int)
    return df


def build_feature_lists(df: pd.DataFrame, use_case: str):
    numeric = ['Age','FlightDistance','DepartureDelay']
    if use_case == 'post' and 'ArrivalDelay' in df.columns:
        numeric.append('ArrivalDelay')
    ratings = [c for c in [
        'InflightWifiService','TimeConvenient','EaseOnline','GateLocation','FoodDrink',
        'OnlineBoarding','SeatComfort','InflightEntertainment','OnBoardService',
        'LegRoomService','BaggageHandling','CheckinService','InflightService','Cleanliness'
    ] if c in df.columns]
    categorical = [c for c in ['Gender','CustomerType','TypeOfTravel','Class'] if c in df.columns]
    features = numeric + ratings + categorical
    return features, numeric, ratings, categorical


def basic_impute_cast(X: pd.DataFrame, numeric, ratings, categorical):
    # numerics/ratings → median, coercing to numeric
    for c in numeric + ratings:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
            X[c] = X[c].fillna(X[c].median())
    # cats → category with UNK
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype('category')
            X[c] = X[c].cat.add_categories(['UNK']).fillna('UNK')
    return X


def choose_threshold(y_true, p, c_fp=1.0, c_fn=5.0, start=0.2, stop=0.8, num=61):
    grid = np.linspace(start, stop, num)
    best = None
    for th in grid:
        yhat = (p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        cost = c_fp*fp + c_fn*fn
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, yhat, average='binary', zero_division=0)
        row = dict(th=float(th), cost=float(cost), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                   precision=float(prec), recall=float(rec), f1=float(f1))
        if best is None or row['cost'] < best['cost']:
            best = row
    return best


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    train = header_unify(train)
    assert 'Satisfaction' in train.columns, "Target column 'Satisfaction' not found in train."

    features, numeric, ratings, categorical = build_feature_lists(train, args.use_case)
    X = train[features].copy()
    y = train['Satisfaction'].astype(int).values

    X = basic_impute_cast(X, numeric, ratings, categorical)
    cat_idx = [X.columns.get_loc(c) for c in categorical]

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X), dtype=float)
    fold_aucs = []
    models = []

    for fold,(tr,va) in enumerate(skf.split(X,y), 1):
        Xt, Xv = X.iloc[tr], X.iloc[va]
        yt, yv = y[tr], y[va]
        train_pool = Pool(Xt, label=yt, cat_features=cat_idx)
        valid_pool = Pool(Xv, label=yv, cat_features=cat_idx)

        model = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            loss_function='Logloss', eval_metric='AUC',
            random_seed=42, od_type='Iter', od_wait=100,
            verbose=200
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        proba = model.predict_proba(valid_pool)[:,1]
        oof[Xv.index] = proba
        auc = roc_auc_score(yv, proba)
        fold_aucs.append(float(auc))
        models.append(model)
        print(f"[Fold {fold}] AUC = {auc:.4f}")

    cv_auc = float(roc_auc_score(y, oof))
    print("CV AUC:", round(cv_auc,4))

    # Calibrate
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(oof, y)
    oof_cal = np.clip(cal.predict(oof), 1e-6, 1-1e-6)
    cv_auc_cal = float(roc_auc_score(y, oof_cal))
    print("Calibrated CV AUC:", round(cv_auc_cal,4))

    # Cost-aware threshold
    best = choose_threshold(y, oof_cal, c_fp=args.cost_fp, c_fn=args.cost_fn)
    print("Best threshold (cost-aware):", best)

    # Fit final model on full data
    full_pool = Pool(X, label=y, cat_features=cat_idx)
    final_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=6,
        loss_function='Logloss', eval_metric='AUC',
        random_seed=42, od_type='Iter', od_wait=200,
        verbose=200
    )
    final_model.fit(full_pool, use_best_model=False)

    # Save artifacts (don’t commit them)
    model_path = outdir / "airline_catboost_calibrated.joblib"
    dump(
        {
            "model": final_model,
            "cal": cal,
            "features": features,
            "categorical": categorical,
            "numeric": numeric,
            "ratings": ratings,
            "use_case": args.use_case,
            "best_threshold": best,
        },
        model_path,
        compress=3  # keeps file size sensible
    )
    print("Saved:", model_path)

    # Save CV report
    with open(outdir / "cv_report.json", "w") as f:
        json.dump(
            {
                "cv_auc": cv_auc,
                "cv_auc_calibrated": cv_auc_cal,
                "fold_aucs": fold_aucs,
                "best_threshold": best,
                "use_case": args.use_case,
                "features": features
            }, f, indent=2
        )

    # Feature importances
    importances = pd.Series(final_model.get_feature_importance(full_pool), index=X.columns).sort_values(ascending=False)
    importances.to_csv(outdir / "feature_importances.csv")
    print("Wrote feature_importances.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train.csv")
    p.add_argument("--use-case", choices=["pre","post"], default="post",
                   help="pre = exclude ArrivalDelay; post = include")
    p.add_argument("--outdir", default="models")
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--cost-fn", type=float, default=5.0)
    args = p.parse_args()
    main(args)
