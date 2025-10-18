import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from catboost import Pool

# Reuse the same header and imputations as training
def header_unify(df: pd.DataFrame) -> pd.DataFrame:
    alias = {
        'Gender':'Gender', 'sex':'Gender',
        'Customer Type':'CustomerType','customer_type':'CustomerType','customertype':'CustomerType',
        'Type of Travel':'TypeOfTravel','type_of_travel':'TypeOfTravel','typeoftravel':'TypeOfTravel',
        'Class':'Class','class':'Class',
        'Satisfaction':'Satisfaction','satisfaction':'Satisfaction',
        'Age':'Age','age':'Age',
        'Flight Distance':'FlightDistance','flight_distance':'FlightDistance','flightdistance':'FlightDistance',
        'DepartureDelayInMinutes':'DepartureDelay','Departure Delay in Minutes':'DepartureDelay',
        'departuredelayinminutes':'DepartureDelay','departuredelay':'DepartureDelay',
        'ArrivalDelayInMinutes':'ArrivalDelay','Arrival Delay in Minutes':'ArrivalDelay',
        'arrivaldelayinminutes':'ArrivalDelay','arrivaldelay':'ArrivalDelay',
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
        'PassengerId':'ID','Passenger ID':'ID','id':'ID','Id':'ID'
    }
    ren = {k:v for k,v in alias.items() if k in df.columns}
    return df.rename(columns=ren)

def basic_impute_cast(X: pd.DataFrame, numeric, ratings, categorical, X_train_for_cats: pd.DataFrame|None=None):
    for c in numeric + ratings:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
            X[c] = X[c].fillna(X[c].median())
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype('category')
            # align unseen categories to UNK if we know training cats
            if X_train_for_cats is not None and c in X_train_for_cats.columns:
                known = pd.Index(X_train_for_cats[c].astype('category').cat.categories)
                X[c] = X[c].where(X[c].isin(known), 'UNK')
            X[c] = X[c].cat.add_categories(['UNK']).fillna('UNK')
    return X

def main(args):
    bundle = load(args.model)
    model = bundle["model"]
    cal = bundle["cal"]
    features = bundle["features"]
    categorical = bundle["categorical"]
    numeric = bundle["numeric"]
    ratings = bundle["ratings"]
    use_case = bundle["use_case"]
    best = bundle.get("best_threshold", {"th":0.5})

    test = pd.read_csv(args.test)
    test = header_unify(test)
    X = test[features].copy()
    # For category alignment we don't have the full train frame here, so just apply general UNK rule
    X = basic_impute_cast(X, numeric, ratings, categorical)

    cat_idx = [X.columns.get_loc(c) for c in categorical]
    pool = Pool(X, cat_features=cat_idx)

    p_raw = model.predict_proba(pool)[:,1]
    p = np.clip(cal.predict(p_raw), 1e-6, 1-1e-6)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    th = float(best.get("th", 0.5))
    yhat = (p >= th).astype(int)

    # IDs if present, else 1..N
    ids = test['ID'] if 'ID' in test.columns else np.arange(1, len(test)+1)

    sub = pd.DataFrame({"id": ids, "satisfaction": yhat})
    pro = pd.DataFrame({"id": ids, "prob_satisfied": p})

    sub_path = outdir / "submission.csv"
    pro_path = outdir / "probabilities.csv"
    sub.to_csv(sub_path, index=False)
    pro.to_csv(pro_path, index=False)

    print("Wrote:", sub_path)
    print("Wrote:", pro_path)
    print("Threshold used:", th)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .joblib produced by train_cv.py")
    p.add_argument("--test",  required=True, help="Path to test.csv")
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()
    main(args)
