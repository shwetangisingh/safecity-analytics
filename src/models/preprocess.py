import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "data/processed/crime_data_cleaned.csv"

# ── reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_classification_features(df: pd.DataFrame):
    feature_cols = [
        "AREA",
        "Hour",
        "Month",
        "IsWeekend",
        "Has Weapon",
        "Premise Category",
        "TimeBucket",
        "Severity",          # used when predicting Crime Category
        "Part 1-2",
        "Reporting Delay (Days)",
    ]

    df_feat = df[feature_cols].copy()

    # Boolean → int
    df_feat["IsWeekend"] = df_feat["IsWeekend"].astype(int)
    df_feat["Has Weapon"] = df_feat["Has Weapon"].astype(int)

    encoders = {}
    for col in ["Premise Category", "TimeBucket", "Severity"]:
        le = LabelEncoder()
        df_feat[col] = le.fit_transform(df_feat[col].astype(str))
        encoders[col] = le

    X = df_feat.values
    feature_names = feature_cols

    # Targets
    le_cat = LabelEncoder()
    y_category = le_cat.fit_transform(df["Crime Category"].astype(str))
    encoders["Crime Category"] = le_cat

    le_sev = LabelEncoder()
    y_severity = le_sev.fit_transform(df["Severity"].astype(str))
    encoders["Severity"] = le_sev

    return X, y_category, y_severity, feature_names, encoders


def get_scaled_features(df: pd.DataFrame):
    X, y_cat, y_sev, feat_names, encoders = get_classification_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_cat, y_sev, feat_names, encoders, scaler


def get_geo_features(df: pd.DataFrame):
    valid = df[df["Valid Coordinates"]].copy()
    X_geo = valid[["LAT", "LON", "Hour", "AREA"]].values
    return X_geo, valid
