"""
neutralcarbon/data/preprocess.py
---------------------------------
Data loading, cleaning, and feature engineering for the
global_emissions.csv dataset (92 countries, 1992–2018).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "global_emissions.csv")

# Columns we care about for anomaly detection
FEATURE_COLS = [
    "Emissions.Production.CH4",
    "Emissions.Production.N2O",
    "Emissions.Production.CO2.Cement",
    "Emissions.Production.CO2.Coal",
    "Emissions.Production.CO2.Gas",
    "Emissions.Production.CO2.Oil",
    "Emissions.Production.CO2.Flaring",
    "Emissions.Production.CO2.Other",
    "Emissions.Production.CO2.Total",
]

ID_COLS = ["Year", "Country.Name", "Country.Code"]


def load_raw() -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"[load] {len(df)} rows, {df['Country.Name'].nunique()} countries, "
          f"years {df['Year'].min()}–{df['Year'].max()}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix tiny float noise (near-zero values from floating-point arithmetic),
    drop duplicates, sort by country + year.
    """
    df = df.copy()

    # Replace near-zero floating point artifacts with 0
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0.0 if abs(x) < 1e-10 else x)

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] Removed {before - len(df)} duplicate rows")

    df = df.sort_values(["Country.Name", "Year"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features useful for anomaly detection:
      - year-over-year % change per country per emission type
      - coal_fraction, oil_fraction, gas_fraction (share of total CO2)
      - gdp_per_capita, emissions_per_gdp
    """
    df = df.copy()

    # YoY change in total CO2
    df["CO2_Total_YoY_pct"] = (
        df.groupby("Country.Name")["Emissions.Production.CO2.Total"]
        .pct_change() * 100
    )

    # Source fractions
    total = df["Emissions.Production.CO2.Total"].replace(0, np.nan)
    for src in ["Coal", "Oil", "Gas", "Cement"]:
        col = f"Emissions.Production.CO2.{src}"
        if col in df.columns:
            df[f"{src}_fraction"] = df[col] / total

    # Emissions intensity (CO2 per unit GDP)
    gdp = df["Country.GDP"].replace(0, np.nan)
    df["emissions_per_gdp"] = df["Emissions.Production.CO2.Total"] / (gdp / 1e9)

    # GDP per capita
    pop = df["Country.Population"].replace(0, np.nan)
    df["gdp_per_capita"] = df["Country.GDP"] / pop

    return df


def get_feature_matrix(
    df: pd.DataFrame,
    extra_features: bool = True,
    scaler_type: str = "standard",
) -> tuple[np.ndarray, pd.DataFrame, object]:
    """
    Returns:
        X_scaled   — scaled feature matrix (n_samples, n_features)
        meta       — DataFrame with ID columns for traceability
        scaler     — fitted scaler object (for inverse transform)
    """
    if extra_features:
        df = engineer_features(df)
        cols = FEATURE_COLS + [
            "CO2_Total_YoY_pct",
            "Coal_fraction", "Oil_fraction", "Gas_fraction",
            "emissions_per_gdp", "gdp_per_capita",
        ]
    else:
        cols = FEATURE_COLS

    # Keep only columns that actually exist
    cols = [c for c in cols if c in df.columns]
    meta = df[ID_COLS].copy()

    X = df[cols].values

    # Impute missing / NaN
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Scale
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[features] Matrix shape: {X_scaled.shape} | cols: {cols}")
    return X_scaled, meta, scaler


def train_test_split_temporal(
    df: pd.DataFrame,
    test_years: list[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: train on earlier years, test on later years.
    Default test set: 2016, 2017, 2018.
    """
    if test_years is None:
        test_years = [2016, 2017, 2018]
    train = df[~df["Year"].isin(test_years)].copy()
    test  = df[ df["Year"].isin(test_years)].copy()
    print(f"[split] Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


if __name__ == "__main__":
    df_raw   = load_raw()
    df_clean = clean(df_raw)
    df_feat  = engineer_features(df_clean)
    print(df_feat.head())
    print(df_feat.describe())

    X, meta, scaler = get_feature_matrix(df_clean)
    print(f"Feature matrix: {X.shape}")
