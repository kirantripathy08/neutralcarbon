"""
neutralcarbon/tests/test_anomaly.py
-------------------------------------
Unit tests for the ML anomaly detection pipeline.
Run: pytest tests/test_anomaly.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.preprocess import load_raw, clean, get_feature_matrix, engineer_features
from ml.anomaly_detection import (
    IsolationForestDetector,
    OneClassSVMDetector,
    IQRDetector,
    AnomalyResult,
    ensemble_predict,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_raw()

@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)

@pytest.fixture(scope="module")
def feature_matrix(clean_df):
    X, meta, scaler = get_feature_matrix(clean_df, extra_features=False)
    return X, meta, scaler

@pytest.fixture(scope="module")
def fitted_isolation_forest(feature_matrix):
    X, _, _ = feature_matrix
    model = IsolationForestDetector(contamination=0.05, n_estimators=50)
    model.fit(X)
    return model


# ── Preprocessing Tests ────────────────────────────────────────────────────

class TestPreprocessing:

    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)
        assert len(raw_df) > 0

    def test_expected_columns(self, raw_df):
        assert "Country.Name" in raw_df.columns
        assert "Year" in raw_df.columns
        assert "Emissions.Production.CO2.Total" in raw_df.columns

    def test_year_range(self, raw_df):
        assert raw_df["Year"].min() >= 1992
        assert raw_df["Year"].max() <= 2018

    def test_clean_removes_duplicates(self, raw_df):
        raw_with_dup = pd.concat([raw_df, raw_df.iloc[:5]])
        cleaned = clean(raw_with_dup)
        assert len(cleaned) <= len(raw_df)

    def test_clean_sorts_by_country_year(self, clean_df):
        first_country = clean_df["Country.Name"].iloc[0]
        first_year    = clean_df["Year"].iloc[0]
        assert first_year == clean_df[clean_df["Country.Name"] == first_country]["Year"].min()

    def test_feature_matrix_shape(self, feature_matrix):
        X, meta, _ = feature_matrix
        assert X.shape[0] == len(meta)
        assert X.shape[1] >= 9         # at least 9 emission features

    def test_no_nan_in_features(self, feature_matrix):
        X, _, _ = feature_matrix
        assert not np.isnan(X).any(), "Feature matrix contains NaN"

    def test_engineer_features_adds_columns(self, clean_df):
        df_eng = engineer_features(clean_df)
        assert "CO2_Total_YoY_pct" in df_eng.columns
        assert "Coal_fraction"     in df_eng.columns
        assert "emissions_per_gdp" in df_eng.columns


# ── Isolation Forest Tests ─────────────────────────────────────────────────

class TestIsolationForest:

    def test_fit_returns_self(self, feature_matrix):
        X, _, _ = feature_matrix
        model = IsolationForestDetector(n_estimators=10)
        result = model.fit(X)
        assert result is model

    def test_predict_returns_anomaly_result(self, fitted_isolation_forest, feature_matrix):
        X, _, _ = feature_matrix
        result  = fitted_isolation_forest.predict(X)
        assert isinstance(result, AnomalyResult)

    def test_labels_are_minus_one_or_one(self, fitted_isolation_forest, feature_matrix):
        X, _, _ = feature_matrix
        result  = fitted_isolation_forest.predict(X)
        assert set(result.labels).issubset({-1, 1})

    def test_contamination_rate(self, feature_matrix):
        X, _, _ = feature_matrix
        model   = IsolationForestDetector(contamination=0.05, n_estimators=50)
        model.fit(X)
        result  = model.predict(X)
        rate    = result.flags.sum() / len(result.flags)
        # Contamination should be ≈ 5% ± some tolerance
        assert 0.01 < rate < 0.15, f"Unexpected anomaly rate: {rate:.3f}"

    def test_flags_match_labels(self, fitted_isolation_forest, feature_matrix):
        X, _, _ = feature_matrix
        result  = fitted_isolation_forest.predict(X)
        assert np.all(result.flags == (result.labels == -1))

    def test_summary_has_correct_columns(self, fitted_isolation_forest, feature_matrix):
        X, meta, _ = feature_matrix
        result = fitted_isolation_forest.predict(X)
        df     = result.summary(meta)
        assert "anomaly_score" in df.columns
        assert "is_anomaly"    in df.columns
        assert "risk_level"    in df.columns

    def test_summary_only_contains_anomalies(self, fitted_isolation_forest, feature_matrix):
        X, meta, _ = feature_matrix
        result = fitted_isolation_forest.predict(X)
        df     = result.summary(meta)
        assert df["is_anomaly"].all()

    def test_summary_sorted_descending(self, fitted_isolation_forest, feature_matrix):
        X, meta, _ = feature_matrix
        result = fitted_isolation_forest.predict(X)
        df     = result.summary(meta)
        if len(df) > 1:
            assert (df["anomaly_score"].diff().dropna() <= 0).all()


# ── One-Class SVM Tests ────────────────────────────────────────────────────

class TestOneClassSVM:

    def test_basic_fit_predict(self, feature_matrix):
        X, meta, _ = feature_matrix
        model  = OneClassSVMDetector(nu=0.05)
        model.fit(X)
        result = model.predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.flags) == len(X)

    def test_different_scores_for_extremes(self, feature_matrix):
        X, _, _ = feature_matrix
        model   = OneClassSVMDetector(nu=0.05)
        model.fit(X)
        # Create obvious anomaly
        X_ext = np.vstack([X, X.max(axis=0) * 10])
        result = model.predict(X_ext)
        assert result.flags[-1], "Extreme outlier should be flagged"


# ── IQR Detector Tests ─────────────────────────────────────────────────────

class TestIQRDetector:

    def test_fit_stores_quartiles(self, feature_matrix):
        X, _, _ = feature_matrix
        model   = IQRDetector(k=2.5)
        model.fit(X)
        assert model.q1 is not None
        assert model.q3 is not None
        assert model.iqr is not None

    def test_obvious_outlier_flagged(self):
        X = np.array([[1, 2], [2, 3], [1.5, 2.5], [100, 200]])   # last = outlier
        model = IQRDetector(k=2.0)
        model.fit(X[:-1])
        result = model.predict(X)
        assert result.flags[-1], "100, 200 should be flagged as outlier"

    def test_normal_values_not_flagged(self):
        rng = np.random.RandomState(42)
        X   = rng.randn(200, 5)                  # all normal
        model = IQRDetector(k=3.0)
        model.fit(X)
        result = model.predict(X)
        # With k=3 and Gaussian data, very few should be flagged
        assert result.flags.sum() < 10


# ── Ensemble Tests ─────────────────────────────────────────────────────────

class TestEnsemble:

    def test_majority_vote(self, feature_matrix):
        X, _, _ = feature_matrix
        models  = [
            IsolationForestDetector(n_estimators=20).fit(X),
            OneClassSVMDetector().fit(X),
            IQRDetector().fit(X),
        ]
        results = [m.predict(X) for m in models]
        ensemble = ensemble_predict(results, vote_threshold=2)
        assert ensemble.dtype == bool
        assert len(ensemble) == len(X)

    def test_vote_threshold_1_equals_union(self, feature_matrix):
        X, _, _ = feature_matrix
        models  = [
            IsolationForestDetector(n_estimators=10).fit(X),
            IQRDetector().fit(X),
        ]
        results  = [m.predict(X) for m in models]
        ensemble = ensemble_predict(results, vote_threshold=1)
        union    = results[0].flags | results[1].flags
        np.testing.assert_array_equal(ensemble, union)

    def test_vote_threshold_n_equals_intersection(self, feature_matrix):
        X, _, _ = feature_matrix
        models  = [
            IsolationForestDetector(n_estimators=10).fit(X),
            IQRDetector().fit(X),
        ]
        results      = [m.predict(X) for m in models]
        ensemble     = ensemble_predict(results, vote_threshold=2)
        intersection = results[0].flags & results[1].flags
        np.testing.assert_array_equal(ensemble, intersection)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
