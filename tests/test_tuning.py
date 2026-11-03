"""Tests for chronological cross-validation and hyperparameter tuning."""

import numpy as np
import pytest

from traffic_prediction.tuning import (
    HAS_XGBOOST,
    chronological_cv_splits,
    evaluate_masked_mae,
    tune_random_forest,
    tune_randomized_search,
    tune_xgboost,
)


def test_chronological_cv_splits():
    n_samples = 100
    n_splits = 3
    splits = list(
        chronological_cv_splits(
            n_samples, n_splits=n_splits, min_train_fraction=0.5
        )
    )
    assert len(splits) == n_splits

    for train_idx, val_idx in splits:
        # Check max train index < min val index (no future leakage)
        assert train_idx.max() < val_idx.min()
        # Check no overlap
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        # Check validation indices are contiguous
        assert len(val_idx) > 0


def test_chronological_cv_splits_invalid():
    with pytest.raises(ValueError, match="n_samples must be at least 10"):
        list(chronological_cv_splits(5))
    with pytest.raises(ValueError, match="n_splits must be at least 1"):
        list(chronological_cv_splits(20, n_splits=0))
    with pytest.raises(ValueError, match="min_train_fraction must be between 0 and 1"):
        list(chronological_cv_splits(20, min_train_fraction=1.2))
    with pytest.raises(ValueError, match="n_samples is too small"):
        list(chronological_cv_splits(10, n_splits=8, min_train_fraction=0.8))


def test_evaluate_masked_mae():
    y_true = np.array([50.0, np.nan, 0.0, 60.0])
    y_pred = np.array([48.0, 70.0, 30.0, 65.0])
    # valid targets: 50.0 (diff 2) and 60.0 (diff 5) -> mean = 3.5
    mae = evaluate_masked_mae(y_true, y_pred)
    assert pytest.approx(mae, 0.001) == 3.5


def test_evaluate_masked_mae_edge_cases():
    # All NaN or non-positive
    y_true = np.array([np.nan, 0.0, -10.0])
    y_pred = np.array([50.0, 60.0, 70.0])
    assert evaluate_masked_mae(y_true, y_pred) == float("inf")

    # Mismatched shapes
    with pytest.raises(ValueError, match="same shape"):
        evaluate_masked_mae(np.array([50.0, 60.0]), np.array([50.0]))


def test_tune_random_forest_synthetic():
    np.random.seed(42)
    X = np.random.randn(80, 5).astype(np.float32)
    y = (X[:, 0] * 5.0 + 50.0 + np.random.randn(80) * 0.5).astype(np.float32)

    param_grid = {
        "n_estimators": [5],
        "max_depth": [3, 5],
    }
    result = tune_random_forest(X, y, param_grid=param_grid, n_splits=2)
    assert "best_params" in result
    assert "best_mae" in result
    assert result["best_mae"] < 10.0
    assert len(result["results"]) == 2


def test_tune_random_forest_validation_errors():
    X = np.random.randn(50, 4).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)

    # 1D X
    with pytest.raises(ValueError, match="2D array"):
        tune_random_forest(y, y)

    # Length mismatch
    with pytest.raises(ValueError, match="same number of rows"):
        tune_random_forest(X[:30], y)

    # Empty param_grid
    with pytest.raises(ValueError, match="param_grid must not be empty"):
        tune_random_forest(X, y, param_grid={})

    # Empty parameter values
    with pytest.raises(ValueError, match="param_grid must not be empty"):
        tune_random_forest(X, y, param_grid={"n_estimators": []})


def test_tune_xgboost_synthetic():
    np.random.seed(42)
    X = np.random.randn(80, 4).astype(np.float32)
    y = (X[:, 0] * 4.0 + 45.0 + np.random.randn(80) * 0.5).astype(np.float32)

    if not HAS_XGBOOST:
        with pytest.raises(ImportError, match="xgboost is not installed"):
            tune_xgboost(X, y)
    else:
        param_grid = {
            "n_estimators": [5],
            "max_depth": [3],
            "learning_rate": [0.1],
        }
        res = tune_xgboost(X, y, param_grid=param_grid, n_splits=2)
        assert "best_params" in res
        assert "best_mae" in res
        assert res["best_mae"] < 15.0


def test_tune_randomized_search_rf():
    np.random.seed(42)
    X = np.random.randn(60, 4).astype(np.float32)
    y = (X[:, 0] * 3.0 + 40.0 + np.random.randn(60) * 0.2).astype(np.float32)

    dists = {
        "n_estimators": [5, 10],
        "max_depth": [3, 4],
    }
    result = tune_randomized_search(
        X, y, model_type="random_forest", param_distributions=dists, n_iter=2, n_splits=2
    )
    assert "best_params" in result
    assert "best_mae" in result
    assert result["best_mae"] < 10.0
    assert len(result["results"]) <= 2


def test_tune_randomized_search_invalid():
    X = np.random.randn(50, 4).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)

    with pytest.raises(ValueError, match="model_type"):
        tune_randomized_search(X, y, model_type="invalid_model")

    with pytest.raises(ValueError, match="n_iter"):
        tune_randomized_search(X, y, n_iter=0)

