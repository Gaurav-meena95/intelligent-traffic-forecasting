"""Chronological validation and tuning for AML traffic models."""

from __future__ import annotations

import itertools
from typing import Any, Generator

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Pre-import torch if installed to avoid OpenMP runtime collision with xgboost on macOS
try:
    import torch
except Exception:
    pass

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None  # type: ignore
    HAS_XGBOOST = False


def chronological_cv_splits(
    n_samples: int,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Generate forward-chaining train and validation indices.

    Ensures that validation sets always succeed training sets,
    preventing temporal autocorrelation and sequence leakage.
    """
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if not 0 < min_train_fraction < 1:
        raise ValueError("min_train_fraction must be between 0 and 1.")

    min_train = int(n_samples * min_train_fraction)
    remaining = n_samples - min_train
    step = remaining // n_splits
    if step == 0:
        raise ValueError(
            "n_samples is too small for the requested number of splits."
        )

    for i in range(n_splits):
        train_end = min_train + i * step
        val_end = min_train + (i + 1) * step if i < n_splits - 1 else n_samples
        yield np.arange(0, train_end), np.arange(train_end, val_end)


def evaluate_masked_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE over positive finite observations."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    mask = np.isfinite(y_t) & (y_t > 0) & np.isfinite(y_p)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(y_t[mask] - y_p[mask])))


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform chronological grid search for Random Forest regressor."""
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of features.")
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows.")

    default_grid = {
        "n_estimators": [10, 25],
        "max_depth": [6, 10],
        "min_samples_leaf": [4, 8],
    }
    grid = param_grid if param_grid is not None else default_grid
    if not grid or any(len(v) == 0 for v in grid.values()):
        raise ValueError(
            "param_grid must not be empty and all parameter value lists must be non-empty."
        )

    keys = list(grid.keys())
    combinations = [
        dict(zip(keys, prod)) for prod in itertools.product(*grid.values())
    ]

    best_score = float("inf")
    best_params = combinations[0]
    results = []

    splits = list(
        chronological_cv_splits(
            len(X_arr), n_splits=n_splits, min_train_fraction=min_train_fraction
        )
    )

    for params in combinations:
        fold_scores = []
        for train_idx, val_idx in splits:
            model = RandomForestRegressor(
                n_jobs=n_jobs,
                random_state=random_state,
                **params,  # type: ignore
            )
            model.fit(X_arr[train_idx], y_arr[train_idx])
            preds = model.predict(X_arr[val_idx])
            score = evaluate_masked_mae(y_arr[val_idx], preds)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        results.append(
            {"params": params, "mean_mae": mean_score, "folds": fold_scores}
        )
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return {
        "best_params": best_params,
        "best_mae": best_score,
        "results": results,
    }


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    n_jobs: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform chronological grid search for XGBoost regressor."""
    if not HAS_XGBOOST or XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed or its native C++ runtime (e.g. OpenMP/libomp) could not be loaded."
        )

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of features.")
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows.")

    default_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    grid = param_grid if param_grid is not None else default_grid
    if not grid or any(len(v) == 0 for v in grid.values()):
        raise ValueError(
            "param_grid must not be empty and all parameter value lists must be non-empty."
        )

    keys = list(grid.keys())
    combinations = [
        dict(zip(keys, prod)) for prod in itertools.product(*grid.values())
    ]

    best_score = float("inf")
    best_params = combinations[0]
    results = []

    splits = list(
        chronological_cv_splits(
            len(X_arr), n_splits=n_splits, min_train_fraction=min_train_fraction
        )
    )

    for params in combinations:
        fold_scores = []
        for train_idx, val_idx in splits:
            model = XGBRegressor(
                n_jobs=n_jobs,
                random_state=random_state,
                **params,
            )
            model.fit(X_arr[train_idx], y_arr[train_idx])
            preds = model.predict(X_arr[val_idx])
            score = evaluate_masked_mae(y_arr[val_idx], preds)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        results.append(
            {"params": params, "mean_mae": mean_score, "folds": fold_scores}
        )
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return {
        "best_params": best_params,
        "best_mae": best_score,
        "results": results,
    }


def tune_randomized_search(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    param_distributions: dict[str, Any] | None = None,
    n_iter: int = 5,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform randomized forward-chaining hyperparameter search.

    Samples parameter configurations across continuous/discrete distributions
    while enforcing strict chronological cross-validation.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of features.")
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    model_type_clean = model_type.strip().lower()
    if model_type_clean not in ("random_forest", "xgboost"):
        raise ValueError("model_type must be either 'random_forest' or 'xgboost'.")

    rng = np.random.default_rng(random_state)

    if model_type_clean == "random_forest":
        default_dists: dict[str, list[Any]] = {
            "n_estimators": [10, 25, 50],
            "max_depth": [4, 6, 8, 12],
            "min_samples_leaf": [2, 4, 8],
        }
    else:
        if not HAS_XGBOOST or XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed or its native C++ runtime could not be loaded."
            )
        default_dists = {
            "n_estimators": [25, 50, 100],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1, 0.15],
            "subsample": [0.7, 0.85, 1.0],
        }

    dists = param_distributions if param_distributions is not None else default_dists
    if not dists or any(len(v) == 0 for v in dists.values()):
        raise ValueError("param_distributions must be non-empty.")

    # Sample n_iter unique combinations
    candidates: list[dict[str, Any]] = []
    keys = list(dists.keys())
    seen: set[tuple[Any, ...]] = set()
    max_attempts = n_iter * 20
    attempts = 0

    while len(candidates) < n_iter and attempts < max_attempts:
        attempts += 1
        sampled = {k: rng.choice(dists[k]) for k in keys}
        # Cast standard numpy scalars to python types
        for k, v in sampled.items():
            if hasattr(v, "item"):
                sampled[k] = v.item()
        signature = tuple(sampled[k] for k in keys)
        if signature not in seen:
            seen.add(signature)
            candidates.append(sampled)

    if not candidates:
        candidates = [{k: dists[k][0] for k in keys}]

    splits = list(
        chronological_cv_splits(
            len(X_arr), n_splits=n_splits, min_train_fraction=min_train_fraction
        )
    )

    best_score = float("inf")
    best_params = candidates[0]
    results = []

    for params in candidates:
        fold_scores = []
        for train_idx, val_idx in splits:
            if model_type_clean == "random_forest":
                model: Any = RandomForestRegressor(
                    n_jobs=n_jobs,
                    random_state=random_state,
                    **params,
                )
            else:
                model = XGBRegressor(
                    n_jobs=1 if n_jobs < 0 else n_jobs,
                    random_state=random_state,
                    **params,
                )
            model.fit(X_arr[train_idx], y_arr[train_idx])
            preds = model.predict(X_arr[val_idx])
            score = evaluate_masked_mae(y_arr[val_idx], preds)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        results.append(
            {"params": params, "mean_mae": mean_score, "folds": fold_scores}
        )
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return {
        "best_params": best_params,
        "best_mae": best_score,
        "results": results,
    }

