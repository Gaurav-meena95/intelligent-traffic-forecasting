#!/usr/bin/env python3
"""Run automated forward-chaining hyperparameter tuning for traffic models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traffic_prediction.data import HORIZON_STEPS, load_speed_csv, prepare_speed_inputs
from traffic_prediction.features import make_sensor_features, time_features
from traffic_prediction.tuning import tune_random_forest, tune_randomized_search, tune_xgboost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chronological hyperparameter tuning for traffic models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/METR-LA.csv",
        help="Path to the traffic speed observations CSV file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost"],
        help="Model type to tune.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="random",
        choices=["grid", "random"],
        help="Search strategy: 'grid' or 'random'.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5,
        help="Number of iterations for random search.",
    )
    parser.add_argument(
        "--sensor-index",
        type=int,
        default=0,
        help="Sensor column index to use for local tuning.",
    )
    parser.add_argument(
        "--horizon-step",
        type=int,
        default=HORIZON_STEPS[1],  # 15m (3 steps)
        help="Horizon step offset (1=5m, 3=15m, 6=30m, 12=60m).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to save JSON tuning results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading dataset from {data_path}...")
    raw = load_speed_csv(data_path)
    speeds = prepare_speed_inputs(raw)
    clock = time_features(raw.index)

    max_lag = 5
    lead = args.horizon_step
    valid_times = np.arange(max_lag, len(raw) - lead)
    X = make_sensor_features(speeds, clock, valid_times, args.sensor_index)
    y = raw.iloc[valid_times + lead, args.sensor_index].to_numpy(dtype=np.float32)

    # Use first 30% of timeline for fast prototype tuning
    subset_len = min(len(valid_times), 3000)
    X_sub = X[:subset_len]
    y_sub = y[:subset_len]

    print(f"Starting {args.method} search for {args.model_type} (samples={subset_len})...")
    if args.method == "random":
        results = tune_randomized_search(
            X_sub,
            y_sub,
            model_type=args.model_type,
            n_iter=args.n_iter,
            n_splits=3,
        )
    elif args.model_type == "random_forest":
        results = tune_random_forest(X_sub, y_sub, n_splits=3)
    else:
        results = tune_xgboost(X_sub, y_sub, n_splits=3)

    print(f"Tuning complete!")
    print(f"Best MAE: {results['best_mae']:.4f}")
    print(f"Best Parameters: {results['best_params']}")

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
