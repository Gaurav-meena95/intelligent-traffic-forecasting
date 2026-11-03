#!/usr/bin/env python3
"""Export trained traffic models to serialized artifacts for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from traffic_prediction.data import HORIZON_STEPS, load_speed_csv, prepare_speed_inputs
from traffic_prediction.export import save_model_artifact
from traffic_prediction.features import make_sensor_features, time_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained models to serialized artifacts.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/METR-LA.csv",
        help="Path to traffic speed CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/models",
        help="Directory where serialized models will be saved.",
    )
    parser.add_argument(
        "--sensor-index",
        type=int,
        default=0,
        help="Sensor index to train and export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    raw = load_speed_csv(data_path)
    speeds = prepare_speed_inputs(raw)
    clock = time_features(raw.index)

    max_lag = 5
    lead = HORIZON_STEPS[1]  # 15m
    valid_times = np.arange(max_lag, len(raw) - lead)
    X = make_sensor_features(speeds, clock, valid_times, args.sensor_index)
    y = raw.iloc[valid_times + lead, args.sensor_index].to_numpy(dtype=np.float32)

    # Train on first 70% (chronological training set)
    train_end = int(len(valid_times) * 0.70)
    X_train, y_train = X[:train_end], y[:train_end]

    print(f"Training Random Forest model for sensor {args.sensor_index} ({len(X_train)} samples)...")
    rf = RandomForestRegressor(n_estimators=25, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    artifact_name = f"rf_sensor_{args.sensor_index}_15m.joblib"
    artifact_path = out_dir / artifact_name
    metadata = {
        "sensor_index": args.sensor_index,
        "sensor_id": str(raw.columns[args.sensor_index]),
        "horizon_minutes": 15,
        "train_samples": len(X_train),
        "model_type": "RandomForestRegressor",
    }

    save_model_artifact(rf, artifact_path, metadata=metadata)
    print(f"Model successfully exported to: {artifact_path}")


if __name__ == "__main__":
    main()
