"""Tests for model artifact serialization and export."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from traffic_prediction.export import load_model_artifact, save_model_artifact


def test_save_and_load_rf(tmp_path: Path):
    X = np.random.randn(50, 4).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X, y)
    orig_preds = rf.predict(X[:5])

    artifact_path = tmp_path / "test_rf.joblib"
    meta = {"horizon": 15, "author": "Mausam"}
    saved = save_model_artifact(rf, artifact_path, metadata=meta)
    assert saved.exists()

    loaded_model, loaded_meta = load_model_artifact(artifact_path)
    loaded_preds = loaded_model.predict(X[:5])
    np.testing.assert_allclose(orig_preds, loaded_preds)
    assert loaded_meta.get("horizon") == 15
    assert loaded_meta.get("author") == "Mausam"


def test_load_nonexistent(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model_artifact(tmp_path / "nonexistent.joblib")
