"""Tests for the Residual Spatial Hybrid architecture."""

import numpy as np
import pytest
import torch

from traffic_prediction.hybrid import ResidualSpatialGAT, ResidualSpatialHybrid


def test_residual_spatial_gat_shapes():
    n_sensors = 4
    batch_size = 8
    input_size = 10
    output_size = 4

    adj = torch.eye(n_sensors, dtype=torch.float32)
    adj[0, 1] = 0.5
    adj[1, 0] = 0.5

    gat = ResidualSpatialGAT(adjacency=adj, input_size=input_size, hidden_size=16, output_size=output_size)
    x = torch.randn(batch_size, n_sensors, input_size)
    out = gat(x)

    assert out.shape == (batch_size, n_sensors, output_size)


def test_residual_spatial_hybrid_fit_and_predict():
    n_sensors = 3
    n_samples = 16
    input_size = 10
    output_size = 4

    adj = np.eye(n_sensors, dtype=np.float32)
    adj[0, 1] = 0.4
    adj[1, 0] = 0.4

    hybrid = ResidualSpatialHybrid(
        adjacency=adj,
        input_size=input_size,
        hidden_size=16,
        output_size=output_size,
    )

    X_graph = np.random.randn(n_samples, n_sensors, input_size).astype(np.float32)
    residuals = np.random.randn(n_samples, n_sensors, output_size).astype(np.float32) * 2.0
    mask = np.ones((n_samples, n_sensors, output_size), dtype=np.float32)

    # Train residuals
    losses = hybrid.fit_residuals(X_graph, residuals, mask=mask, epochs=2, batch_size=8)
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)

    # Predict combined
    base_preds = np.full((n_samples, n_sensors, output_size), 50.0, dtype=np.float32)
    final_preds = hybrid.predict(base_preds, X_graph)

    assert final_preds.shape == (n_samples, n_sensors, output_size)
    assert not np.allclose(final_preds, base_preds)  # GAT should add non-zero residual correction
