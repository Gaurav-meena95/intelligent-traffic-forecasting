"""Residual Spatial Hybrid Architecture combining AML tree models with Graph Attention."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from traffic_prediction.models import GraphAttentionNetwork


class ResidualSpatialGAT(nn.Module):
    """Deep residual graph network trained to predict spatial error residuals."""

    def __init__(
        self,
        adjacency: torch.Tensor,
        input_size: int = 10,
        hidden_size: int = 32,
        output_size: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gat = GraphAttentionNetwork(
            adjacency=adjacency,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Predicts continuous error residual corrections across all nodes and horizons
        return self.gat(features)


class ResidualSpatialHybrid:
    """Two-stage hybrid architecture:

    Stage 1: Base localized AML model (e.g. Random Forest / XGBoost) predicts base speed.
    Stage 2: Graph Attention Network models propagating spatial error residuals.
    """

    def __init__(
        self,
        base_model: Any | None = None,
        adjacency: np.ndarray | torch.Tensor | None = None,
        input_size: int = 10,
        hidden_size: int = 32,
        output_size: int = 4,
        device: str = "cpu",
    ) -> None:
        self.base_model = base_model
        self.device = torch.device(device)
        self.output_size = output_size
        self.input_size = input_size

        if adjacency is not None:
            if isinstance(adjacency, np.ndarray):
                adj_tensor = torch.from_numpy(adjacency.astype(np.float32))
            else:
                adj_tensor = adjacency.float()
            self.gat_compensator = ResidualSpatialGAT(
                adjacency=adj_tensor.to(self.device),
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
            ).to(self.device)
        else:
            self.gat_compensator = None

    def fit_residuals(
        self,
        graph_features: torch.Tensor | np.ndarray,
        residuals: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray | None = None,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> list[float]:
        """Train the graph compensator on observed error residuals."""
        if self.gat_compensator is None:
            raise ValueError("GAT compensator must be initialized with an adjacency matrix.")

        if isinstance(graph_features, np.ndarray):
            X_tensor = torch.from_numpy(graph_features.astype(np.float32)).to(self.device)
        else:
            X_tensor = graph_features.float().to(self.device)

        if isinstance(residuals, np.ndarray):
            R_tensor = torch.from_numpy(residuals.astype(np.float32)).to(self.device)
        else:
            R_tensor = residuals.float().to(self.device)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                M_tensor = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            else:
                M_tensor = mask.float().to(self.device)
        else:
            M_tensor = torch.ones_like(R_tensor)

        self.gat_compensator.train()
        optimizer = torch.optim.Adam(self.gat_compensator.parameters(), lr=lr)

        n_samples = len(X_tensor)
        epoch_losses: list[float] = []

        for _ in range(epochs):
            permutation = torch.randperm(n_samples)
            batch_losses = []
            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                b_X = X_tensor[indices]
                b_R = R_tensor[indices]
                b_M = M_tensor[indices]

                optimizer.zero_grad()
                pred_R = self.gat_compensator(b_X)

                # Masked MSE loss on residuals
                diff_sq = (pred_R - b_R) ** 2
                valid_mask = b_M * (torch.isfinite(b_R).float())
                loss = (diff_sq * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(float(np.mean(batch_losses)))

        self.gat_compensator.eval()
        return epoch_losses

    def predict(
        self,
        base_predictions: np.ndarray | torch.Tensor,
        graph_features: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Combine base AML predictions with graph spatial residual corrections."""
        if isinstance(base_predictions, torch.Tensor):
            base_arr = base_predictions.detach().cpu().numpy()
        else:
            base_arr = np.asarray(base_predictions, dtype=np.float32)

        if self.gat_compensator is not None:
            if isinstance(graph_features, np.ndarray):
                X_tensor = torch.from_numpy(graph_features.astype(np.float32)).to(self.device)
            else:
                X_tensor = graph_features.float().to(self.device)

            self.gat_compensator.eval()
            with torch.no_grad():
                pred_residuals = self.gat_compensator(X_tensor).detach().cpu().numpy()
            return (base_arr + pred_residuals).astype(np.float32)

        return base_arr
