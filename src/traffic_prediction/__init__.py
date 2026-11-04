"""Reusable components for traffic-speed forecasting."""

from typing import TYPE_CHECKING

from traffic_prediction.data import (
    HORIZON_STEPS,
    TimeSplit,
    load_speed_csv,
    observed_speeds,
    prepare_speed_inputs,
    time_split,
)
from traffic_prediction.features import (
    make_enhanced_sensor_features,
    make_panel_features,
    make_rolling_features,
    make_sensor_features,
    make_spatial_lag_features,
    time_features,
)
from traffic_prediction.graph import (
    build_directed_distance_adjacency,
    build_distance_adjacency,
    compute_dual_diffusion_matrices,
)
from traffic_prediction.hybrid import ResidualSpatialGAT, ResidualSpatialHybrid
from traffic_prediction.metrics import peak_mask, regression_metrics
from traffic_prediction.tuning import (
    chronological_cv_splits,
    evaluate_masked_mae,
    tune_random_forest,
    tune_randomized_search,
    tune_xgboost,
)

if TYPE_CHECKING:
    from traffic_prediction.models import (
        GraphAttentionLayer,
        GraphAttentionNetwork,
    )

__all__ = [
    "HORIZON_STEPS",
    "GraphAttentionLayer",
    "GraphAttentionNetwork",
    "ResidualSpatialGAT",
    "ResidualSpatialHybrid",
    "TimeSplit",
    "build_directed_distance_adjacency",
    "build_distance_adjacency",
    "chronological_cv_splits",
    "compute_dual_diffusion_matrices",
    "evaluate_masked_mae",
    "load_speed_csv",
    "make_enhanced_sensor_features",
    "make_panel_features",
    "make_rolling_features",
    "make_sensor_features",
    "make_spatial_lag_features",
    "observed_speeds",
    "peak_mask",
    "prepare_speed_inputs",
    "regression_metrics",
    "time_features",
    "time_split",
    "tune_random_forest",
    "tune_randomized_search",
    "tune_xgboost",
]


def __getattr__(name: str) -> object:
    if name in ("GraphAttentionNetwork", "GraphAttentionLayer"):
        import traffic_prediction.models as models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
