"""Model artifact serialization and production export utilities."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import joblib

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


def save_model_artifact(
    model: Any,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialize and export a model alongside descriptive metadata.

    Supports PyTorch nn.Module architectures and Scikit-Learn/XGBoost regressors.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata.copy() if metadata is not None else {}
    meta.setdefault("exported_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
    meta.setdefault("model_class", model.__class__.__name__)

    if HAS_TORCH and isinstance(model, nn.Module):
        payload = {
            "state_dict": model.state_dict(),
            "metadata": meta,
        }
        torch.save(payload, path)
    else:
        payload = {
            "model": model,
            "metadata": meta,
        }
        joblib.dump(payload, path)

    return path


def load_model_artifact(
    filepath: str | Path,
    model_class: type | None = None,
    **init_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load a serialized model artifact and return (model, metadata).

    If model_class is provided for PyTorch modules, instantiates the architecture
    and loads the trained state_dict.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")

    # Try PyTorch loader first if file extension suggests torch or torch is available
    if HAS_TORCH and (path.suffix in (".pt", ".pth") or model_class is not None):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "state_dict" in payload:
                metadata = payload.get("metadata", {})
                if model_class is not None:
                    model = model_class(**init_kwargs)
                    model.load_state_dict(payload["state_dict"])
                    model.eval()
                    return model, metadata
                return payload["state_dict"], metadata
        except Exception:
            pass

    # Fallback to joblib
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})
    return payload, {}
