from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


_NDARRAY_KEY = "__ndarray__"
_DTYPE_KEY = "dtype"
_SHAPE_KEY = "shape"
_DATA_KEY = "data"


def encode_payload(value: Any) -> Any:
    """Convert numpy-heavy payloads into JSON-serializable objects."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()

    if isinstance(value, np.ndarray):
        return {
            _NDARRAY_KEY: True,
            _DTYPE_KEY: str(value.dtype),
            _SHAPE_KEY: list(value.shape),
            _DATA_KEY: value.tolist(),
        }
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): encode_payload(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_payload(item) for item in value]
    return value


def _decode_ndarray(payload: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(payload[_DATA_KEY], dtype=np.dtype(payload[_DTYPE_KEY]))
    shape = tuple(payload.get(_SHAPE_KEY, arr.shape))
    if shape != arr.shape:
        arr = arr.reshape(shape)
    return arr


def decode_payload(value: Any) -> Any:
    """Restore encoded numpy values from JSON payloads."""
    if isinstance(value, dict):
        if value.get(_NDARRAY_KEY):
            return _decode_ndarray(value)
        return {key: decode_payload(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [decode_payload(item) for item in value]
    return value


class InferTwoStageRequest(BaseModel):
    obs: Any
    action_seq: Any
    normalize: bool = False
    question_key: str = "grasping"
    include_pred_frames: bool = True


class InferTwoStageResponse(BaseModel):
    predictions: List[str] = Field(default_factory=list)
    text_input: str = ""
    predictions_second_stage: List[str] = Field(default_factory=list)
    pred_frames: Optional[Any] = None
    backend: str = "local"
    error: Optional[str] = None

