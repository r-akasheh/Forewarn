from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from safety.forewarn.examples.vlm_api_models import encode_payload
from safety.forewarn.examples.vlm_backend import RemoteVLMBackend


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    posted_json = None

    def __init__(self, *args, **kwargs):
        self.last_json = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, endpoint, json):
        assert endpoint.endswith("/infer-two-stage")
        self.last_json = json
        _FakeClient.posted_json = json
        pred_frames = {
            "pred_cam_rs": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        }
        return _FakeResponse(
            {
                "predictions": ["center"],
                "text_input": "input",
                "predictions_second_stage": ["1"],
                "pred_frames": encode_payload(pred_frames),
                "backend": "local",
                "error": None,
            }
        )


def test_remote_backend_decodes_frames():
    backend = RemoteVLMBackend("http://localhost:8010", timeout_s=1.0, max_retries=0)

    with patch("safety.forewarn.examples.vlm_backend.httpx.Client", _FakeClient):
        predictions, text_input, predictions_second_stage, pred_frames = backend.infer_two_stage(
            obs={"state": torch.zeros(4, dtype=torch.float32)},
            action_seq=torch.zeros((1, 2, 4), dtype=torch.float32),
            normalize=True,
            question_key="grasping",
        )

    assert predictions == ["center"]
    assert text_input == "input"
    assert predictions_second_stage == ["1"]
    assert isinstance(pred_frames["pred_cam_rs"], np.ndarray)
    assert pred_frames["pred_cam_rs"].shape == (1, 2, 4, 4, 3)
    assert isinstance(_FakeClient.posted_json["obs"]["state"], dict)
    assert _FakeClient.posted_json["obs"]["state"].get("__ndarray__") is True

