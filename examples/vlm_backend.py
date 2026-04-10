from __future__ import annotations

import time
from typing import Optional

import httpx

try:
    from .vlm_api_models import (
        InferTwoStageRequest,
        InferTwoStageResponse,
        decode_payload,
        encode_payload,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from vlm_api_models import (
        InferTwoStageRequest,
        InferTwoStageResponse,
        decode_payload,
        encode_payload,
    )


class LocalVLMBackend:
    """Local wrapper around VLMInference with a unified remote/local interface."""

    def __init__(
        self,
        wm_configs,
        model_name: Optional[str] = None,
        peft_model: Optional[str] = None,
        answer_type: str = "snippet",
    ):
        try:
            from .wm_pred_fork import VLMInference
        except ImportError:  # pragma: no cover - direct script fallback
            from wm_pred_fork import VLMInference

        self._inference = VLMInference(
            wm_configs=wm_configs,
            model_name=model_name,
            peft_model=peft_model,
            answer_type=answer_type,
        )

    def infer_two_stage(self, obs, action_seq, normalize: bool = False, question_key: str = "grasping"):
        return self._inference.infer_two_stage(
            obs,
            action_seq,
            normalize=normalize,
            question_key=question_key,
        )


class RemoteVLMBackend:
    """HTTP client that calls a remote FastAPI VLM service."""

    def __init__(
        self,
        server_url: str,
        timeout_s: float = 45.0,
        max_retries: int = 1,
        include_pred_frames: bool = True,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.include_pred_frames = bool(include_pred_frames)

    def infer_two_stage(self, obs, action_seq, normalize: bool = False, question_key: str = "grasping"):
        request = InferTwoStageRequest(
            obs=encode_payload(obs),
            action_seq=encode_payload(action_seq),
            normalize=normalize,
            question_key=question_key,
            include_pred_frames=self.include_pred_frames,
        )

        endpoint = f"{self.server_url}/infer-two-stage"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(endpoint, json=request.model_dump())
                    response.raise_for_status()
                payload = InferTwoStageResponse.model_validate(response.json())
                if payload.error:
                    raise RuntimeError(payload.error)
                pred_frames = decode_payload(payload.pred_frames)
                return (
                    payload.predictions,
                    payload.text_input,
                    payload.predictions_second_stage,
                    pred_frames,
                )
            except Exception as exc:  # pragma: no cover - runtime network failures
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break

        raise RuntimeError(
            f"Remote VLM inference failed after {self.max_retries + 1} attempts: {last_error}"
        )

