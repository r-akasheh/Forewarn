from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException

try:
    from .vlm_api_models import (
        InferTwoStageRequest,
        InferTwoStageResponse,
        decode_payload,
        encode_payload,
    )
    from .vlm_backend import LocalVLMBackend
except ImportError:  # pragma: no cover - direct script fallback
    from vlm_api_models import (
        InferTwoStageRequest,
        InferTwoStageResponse,
        decode_payload,
        encode_payload,
    )
    from vlm_backend import LocalVLMBackend


def _pin_cuda_visible_devices(cuda_visible_devices: str | None) -> None:
    """Restrict this process to a specific CUDA device set (e.g. "0")."""
    if cuda_visible_devices is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)


def _log_cuda_runtime_info() -> None:
    """Print CUDA runtime details for quick startup verification."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[VLM Server] CUDA_VISIBLE_DEVICES={visible}")
    try:
        import torch

        is_available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if is_available else 0
        print(f"[VLM Server] torch.cuda.is_available={is_available}, device_count={count}")
        if count > 0:
            name = torch.cuda.get_device_name(0)
            print(f"[VLM Server] active device index=0 name={name}")
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        print(f"[VLM Server] CUDA diagnostics unavailable: {exc}")


def _load_wm_config(config_path: str) -> Dict[str, Any]:
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"WM config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if "defaults" not in data:
        data = {"defaults": data}
    return data


def create_app(
    config_path: str,
    model_name: str,
    peft_model: str | None,
    answer_type: str = "snippet",
    cuda_visible_devices: str | None = "0",
) -> FastAPI:
    state: Dict[str, Any] = {"backend": None, "ready": False, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            _pin_cuda_visible_devices(cuda_visible_devices)
            wm_config = _load_wm_config(config_path)
            state["backend"] = LocalVLMBackend(
                wm_configs=wm_config,
                model_name=model_name,
                peft_model=peft_model,
                answer_type=answer_type,
            )
            state["ready"] = True
        except Exception as exc:  # pragma: no cover - startup failures are runtime dependent
            state["error"] = str(exc)
            state["ready"] = False
        yield

    app = FastAPI(title="Forewarn VLM Remote Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if state["ready"] else "error",
            "ready": state["ready"],
            "error": state["error"],
        }

    @app.post("/infer-two-stage", response_model=InferTwoStageResponse)
    async def infer_two_stage(request: InferTwoStageRequest):
        backend = state["backend"]
        if backend is None:
            raise HTTPException(status_code=503, detail=state["error"] or "VLM backend not ready")

        try:
            obs = decode_payload(request.obs)
            action_seq = decode_payload(request.action_seq)
            predictions, text_input, predictions_second_stage, pred_frames = backend.infer_two_stage(
                obs,
                action_seq,
                normalize=request.normalize,
                question_key=request.question_key,
            )
            encoded_frames = encode_payload(pred_frames) if request.include_pred_frames else None
            return InferTwoStageResponse(
                predictions=predictions,
                text_input=text_input,
                predictions_second_stage=predictions_second_stage,
                pred_frames=encoded_frames,
                backend="local",
            )
        except Exception as exc:  # pragma: no cover - model/runtime failures
            return InferTwoStageResponse(error=str(exc), backend="local")

    return app


def main():
    parser = argparse.ArgumentParser(description="Run the remote VLM FastAPI service")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(
        os.path.join(script_dir, "..", "configs", "wm_example_config_48d_state_only.yaml")
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument(
        "--model-name",
        type=str,
        default="/data/mllama/Llama-3.2-11B-Vision-Instruct/custom",
        help="Path to base VLM model",
    )
    parser.add_argument("--peft-model", type=str, default=None, help="Path to VLM PEFT checkpoint")
    parser.add_argument("--answer-type", type=str, default="snippet")
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0",
        help="Value for CUDA_VISIBLE_DEVICES (default: 0)",
    )
    args = parser.parse_args()

    _pin_cuda_visible_devices(args.cuda_visible_devices)
    _log_cuda_runtime_info()
    app = create_app(
        config_path=args.config,
        model_name=args.model_name,
        peft_model=args.peft_model,
        answer_type=args.answer_type,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

