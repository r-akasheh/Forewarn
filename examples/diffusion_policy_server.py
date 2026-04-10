"""FastAPI server for diffusion policy inference.

Run in diffusion conda environment:
  conda activate diffusion
  python diffusion_policy_server.py \
    --policy-ckpt /path/to/checkpoint.pt \
    --obs-dim 48 \
    --train-script /path/to/inference.py \
    --port 9001

Then call from parent process via HTTP:
  POST http://localhost:9001/infer
  {"obs_batch": [[...], [...], ...]}
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

os.environ["SAPIEN_RENDERER"] = "egl"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# Request/Response models
# -------------------------
class InferRequest(BaseModel):
    obs_batch: list  # Will be converted to numpy array


class InferResponse(BaseModel):
    actions: list
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_ready: bool


# -------------------------
# Global agent state
# -------------------------
app = FastAPI(title="Diffusion Policy Inference")
AGENT = None
DEVICE = None


# -------------------------
# Initialization function
# -------------------------
def load_agent(
    train_script: str,
    policy_ckpt: str,
    obs_dim: int,
    action_dim: int = 4,
    obs_horizon: int = 2,
    act_horizon: int = 8,
    pred_horizon: int = 16,
) -> tuple:
    """Load the agent and return (agent, device, obs_dim)."""
    global AGENT, DEVICE

    logger.info(f"Loading module from: {train_script}")
    train_path = Path(train_script)
    module_name = "diffusion_policy_inference"

    spec = importlib.util.spec_from_file_location(module_name, str(train_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load: {train_path}")

    module = importlib.util.module_from_spec(spec)

    # Add the directory containing the inference script to sys.path
    inference_dir = str(train_path.parent)
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)

    logger.info("Executing module...")
    spec.loader.exec_module(module)

    logger.info("Module loaded. Extracting Agent and Args...")
    Agent = module.Agent
    ArgsCls = module.Args

    logger.info("Building policy args...")
    dp_args = ArgsCls(
        env_id="StackCube-v1",
        obs_horizon=obs_horizon,
        act_horizon=act_horizon,
        pred_horizon=pred_horizon,
    )

    logger.info("Creating environment shim...")
    import gymnasium as gym

    shim = type("EnvShim", (), {})()
    shim.single_observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_horizon, obs_dim),
        dtype=np.float32,
    )
    shim.single_action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        dtype=np.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating agent...")
    agent = Agent(shim, dp_args).to(device)

    logger.info(f"Loading checkpoint from {policy_ckpt}...")
    checkpoint = torch.load(policy_ckpt, map_location=device)
    state_dict = (
        checkpoint.get("ema_agent")
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    logger.info("Loading state dict...")
    agent.load_state_dict(state_dict)
    agent.eval()
    logger.info("Agent ready.")

    AGENT = agent
    DEVICE = device
    return agent, device, obs_dim


# -------------------------
# API endpoints
# -------------------------
@app.on_event("startup")
async def startup():
    """Called when server starts."""
    pass  # Agent is loaded in __main__


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ready",
        device=str(DEVICE),
        model_ready=AGENT is not None,
    )


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """Run inference on observation batch."""
    if AGENT is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not loaded. Server may still be initializing."
        )

    try:
        obs_batch = np.asarray(request.obs_batch, dtype=np.float32)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_batch).to(DEVICE)
            actions = AGENT.get_action(obs_tensor).cpu().numpy()

        return InferResponse(
            actions=actions.tolist(),
            device=str(DEVICE),
        )

    except Exception as e:
        logger.error(f"Inference error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FastAPI server for diffusion policy inference"
    )
    parser.add_argument("--policy-ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--obs-dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--action-dim", type=int, default=4, help="Action dimension")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--act-horizon", type=int, default=8)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--train-script", required=True, help="Path to inference.py")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=9001, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")

    args = parser.parse_args()

    # Load agent before starting server
    logger.info("=" * 80)
    logger.info("Initializing diffusion policy agent...")
    logger.info("=" * 80)

    try:
        load_agent(
            train_script=args.train_script,
            policy_ckpt=args.policy_ckpt,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            obs_horizon=args.obs_horizon,
            act_horizon=args.act_horizon,
            pred_horizon=args.pred_horizon,
        )
    except Exception as e:
        logger.error(f"Failed to load agent: {traceback.format_exc()}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("=" * 80)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()

