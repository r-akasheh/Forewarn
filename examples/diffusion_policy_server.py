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

try:
    from tslearn.clustering import TimeSeriesKMeans
except ImportError:
    TimeSeriesKMeans = None

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
    generation_mode: str | None = None  # "cluster" or "affine_corrupt"
    num_samples: int | None = None
    num_modes: int | None = None
    cluster_metric: str | None = None  # "euclidean", "dtw", etc. (if tslearn is available; otherwise ignored)
    affine_scale_range: float | None = None
    affine_bias_range: float | None = None
    affine_noise_std: float | None = None
    affine_seed: int | None = None


class InferResponse(BaseModel):
    actions: list
    device: str


def _sample_actions_independent(obs_seq: np.ndarray, num_samples: int) -> np.ndarray:
    """Sample action trajectories independently with B=1 calls to maximize diversity."""
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            obs_tensor = torch.from_numpy(obs_seq[None]).to(DEVICE)
            act = AGENT.get_action(obs_tensor).detach().cpu().numpy()[0]
            samples.append(act)
    return np.asarray(samples, dtype=np.float32)


def _greedy_diverse_subset(trajs: np.ndarray, k: int) -> np.ndarray:
    """Fallback when tslearn is unavailable: pick k diverse trajectories by farthest-first."""
    n = int(trajs.shape[0])
    if k >= n:
        return trajs.copy()
    flat = trajs.reshape(n, -1)
    chosen = [0]
    min_d2 = np.sum((flat - flat[0]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_d2))
        chosen.append(idx)
        d2 = np.sum((flat - flat[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return trajs[np.asarray(chosen, dtype=np.int64)]


def _cluster_trajectories(trajs: np.ndarray, num_modes: int, metric: str = "euclidean") -> np.ndarray:
    """Cluster sampled trajectories and return representative mode trajectories."""
    num_modes = max(1, min(int(num_modes), int(trajs.shape[0])))
    metric = str(metric).lower()

    if TimeSeriesKMeans is None:
        logger.warning("tslearn not installed; using greedy diverse subset fallback")
        return _greedy_diverse_subset(trajs, num_modes)

    km = TimeSeriesKMeans(n_clusters=num_modes, metric=metric, random_state=None, n_init=2, max_iter=30)
    km.fit(trajs)
    centers = np.asarray(km.cluster_centers_, dtype=np.float32)
    return centers


def _generate_affine_corruptions(
    base_traj: np.ndarray,
    num_modes: int,
    scale_range: float,
    bias_range: float,
    noise_std: float,
    seed: int | None = None,
) -> np.ndarray:
    """Return one original trajectory + (num_modes-1) affine-corrupted variants."""
    num_modes = max(1, int(num_modes))
    base = np.asarray(base_traj, dtype=np.float32)
    T, A = base.shape
    rng = np.random.default_rng(seed)

    modes = [base.copy()]
    for _ in range(num_modes - 1):
        scale = rng.uniform(1.0 - scale_range, 1.0 + scale_range, size=(1, A)).astype(np.float32)
        bias = rng.uniform(-bias_range, bias_range, size=(1, A)).astype(np.float32)
        noise = rng.normal(0.0, noise_std, size=(T, A)).astype(np.float32)
        corrupted = np.clip(base * scale + bias + noise, -1.0, 1.0)
        modes.append(corrupted.astype(np.float32))
    return np.stack(modes, axis=0)


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
DEFAULT_GENERATION_MODE = "cluster"
DEFAULT_NUM_SAMPLES = 32
DEFAULT_NUM_MODES = 6
DEFAULT_CLUSTER_METRIC = "euclidean"
DEFAULT_AFFINE_SCALE_RANGE = 0.35
DEFAULT_AFFINE_BIAS_RANGE = 0.15
DEFAULT_AFFINE_NOISE_STD = 0.05


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

        if obs_batch.ndim != 3:
            raise ValueError(f"Expected obs_batch shape (B, obs_horizon, obs_dim), got {obs_batch.shape}")

        obs_seq = obs_batch[0]
        mode = (request.generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
        num_modes = max(1, int(request.num_modes if request.num_modes is not None else DEFAULT_NUM_MODES))

        if mode == "cluster":
            num_samples = max(1, int(request.num_samples if request.num_samples is not None else DEFAULT_NUM_SAMPLES))
            metric = str(request.cluster_metric if request.cluster_metric is not None else DEFAULT_CLUSTER_METRIC)
            sampled_trajs = _sample_actions_independent(obs_seq, num_samples)
            actions = _cluster_trajectories(sampled_trajs, num_modes, metric=metric)
            logger.info(
                "infer(mode=cluster): sampled=%d clustered=%d horizon=%d act_dim=%d metric=%s",
                int(sampled_trajs.shape[0]), int(actions.shape[0]), int(actions.shape[1]), int(actions.shape[2]),
                metric,
            )
        elif mode == "affine_corrupt":
            base = _sample_actions_independent(obs_seq, 1)[0]
            scale_range = float(request.affine_scale_range if request.affine_scale_range is not None else DEFAULT_AFFINE_SCALE_RANGE)
            bias_range = float(request.affine_bias_range if request.affine_bias_range is not None else DEFAULT_AFFINE_BIAS_RANGE)
            noise_std = float(request.affine_noise_std if request.affine_noise_std is not None else DEFAULT_AFFINE_NOISE_STD)
            actions = _generate_affine_corruptions(
                base_traj=base,
                num_modes=num_modes,
                scale_range=max(0.0, scale_range),
                bias_range=max(0.0, bias_range),
                noise_std=max(0.0, noise_std),
                seed=request.affine_seed,
            )
            logger.info(
                "infer(mode=affine_corrupt): modes=%d horizon=%d act_dim=%d scale=%.3f bias=%.3f noise=%.3f",
                int(actions.shape[0]), int(actions.shape[1]), int(actions.shape[2]),
                max(0.0, scale_range), max(0.0, bias_range), max(0.0, noise_std),
            )
        else:
            raise ValueError(f"Unknown generation_mode={mode!r}; expected 'cluster' or 'affine_corrupt'")

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
    parser.add_argument("--candidate-generation-mode", default="cluster",
                        choices=("cluster", "affine_corrupt"),
                        help="How candidate trajectories are generated")
    parser.add_argument("--default-num-samples", type=int, default=32,
                        help="Default number of diffusion samples for cluster mode")
    parser.add_argument("--default-num-modes", type=int, default=6,
                        help="Default number of returned candidate trajectories")
    parser.add_argument("--default-cluster-metric", default="euclidean",
                        help="tslearn metric for cluster mode, e.g. dtw or euclidean")
    parser.add_argument("--default-affine-scale-range", type=float, default=0.35)
    parser.add_argument("--default-affine-bias-range", type=float, default=0.15)
    parser.add_argument("--default-affine-noise-std", type=float, default=0.05)

    args = parser.parse_args()

    global DEFAULT_GENERATION_MODE
    global DEFAULT_NUM_SAMPLES
    global DEFAULT_NUM_MODES
    global DEFAULT_CLUSTER_METRIC
    global DEFAULT_AFFINE_SCALE_RANGE
    global DEFAULT_AFFINE_BIAS_RANGE
    global DEFAULT_AFFINE_NOISE_STD

    DEFAULT_GENERATION_MODE = str(args.candidate_generation_mode)
    DEFAULT_NUM_SAMPLES = max(1, int(args.default_num_samples))
    DEFAULT_NUM_MODES = max(1, int(args.default_num_modes))
    DEFAULT_CLUSTER_METRIC = str(args.default_cluster_metric)
    DEFAULT_AFFINE_SCALE_RANGE = max(0.0, float(args.default_affine_scale_range))
    DEFAULT_AFFINE_BIAS_RANGE = max(0.0, float(args.default_affine_bias_range))
    DEFAULT_AFFINE_NOISE_STD = max(0.0, float(args.default_affine_noise_std))

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

