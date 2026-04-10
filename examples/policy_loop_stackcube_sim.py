"""ManiSkill StackCube policy loop with WM and VLM integration.

This module replicates the policy_loop.py control logic but runs in the ManiSkill
StackCube-v1 simulation environment instead of on real robot hardware via Manimo.

It uses the same WM predictor and VLM inference stack for plan generation and
verification, enabling simulation-based policy development and evaluation.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import imageio
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for older installs
    import gym  # type: ignore[no-redef]

try:
    from .vlm_backend import LocalVLMBackend, RemoteVLMBackend
    from .wm_pred_fork import WMPredictor
except ImportError:  # pragma: no cover - direct script fallback
    from vlm_backend import LocalVLMBackend, RemoteVLMBackend
    from wm_pred_fork import WMPredictor


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_vector(value: Any, fallback_dim: int = 0) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.size == 0 and fallback_dim:
        return np.zeros(fallback_dim, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32, copy=False)


def _softmax(x: Sequence[float]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.size == 0:
        return values
    values = values - np.max(values)
    exp = np.exp(values)
    denom = np.sum(exp)
    if denom <= 0:
        return np.ones_like(values) / len(values)
    return exp / denom


def _extract_stackcube_state(obs: Any) -> Dict[str, np.ndarray]:
    if not isinstance(obs, dict):
        return {"state": _flatten_vector(obs)}

    state: Dict[str, np.ndarray] = {}
    for key in (
        "state",
        "tcp_pose",
        "cubeA_pose",
        "cubeB_pose",
        "tcp_to_cubeA_pos",
        "tcp_to_cubeB_pos",
        "cubeA_to_cubeB_pos",
        "success",
        "is_cubeA_grasped",
        "is_cubeA_on_cubeB",
        "is_cubeA_static",
    ):
        if key in obs:
            state[key] = _flatten_vector(obs[key])

    if "state" not in state:
        numeric_keys = [
            key
            for key, value in obs.items()
            if isinstance(value, (np.ndarray, list, tuple))
            and "image" not in key.lower()
            and "rgb" not in key.lower()
            and "depth" not in key.lower()
        ]
        if numeric_keys:
            state["state"] = np.concatenate([
                _flatten_vector(obs[key]) for key in numeric_keys
            ])

    return state


def _default_env_meta(
    env_id: str = "StackCube-v1",
    obs_mode: str = "state",
    control_mode: str = "pd_ee_delta_pos",
    max_episode_steps: int = 50,
    render_mode: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "env_name": env_id,
        "max_episode_steps": max_episode_steps,
        "render_mode": render_mode,
        "env_kwargs": {
            "obs_mode": obs_mode,
            "control_mode": control_mode,
            "robot_uids": "panda_wristcam",
        },
    }


class DummyPolicyCallback:
    """A lightweight callback template for state-only StackCube simulation."""

    def __init__(
        self,
        logdir: str = "./logs",
        experiment_name: str = "stackcube_sim",
        horizon: int = 64,
        action_dim: int = 4,
        policy_ckpt: Optional[str] = None,
        dp_obs_horizon: int = 2,
        dp_act_horizon: int = 8,
        dp_pred_horizon: int = 16,
    ):
        self.logger = type(
            "Logger",
            (object,),
            {"storage_path": logdir, "experiment_name": experiment_name},
        )()
        self.pred_obs = None
        self.horizon = horizon
        self.action_dim = action_dim
        self.logged_steps = []
        self._selected_traj = None
        self._selected_cursor = 0
        self._selected_step = 0
        self.policy_ckpt: Optional[str] = None
        self.dp_obs_horizon = int(dp_obs_horizon)
        self.dp_act_horizon = int(dp_act_horizon)
        self.dp_pred_horizon = int(dp_pred_horizon)
        self._dp_agent = None
        self._dp_device = None
        self._dp_torch = None
        self._dp_obs_history = None
        self._dp_loaded = False
        if policy_ckpt:
            self.load_policy_checkpoint(policy_ckpt)

    def load_policy_checkpoint(self, checkpoint_path: str):
        """Register a generative policy checkpoint path for custom callbacks."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
        self.policy_ckpt = checkpoint_path
        self._dp_loaded = False
        self._dp_agent = None
        self._dp_obs_history = None

    @staticmethod
    def _load_diffusion_training_symbols():
        """Dynamically load ManiSkill diffusion `Agent` and `Args` from train.py."""
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[3]
        baselines_dir = repo_root / "maniskill" / "Maniskill" / "examples" / "baselines"
        train_path = repo_root / "maniskill" / "Maniskill" / "examples" / "baselines" / "diffusion_policy" / "train.py"
        if not train_path.exists():
            raise FileNotFoundError(f"Diffusion training script not found: {train_path}")

        baselines_dir_str = str(baselines_dir)
        if baselines_dir_str not in sys.path:
            sys.path.insert(0, baselines_dir_str)

        spec = importlib.util.spec_from_file_location("maniskill_diffusion_train", str(train_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from: {train_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Agent, module.Args

    def _ensure_diffusion_agent(self, obs_dim: int):
        if self._dp_loaded:
            return
        if not self.policy_ckpt:
            return

        import torch

        Agent, Args = self._load_diffusion_training_symbols()
        args = Args(
            env_id="StackCube-v1",
            obs_horizon=self.dp_obs_horizon,
            act_horizon=self.dp_act_horizon,
            pred_horizon=self.dp_pred_horizon,
        )

        # Agent only needs these env-space fields during construction.
        shim = type("EnvShim", (), {})()
        shim.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dp_obs_horizon, obs_dim),
            dtype=np.float32,
        )
        shim.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = Agent(shim, args).to(device)
        checkpoint = torch.load(self.policy_ckpt, map_location=device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("ema_agent") or checkpoint.get("agent") or checkpoint
        else:
            state_dict = checkpoint
        agent.load_state_dict(state_dict)
        agent.eval()

        self._dp_torch = torch
        self._dp_device = device
        self._dp_agent = agent
        self._dp_loaded = True

    def on_begin_traj(self, traj_idx):
        self.logged_steps = []
        self._selected_traj = None
        self._selected_cursor = 0
        self._selected_step = 0
        self._dp_obs_history = None

    def get_candidate_plans_w_current_pose(self, obs, agent_choice=2, n_clusters=6):
        """Generate candidate plans from diffusion policy or heuristic fallback.

        Returns:
            tuple: (trajs, pred_trajs, aggregated_trajs, mode_probs, labels, current_pose)
                - trajs: Raw action sequences [n_clusters, horizon, action_dim]
                - pred_trajs: Same as trajs (no separate prediction space)
                - aggregated_trajs: Cumulative position changes [n_clusters, horizon, ...]
                - mode_probs: Probability/confidence for each trajectory
                - labels: Binary labels (1 for good, 0 for neutral, 2 for bad, 3 for unknown)
                - current_pose: Current TCP pose for context
        """
        state = _extract_stackcube_state(obs)
        state_vec = state.get("state", np.zeros(0, dtype=np.float32))
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        if state_vec.size == 0:
            state_vec = np.zeros(1, dtype=np.float32)

        current_pose = state.get("tcp_pose", np.zeros(7, dtype=np.float32))

        # Try to use diffusion policy if loaded
        if self.policy_ckpt:
            self._ensure_diffusion_agent(obs_dim=int(state_vec.shape[0]))
            if self._dp_agent is not None:
                if self._dp_obs_history is None or self._dp_obs_history.shape[1] != state_vec.shape[0]:
                    self._dp_obs_history = np.stack([state_vec] * self.dp_obs_horizon, axis=0)
                else:
                    self._dp_obs_history = np.roll(self._dp_obs_history, shift=-1, axis=0)
                    self._dp_obs_history[-1] = state_vec

                obs_batch = np.repeat(self._dp_obs_history[None], n_clusters, axis=0)
                with self._dp_torch.no_grad():
                    obs_tensor = self._dp_torch.from_numpy(obs_batch).float().to(self._dp_device)
                    action_seq = self._dp_agent.get_action(obs_tensor).detach().cpu().numpy()

                trajs = np.asarray(action_seq, dtype=np.float32)
                if trajs.ndim != 3:
                    trajs = trajs.reshape(n_clusters, -1, self.action_dim)
                if trajs.shape[-1] != self.action_dim:
                    if trajs.shape[-1] < self.action_dim:
                        pad = self.action_dim - trajs.shape[-1]
                        trajs = np.pad(trajs, ((0, 0), (0, 0), (0, pad)))
                    else:
                        trajs = trajs[:, :, : self.action_dim]

                # Compute aggregated trajectories (cumulative sum of actions)
                aggregated_trajs = np.cumsum(trajs[:, :, :3], axis=1) if trajs.shape[-1] >= 3 else np.zeros((n_clusters, trajs.shape[1], 3))
                if trajs.shape[-1] > 3:
                    aggregated_trajs = np.concatenate([aggregated_trajs, trajs[:, :, 3:4]], axis=-1)

                pred_trajs = trajs.copy()
                mode_probs = np.ones(len(trajs), dtype=np.float32) / max(len(trajs), 1)
                labels = np.zeros(len(trajs), dtype=np.int64)
                return trajs, pred_trajs, aggregated_trajs, mode_probs, labels, current_pose

    def visualize_plans_w_agg(self, trajs, aggregated_trajs, mode_probs, labels, current_pose):
        trajs = np.asarray(trajs)
        mode_probs = np.asarray(mode_probs)
        labels = np.asarray(labels)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        axes[0].set_title("Candidate plan summary")
        axes[0].bar(np.arange(len(mode_probs)), mode_probs, color="tab:blue")
        axes[0].set_ylabel("mode prob")
        axes[0].set_xlabel("candidate")
        for idx, label in enumerate(labels):
            if label == 1:
                axes[0].bar(idx, mode_probs[idx], color="tab:green")

        if trajs.ndim == 3 and trajs.shape[-1] >= 3:
            for idx, traj in enumerate(trajs):
                axes[1].plot(traj[:, 0], alpha=0.6, label=f"cand {idx}" if idx < 6 else None)
            axes[1].set_ylabel("action dim 0")
            axes[1].set_xlabel("horizon")
        else:
            axes[1].text(0.1, 0.5, "No trajectory data to plot", transform=axes[1].transAxes)

        if current_pose is not None and len(current_pose) >= 3:
            fig.suptitle(f"tcp xyz = {np.round(current_pose[:3], 3)}")
        fig.tight_layout()
        return fig

    def set_traj_in_the_middle(self, selected_action, pred_action, step_idx):
        self._selected_traj = np.asarray(selected_action, dtype=np.float32)
        self._selected_cursor = 0
        self._selected_step = step_idx

    def get_action(self, obs, pred_action=None):
        if self._selected_traj is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        cursor = min(self._selected_cursor, len(self._selected_traj) - 1)
        action = self._selected_traj[cursor]
        self._selected_cursor += 1
        return np.asarray(action, dtype=np.float32)

    def log_obs(self, obs, action_dict):
        self.logged_steps.append({"obs": obs, "action": action_dict})

    def on_step(self, traj_idx, step_idx):
        return None

    def on_end_traj(self, traj_idx, traj_status):
        return False


class PolicyLoopSim:
    def __init__(
        self,
        wm_config=None,
        env_meta=None,
        callbacks=None,
        T=50,
        mode="eval",
        logdir="./logs",
        answer_type="snippet",
        steering_mode="vlm",
        auto_start=False,
        max_trajectories=1,
        plan_interval=50,
        policy_ckpt: Optional[str] = None,
        peft_model: Optional[str] = None,
        model_name: Optional[str] = None,
        seed: Optional[int] = None,
        vlm_backend_mode: str = "auto",
        vlm_server_url: Optional[str] = None,
        vlm_timeout_s: float = 45.0,
        vlm_max_retries: int = 1,
        vlm_include_pred_frames: bool = True,
        allow_local_vlm_fallback: bool = False,
    ):
        self.callbacks = list(callbacks or [])
        self.wm_config = wm_config or {}
        self.logdir = logdir
        self.answer_type = answer_type
        self.steering_mode = steering_mode
        self.auto_start = auto_start
        self.max_trajectories = max_trajectories
        self.plan_interval = max(1, int(plan_interval))
        self.policy_ckpt = policy_ckpt
        self.T = int(T)
        self.mode = mode
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._default_plan_horizon = 64
        self._default_num_candidates = 6
        self._default_zero_action = None
        self.model_name = model_name
        self.peft_model = peft_model
        self.vlm_backend_mode = str(vlm_backend_mode).lower()
        self.vlm_server_url = vlm_server_url
        self.vlm_timeout_s = float(vlm_timeout_s)
        self.vlm_max_retries = int(vlm_max_retries)
        self.vlm_include_pred_frames = bool(vlm_include_pred_frames)
        self.allow_local_vlm_fallback = bool(allow_local_vlm_fallback)
        self.vlm_backend = None
        self._local_vlm_backend = None

        os.makedirs(self.logdir, exist_ok=True)

        # Initialize WM predictor
        self.wm_predictor = WMPredictor(self.wm_config)

        # Initialize VLM inference if steering_mode is vlm
        if self.steering_mode == "vlm":
            if self.model_name is None:
                self.model_name = "/data/mllama/Llama-3.2-11B-Vision-Instruct/custom"
            if self.peft_model is None:
                self.peft_model = "/data/peft_models/run_02_21_custom_wm_150k_vlm_finetuning_0.2%_imagined_step63_1_history_16sample_size_fork_task_open-word-fork-all_18epoch_print_eval_metrics_3class_aug_failure_by2_shuffle_key_correct_prompt_hist_no_start_from_75/peft_checkpoint_18"
            self._init_vlm_backend()

        self.env_meta = env_meta or _default_env_meta()
        self.env = self._make_env(self.env_meta)
        self.action_dim = self._infer_action_dim()
        self._default_zero_action = np.zeros(self.action_dim, dtype=np.float32)

        for callback in self.callbacks:
            if hasattr(callback, "action_dim"):
                callback.action_dim = self.action_dim
            if hasattr(callback, "horizon"):
                callback.horizon = self._default_plan_horizon
            if hasattr(callback, "logger"):
                callback.logger.storage_path = self.logdir
            if self.policy_ckpt:
                if hasattr(callback, "load_policy_checkpoint"):
                    callback.load_policy_checkpoint(self.policy_ckpt)
                elif hasattr(callback, "policy_ckpt"):
                    callback.policy_ckpt = self.policy_ckpt
                else:
                    print(
                        "Warning: callback does not expose checkpoint loading hooks; "
                        f"ignoring --policy-ckpt={self.policy_ckpt}"
                    )

    def _build_local_vlm_backend(self):
        return LocalVLMBackend(
            wm_configs=self.wm_config,
            model_name=self.model_name,
            peft_model=self.peft_model,
            answer_type=self.answer_type,
        )

    def _init_vlm_backend(self):
        mode = self.vlm_backend_mode
        if mode not in {"auto", "local", "remote"}:
            raise ValueError(f"Invalid --vlm-backend value: {mode}")

        if mode == "remote":
            if not self.vlm_server_url:
                raise ValueError("--vlm-server-url is required when --vlm-backend=remote")
            self.vlm_backend = RemoteVLMBackend(
                server_url=self.vlm_server_url,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                include_pred_frames=self.vlm_include_pred_frames,
            )
            print(f"VLM backend: remote ({self.vlm_server_url})")
            return

        if mode == "local":
            self._local_vlm_backend = self._build_local_vlm_backend()
            self.vlm_backend = self._local_vlm_backend
            print("VLM backend: local")
            return

        if self.vlm_server_url:
            self.vlm_backend = RemoteVLMBackend(
                server_url=self.vlm_server_url,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                include_pred_frames=self.vlm_include_pred_frames,
            )
            print(f"VLM backend: auto -> remote ({self.vlm_server_url})")
        else:
            self._local_vlm_backend = self._build_local_vlm_backend()
            self.vlm_backend = self._local_vlm_backend
            print("VLM backend: auto -> local")

    def _infer_vlm_two_stage(self, obs, trajs_candidates):
        if self.vlm_backend is None:
            return None, None, None, None

        # Debug: log what we're sending to VLM
        if isinstance(trajs_candidates, np.ndarray):
            print(f"Debug: trajs_candidates shape={trajs_candidates.shape}, dtype={trajs_candidates.dtype}, "
                  f"first traj sample=[{trajs_candidates[0, 0, :]}]")
        else:
            print(f"Debug: trajs_candidates type={type(trajs_candidates)}")

        try:
            return self.vlm_backend.infer_two_stage(
                obs,
                trajs_candidates,
                normalize=False,  # trajectories are already in delta/action space
                question_key="grasping",
            )
        except Exception as exc:
            print(f"Warning: primary VLM backend failed: {exc}")

        if not self.allow_local_vlm_fallback:
            return None, None, None, None

        if isinstance(self.vlm_backend, RemoteVLMBackend):
            try:
                if self._local_vlm_backend is None:
                    self._local_vlm_backend = self._build_local_vlm_backend()
                print("Falling back to local VLM backend")
                return self._local_vlm_backend.infer_two_stage(
                    obs,
                    trajs_candidates,
                    normalize=True,
                    question_key="grasping",
                )
            except Exception as fallback_exc:
                print(f"Warning: local VLM fallback failed: {fallback_exc}")

        return None, None, None, None


    def _make_env(self, env_meta: Dict[str, Any]):
        env_name = env_meta.get("env_name", "StackCube-v1")
        env_kwargs = copy.deepcopy(env_meta.get("env_kwargs", {}))
        env_kwargs.setdefault("obs_mode", "state")
        env_kwargs.setdefault("control_mode", "pd_ee_delta_pos")

        max_episode_steps = env_meta.get("max_episode_steps")
        render_mode = env_meta.get("render_mode", None)

        # Register the environment if needed.
        try:  # pragma: no cover - import is runtime-only
            from mani_skill.envs.tasks.tabletop import stack_cube as _stack_cube_env  # noqa: F401
        except Exception as exc:  # pragma: no cover - import is runtime-only
            raise ImportError(
                "ManiSkill StackCube environment is not available in this runtime"
            ) from exc

        make_kwargs = dict(env_kwargs)
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = max_episode_steps
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode

        return gym.make(env_name, **make_kwargs)

    def _infer_action_dim(self) -> int:
        action_space = getattr(self.env, "action_space", None)
        if action_space is not None and getattr(action_space, "shape", None):
            return int(np.prod(action_space.shape))
        return 4

    @staticmethod
    def _reset_env(env, seed=None):
        try:
            if seed is None:
                result = env.reset()
            else:
                result = env.reset(seed=seed)
        except TypeError:
            result = env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    @staticmethod
    def _step_env(env, action):
        result = env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            return result
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        raise RuntimeError(f"Unexpected step return signature: {type(result)}")

    @staticmethod
    def _obs_summary(obs: Any) -> Dict[str, Any]:
        if not isinstance(obs, dict):
            return {"obs_type": type(obs).__name__}
        summary = {}
        for key, value in obs.items():
            if isinstance(value, (np.ndarray, list, tuple)):
                summary[key] = list(np.asarray(value).shape)
            else:
                summary[key] = type(value).__name__
        return summary

    @staticmethod
    def _info_success(info: Any) -> bool:
        if not isinstance(info, dict):
            return False
        for key in ("success", "is_success"):
            if key in info:
                value = info[key]
                if isinstance(value, np.ndarray):
                    return bool(np.any(value))
                return bool(value)
        return False

    def _coerce_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self.action_dim:
            arr = np.pad(arr, (0, self.action_dim - arr.size))
        elif arr.size > self.action_dim:
            arr = arr[: self.action_dim]
        return arr

    def _state_vector(self, obs: Any) -> np.ndarray:
        state = _extract_stackcube_state(obs)
        if "state" in state:
            return state["state"]
        return np.concatenate(list(state.values())) if state else np.zeros(0, dtype=np.float32)

    def _candidate_score(self, obs: Any, trajectory: np.ndarray, label: Optional[int], mode_prob: Optional[float]) -> float:
        state = _extract_stackcube_state(obs)
        tcp = state.get("tcp_pose", np.zeros(7, dtype=np.float32))
        cube_a = state.get("cubeA_pose", tcp)
        cube_b = state.get("cubeB_pose", tcp)

        tcp_xyz = tcp[:3] if tcp.size >= 3 else np.zeros(3, dtype=np.float32)
        cube_a_xyz = cube_a[:3] if cube_a.size >= 3 else tcp_xyz
        cube_b_xyz = cube_b[:3] if cube_b.size >= 3 else tcp_xyz

        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim == 1:
            traj = traj[None, :]

        first = traj[0, :3] if traj.shape[1] >= 3 else np.zeros(3, dtype=np.float32)
        last = traj[-1, :3] if traj.shape[1] >= 3 else first
        smoothness = 0.0
        if len(traj) > 1 and traj.shape[1] >= 3:
            smoothness = float(np.mean(np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)))

        score = -float(np.linalg.norm(tcp_xyz + first - cube_a_xyz))
        score -= 0.5 * float(np.linalg.norm(cube_a_xyz + last - cube_b_xyz))
        score -= 0.1 * smoothness

        if traj.shape[1] > 3:
            score -= 0.05 * float(np.mean(np.abs(traj[:, 3])))

        if label == 1:
            score += 1.0
        elif label == 2:
            score -= 0.25

        if mode_prob is not None:
            score += 0.2 * float(mode_prob)

        return score

    def _candidate_bundle(self, result: Any):
        if isinstance(result, tuple):
            if len(result) == 6:
                return result
            if len(result) == 5:
                trajs, pred_trajs, aggregated_trajs, mode_probs, labels = result
                return trajs, pred_trajs, aggregated_trajs, mode_probs, labels, None
            if len(result) == 4:
                trajs, pred_trajs, aggregated_trajs, mode_probs = result
                return trajs, pred_trajs, aggregated_trajs, mode_probs, None, None
        raise ValueError(
            "callback.get_candidate_plans_w_current_pose must return 4, 5, or 6 values"
        )

    def _select_candidate(
        self,
        obs: Any,
        trajs_candidates: np.ndarray,
        pred_trajs_candidates: Optional[np.ndarray],
        mode_probs: Optional[Sequence[float]],
        labels: Optional[Sequence[int]],
    ):
        if trajs_candidates is None or len(trajs_candidates) == 0:
            return None, None, None

        best_idx = None
        best_score = -np.inf
        scores = []
        for idx, traj in enumerate(trajs_candidates):
            label = labels[idx] if labels is not None and idx < len(labels) else None
            mode_prob = mode_probs[idx] if mode_probs is not None and idx < len(mode_probs) else None
            score = self._candidate_score(obs, traj, label, mode_prob)
            scores.append(score)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            return None, None, scores

        return best_idx, trajs_candidates[best_idx], scores

    def process_pred(self, predictions):
        """Parse VLM predictions into class labels."""
        class_labels = []
        for text in predictions:
            keywords = {
                2: ["fail", "unable", "struggle", "did not", "could not", "does not", "cannot",
                    "incomplete", "unsuccessful", "trouble", "not succeed", "not manage", "ineffective"],
                0: ["corner", "edge", "side", "outer", "border", "in"],
                1: ["center", "midsection", "middle", "central", "core", "midpoint"],
            }
            class_labels.append(3)
            for label, terms in keywords.items():
                if any(term in text.lower() for term in terms):
                    class_labels[-1] = label
                    break
        return class_labels

    def _save_plan_figure(self, fig, step_idx: int):
        if fig is None:
            return

        plan_dir = os.path.join(self.logdir, "plans")
        os.makedirs(plan_dir, exist_ok=True)
        base_path = os.path.join(plan_dir, f"plan_step_{step_idx:04d}")

        if hasattr(fig, "savefig"):
            fig.savefig(f"{base_path}.png", bbox_inches="tight")
            plt.close(fig)
        elif hasattr(fig, "write_html"):
            fig.write_html(f"{base_path}.html")
        else:
            plt.close("all")

    def generate_plans(self, obs, step_idx):
        """Generate candidate plans and optionally verify with VLM."""
        if not self.callbacks:
            return None, None, None, None, None, None, None, None

        callback = self.callbacks[0]
        if not hasattr(callback, "get_candidate_plans_w_current_pose"):
            raise AttributeError(
                "The callback must define get_candidate_plans_w_current_pose(obs, agent_choice, n_clusters)"
            )

        result = callback.get_candidate_plans_w_current_pose(
            obs, agent_choice=2, n_clusters=self._default_num_candidates
        )
        trajs, pred_trajs, aggregated_trajs, mode_probs, labels, current_pose = self._candidate_bundle(result)

        trajs_candidates = aggregated_trajs if aggregated_trajs is not None else trajs
        pred_trajs_candidates = pred_trajs if pred_trajs is not None else trajs_candidates

        # VLM verification if enabled
        vlm_predictions = None
        vlm_predictions_2 = None
        pred_frames = None
        vlm_labels = None

        if self.steering_mode == "vlm" and self.vlm_backend is not None:
            print(f"Debug: Passing {trajs_candidates.shape if isinstance(trajs_candidates, np.ndarray) else 'non-array'} trajectories to VLM")
            infer_time = time.time()
            vlm_predictions, _text_input, vlm_predictions_2, pred_frames = self._infer_vlm_two_stage(
                obs,
                trajs_candidates,
            )
            if vlm_predictions is not None:
                vlm_labels = self.process_pred(vlm_predictions)
                print("vlm prediction", vlm_predictions, vlm_labels)
                print("vlm prediction 2", vlm_predictions_2)
                print("vlm inference time", time.time() - infer_time)

                # Save prediction gifs if available
                if pred_frames is not None and "pred_cam_rs" in pred_frames:
                    for choice in range(len(pred_frames["pred_cam_rs"])):
                        label = vlm_labels[choice] if choice < len(vlm_labels) else "unknown"
                        imageio.mimsave(
                            os.path.join(self.logdir, f"pred_{step_idx}_{choice}_{label}.gif"),
                            pred_frames["pred_cam_rs"][choice],
                        )
            else:
                print("VLM inference unavailable; falling back to heuristic candidate selection")

        # Visualize plans
        vis_fn = getattr(callback, "visualize_plans_w_agg", None)
        if callable(vis_fn):
            try:
                fig = vis_fn(
                    trajs,
                    aggregated_trajs if aggregated_trajs is not None else trajs,
                    mode_probs,
                    labels,
                    current_pose=current_pose,
                )
                self._save_plan_figure(fig, step_idx)
            except Exception as exc:
                print(f"Warning: plan visualization failed: {exc}")

        return trajs_candidates, pred_trajs_candidates, vlm_labels, vlm_predictions_2, pred_frames, mode_probs, labels, current_pose

    def _log_callback_obs(self, obs, action_dict):
        obs_copy = copy.deepcopy(obs)
        for callback in self.callbacks:
            if hasattr(callback, "log_obs"):
                callback.log_obs(obs_copy, copy.deepcopy(action_dict))

    def run(self):
        traj_idx = 0
        try:
            while self.max_trajectories is None or traj_idx < self.max_trajectories:
                if not self.auto_start:
                    key = input(f"--> Press Enter to start new trajectory {traj_idx} (q to quit): ")
                    if key.strip().lower() == "q":
                        break
                else:
                    print(f"Starting trajectory {traj_idx}")

                obs, info = self._reset_env(self.env, seed=self.seed)

                for callback in self.callbacks:
                    callback.pred_obs = copy.deepcopy(obs)
                    if hasattr(callback, "on_begin_traj"):
                        callback.on_begin_traj(traj_idx)

                start_time = time.time()
                steps = 0
                traj_status = "unknown"
                break_loop = False
                next_plan_step = 0
                start_index = 0
                pred_length = 64
                selected_ind = None
                selected_score = None
                selected_traj = None
                actions_candidates = None
                pred_actions_candidates = None
                vlm_labels = None
                vlm_pred_2 = None
                pred_frames = None
                mode_probs = None
                labels = None
                current_pose = None

                for step_idx in range(self.T):
                    steps += 1
                    state_vec = self._state_vector(obs)
                    action = None
                    success = True

                    if step_idx == next_plan_step:
                        (
                            actions_candidates,
                            pred_actions_candidates,
                            vlm_labels,
                            vlm_pred_2,
                            pred_frames,
                            mode_probs,
                            labels,
                            current_pose,
                        ) = self.generate_plans(obs, step_idx)

                        if actions_candidates is None or len(actions_candidates) == 0:
                            print("No candidate plans were produced; stopping trajectory.")
                            traj_status = "failure"
                            break

                        # VLM-based selection
                        if self.steering_mode == "vlm" and vlm_labels is not None:
                            success = False
                            selected_ind = None

                            # Try VLM verification answer
                            if vlm_pred_2 is not None and len(vlm_pred_2) > 0:
                                for choice in range(len(vlm_labels)):
                                    if str(choice + 1) in vlm_pred_2[0]:
                                        selected_ind = choice
                                        success = True
                                        break

                            # Fallback to label-based selection
                            if not success:
                                for choice in range(len(vlm_labels)):
                                    if vlm_labels[choice] == 1:
                                        selected_ind = choice
                                        success = True
                                        break

                            if not success:
                                print("No successful trajectory from VLM; stopping trajectory.")
                                traj_status = "failure"
                                break
                        else:
                            # Non-VLM steering modes not currently implemented for state-only StackCube
                            print("Heuristic steering mode not implemented for this configuration; stopping trajectory.")
                            traj_status = "failure"
                            break

                        if selected_ind is not None:
                            selected_traj = actions_candidates[selected_ind]

                            for callback in self.callbacks:
                                if hasattr(callback, "set_traj_in_the_middle"):
                                    callback.set_traj_in_the_middle(
                                        selected_traj,
                                        pred_actions_candidates[selected_ind]
                                        if pred_actions_candidates is not None
                                        else selected_traj,
                                        step_idx,
                                    )

                        next_plan_step = step_idx + self.plan_interval

                    for callback in self.callbacks:
                        if hasattr(callback, "get_action"):
                            candidate_action = callback.get_action(obs, pred_action=state_vec)
                            if candidate_action is not None:
                                action = candidate_action

                    if action is None:
                        action = self._default_zero_action.copy()

                    action = self._coerce_action(action)
                    new_obs, reward, terminated, truncated, info = self._step_env(self.env, action)

                    action_dict = {
                        "action": action,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "selected_ind": selected_ind,
                        "selected_score": selected_score,
                        "traj_idx": traj_idx,
                        "step_idx": step_idx,
                        "state": state_vec,
                        "obs_keys": list(obs.keys()) if isinstance(obs, dict) else [],
                    }
                    if selected_traj is not None:
                        action_dict["selected_traj"] = np.asarray(selected_traj)
                    if mode_probs is not None:
                        action_dict["mode_probs"] = np.asarray(mode_probs)
                    if labels is not None:
                        action_dict["labels"] = np.asarray(labels)
                    if current_pose is not None:
                        action_dict["current_pose"] = np.asarray(current_pose)
                    if vlm_labels is not None:
                        action_dict["vlm_labels"] = np.asarray(vlm_labels)
                    if isinstance(info, dict):
                        action_dict["info"] = {
                            key: (bool(value) if isinstance(value, (np.bool_, bool)) else value)
                            for key, value in info.items()
                        }

                    # Build a mutable container for logging-only enrichments.
                    if isinstance(obs, dict):
                        obs_for_log = copy.deepcopy(obs)
                    else:
                        obs_for_log = {
                            "obs_raw": _to_numpy(obs),
                            "obs_type": type(obs).__name__,
                        }

                    # Log prediction frames and labels during execution
                    if step_idx < pred_length + start_index and step_idx >= start_index:
                        if pred_frames is not None:
                            for key in pred_frames.keys():
                                for choice in range(self._default_num_candidates):
                                    if choice < len(pred_frames[key]):
                                        obs_for_log[f"{key}_{choice}"] = pred_frames[key][choice][step_idx - start_index]
                                if selected_ind is not None and selected_ind < len(pred_frames[key]):
                                    obs_for_log[key] = pred_frames[key][selected_ind][step_idx - start_index]
                        if vlm_labels is not None:
                            obs_for_log["vlm_labels"] = vlm_labels

                    self._log_callback_obs(obs_for_log, action_dict)

                    obs = new_obs
                    if self._info_success(info):
                        traj_status = "success"
                        break
                    if terminated or truncated:
                        traj_status = "timeout" if truncated and not self._info_success(info) else "terminated"
                        break

                    for callback in self.callbacks:
                        if hasattr(callback, "on_step"):
                            status = callback.on_step(traj_idx, step_idx)
                            if status is not None:
                                traj_status = status
                                break_loop = True
                                break
                    if break_loop:
                        break

                if traj_status == "unknown":
                    traj_status = "timeout" if steps >= self.T else "terminated"

                elapsed = max(time.time() - start_time, 1e-6)
                print(f"Trajectory {traj_idx} finished. Status: {traj_status}. FPS: {steps / elapsed:.2f}")

                for callback in self.callbacks:
                    if hasattr(callback, "on_end_traj"):
                        break_loop = callback.on_end_traj(traj_idx, traj_status=traj_status) or break_loop

                if break_loop:
                    print("Breaking loop requested by callback")
                    break

                traj_idx += 1

        except KeyboardInterrupt:
            print("Exiting Policy Loop Sim")
        finally:
            self.env.close()


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "defaults" not in data:
        data = {"defaults": data}
    return data


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(
        os.path.join(script_dir, "..", "configs", "wm_example_config_48d_state_only.yaml")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config, help="Path to WM config")
    parser.add_argument("--logdir", type=str, default=os.path.join(script_dir, "..", "logs", "policy_loop_sim"))
    parser.add_argument("--env-id", type=str, default="StackCube-v1")
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--traj-len", type=int, default=50)
    parser.add_argument("--plan-interval", type=int, default=50)
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--auto-start", action="store_true", help="Run trajectories without interactive prompts")
    parser.add_argument("--render", action="store_true", help="Render the simulation in the ManiSkill viewer")
    parser.add_argument(
        "--steering-mode",
        type=str,
        default="vlm",
        choices=("heuristic", "classifier", "vlm"),
        help="Plan selection mode: heuristic (state-based scoring), classifier, or vlm (VLM verification)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/data/mllama/Llama-3.2-11B-Vision-Instruct/custom",
        help="Path to base VLM model"
    )
    parser.add_argument(
        "--peft-model",
        type=str,
        default=None,
        help="Path to VLM PEFT checkpoint (trained adapter); if not set, will use default or training path"
    )
    parser.add_argument(
        "--policy-ckpt",
        type=str,
        default='/home/rakasheh/master/cassandra/maniskill/checkpoints/diffusion_state/checkpoint_diffusion_policy_stack_cube.pt',
        help="Path to trained generative policy checkpoint (e.g., diffusion policy checkpoint)",
    )
    parser.add_argument("--dp-obs-horizon", type=int, default=2, help="Diffusion policy observation horizon")
    parser.add_argument("--dp-act-horizon", type=int, default=8, help="Diffusion policy action horizon")
    parser.add_argument("--dp-pred-horizon", type=int, default=16, help="Diffusion policy prediction horizon")
    parser.add_argument(
        "--vlm-backend",
        type=str,
        default="auto",
        choices=("auto", "local", "remote"),
        help="VLM backend mode: local in-process, remote FastAPI, or auto",
    )
    parser.add_argument(
        "--vlm-server-url",
        type=str,
        default=None,
        help="Remote VLM server URL, e.g. http://my-gpu-server:8010",
    )
    parser.add_argument(
        "--vlm-timeout-s",
        type=float,
        default=45.0,
        help="Timeout (seconds) for remote VLM inference requests",
    )
    parser.add_argument(
        "--vlm-max-retries",
        type=int,
        default=1,
        help="Retry count for remote VLM requests",
    )
    parser.add_argument(
        "--vlm-include-pred-frames",
        action="store_true",
        help="Request predicted frames from remote VLM server (larger payloads)",
    )
    parser.add_argument(
        "--allow-local-vlm-fallback",
        action="store_true",
        help="If remote VLM fails, load local VLM and retry inference",
    )
    args = parser.parse_args()

    wm_config = _load_yaml_config(args.config)
    env_meta = _default_env_meta(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode="human" if args.render else None,
    )

    policy = DummyPolicyCallback(
        logdir=args.logdir,
        horizon=64,
        action_dim=4,
        policy_ckpt=args.policy_ckpt,
        dp_obs_horizon=args.dp_obs_horizon,
        dp_act_horizon=args.dp_act_horizon,
        dp_pred_horizon=args.dp_pred_horizon,
    )

    loop = PolicyLoopSim(
        wm_config=wm_config,
        env_meta=env_meta,
        callbacks=[policy],
        T=args.traj_len,
        logdir=args.logdir,
        steering_mode=args.steering_mode,
        auto_start=args.auto_start,
        max_trajectories=args.max_trajectories,
        plan_interval=args.plan_interval,
        policy_ckpt=args.policy_ckpt,
        seed=args.seed,
        model_name=args.model_name,
        peft_model=args.peft_model,
        vlm_backend_mode=args.vlm_backend,
        vlm_server_url=args.vlm_server_url,
        vlm_timeout_s=args.vlm_timeout_s,
        vlm_max_retries=args.vlm_max_retries,
        vlm_include_pred_frames=args.vlm_include_pred_frames,
        allow_local_vlm_fallback=args.allow_local_vlm_fallback,
    )
    loop.run()


if __name__ == "__main__":
    main()

