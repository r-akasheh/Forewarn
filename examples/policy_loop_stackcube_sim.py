from __future__ import annotations

import argparse
import copy
import imageio
import os
import pickle
import re
import select
import struct
import subprocess
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    import requests
except ImportError:
    requests = None

try:
    import gymnasium as gym
except ImportError:
    import gym

try:
    from .vlm_backend import LocalVLMBackend, RemoteVLMBackend
except ImportError:
    from vlm_backend import LocalVLMBackend, RemoteVLMBackend


def _pipe_send(proc: subprocess.Popen, obj: object) -> None:
    data = pickle.dumps(obj, protocol=4)
    proc.stdin.write(struct.pack(">I", len(data)))
    proc.stdin.write(data)
    proc.stdin.flush()


def _pipe_recv(proc: subprocess.Popen, timeout_s: float = 300.0) -> object:
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_s:
            raise TimeoutError(f"Worker timeout after {timeout_s}s")

        remaining = timeout_s - elapsed
        ready, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
        if ready:
            raw = proc.stdout.read(4)
            if len(raw) < 4:
                raise EOFError("Worker pipe closed")
            length, = struct.unpack(">I", raw)
            return pickle.loads(proc.stdout.read(length))


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


def _extract_stackcube_state(obs: Any) -> Dict[str, np.ndarray]:
    if not isinstance(obs, dict):
        return {"state": _flatten_vector(obs)}

    state = {}
    for key in ("state", "tcp_pose", "cubeA_pose", "cubeB_pose",
                "tcp_to_cubeA_pos", "tcp_to_cubeB_pos", "cubeA_to_cubeB_pos",
                "success", "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static"):
        if key in obs:
            state[key] = _flatten_vector(obs[key])

    if "state" not in state:
        numeric_keys = [k for k, v in obs.items()
                       if isinstance(v, (np.ndarray, list, tuple))
                       and not any(x in k.lower() for x in ("image", "rgb", "depth"))]
        if numeric_keys:
            state["state"] = np.concatenate([_flatten_vector(obs[k]) for k in numeric_keys])
    return state


def _default_env_meta(env_id="StackCube-v1", obs_mode="state",
                      control_mode="pd_ee_delta_pos", max_episode_steps=50,
                      render_mode=None):
    return {
        "env_name": env_id,
        "max_episode_steps": max_episode_steps,
        "render_mode": render_mode,
        "env_kwargs": {"obs_mode": obs_mode, "control_mode": control_mode,
                       "robot_uids": "panda_wristcam"},
    }


class FastAPIDiffusionPolicyBackend:
    def __init__(self, logdir="./logs", experiment_name="stackcube_sim", horizon=64,
                 action_dim=4, server_url="http://127.0.0.1:9001",
                 dp_obs_horizon=2, dp_act_horizon=8, dp_pred_horizon=16, timeout_s=30.0):
        if requests is None:
            raise ImportError("requests required for FastAPI backend")
        self.logger = type("Logger", (object,),
                          {"storage_path": logdir, "experiment_name": experiment_name})()
        self.pred_obs = None
        self.horizon = horizon
        self.action_dim = action_dim
        self.logged_steps = []
        self._selected_traj = None
        self._selected_cursor = 0
        self._selected_step = 0
        self.dp_obs_horizon = int(dp_obs_horizon)
        self.dp_act_horizon = int(dp_act_horizon)
        self.dp_pred_horizon = int(dp_pred_horizon)
        self._dp_obs_history = None
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self._check_server_health()
    
    def _check_server_health(self):
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            print(f"[DIFFUSION] Server ready on {data.get('device', '?')}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to diffusion server at {self.server_url}")
        except Exception as e:
            raise RuntimeError(f"Diffusion server health check failed: {e}")
    
    def on_begin_traj(self, traj_idx: int):
        self.logged_steps = []
        self._selected_traj = None
        self._selected_cursor = 0
        self._selected_step = 0
        self._dp_obs_history = None
    
    def get_candidate_plans_w_current_pose(self, obs, agent_choice=2, n_clusters=6):
        state = _extract_stackcube_state(obs)
        state_vec = np.asarray(state.get("state", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if state_vec.size == 0:
            raise RuntimeError("State vector is empty")

        current_pose = state.get("tcp_pose", np.zeros(7, dtype=np.float32))
        obs_dim = int(state_vec.shape[0])
        
        if self._dp_obs_history is None or self._dp_obs_history.shape[1] != obs_dim:
            self._dp_obs_history = np.stack([state_vec] * self.dp_obs_horizon, axis=0)
        else:
            self._dp_obs_history = np.roll(self._dp_obs_history, shift=-1, axis=0)
            self._dp_obs_history[-1] = state_vec
        
        obs_batch = np.repeat(self._dp_obs_history[None], n_clusters, axis=0)
        
        try:
            print(f"[DIFFUSION] Calling /infer with obs_batch shape {obs_batch.shape}")
            response = requests.post(f"{self.server_url}/infer",
                                    json={"obs_batch": obs_batch.tolist()},
                                    timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            print(f"[DIFFUSION] Got response with actions shape")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Diffusion server timeout after {self.timeout_s}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Diffusion server request failed: {e}")
        
        trajs = np.asarray(data["actions"], dtype=np.float32)
        if trajs.ndim != 3:
            trajs = trajs.reshape(n_clusters, -1, self.action_dim)
        if trajs.shape[-1] < self.action_dim:
            trajs = np.pad(trajs, ((0, 0), (0, 0), (0, self.action_dim - trajs.shape[-1])))
        elif trajs.shape[-1] > self.action_dim:
            trajs = trajs[:, :, :self.action_dim]
        
        aggregated_trajs = np.cumsum(trajs[:, :, :3], axis=1)
        if trajs.shape[-1] > 3:
            aggregated_trajs = np.concatenate([aggregated_trajs, trajs[:, :, 3:4]], axis=-1)
        
        mode_probs = np.ones(len(trajs), dtype=np.float32) / max(len(trajs), 1)
        labels = np.zeros(len(trajs), dtype=np.int64)
        return trajs, trajs.copy(), aggregated_trajs, mode_probs, labels, current_pose

    def set_traj_in_the_middle(self, selected_action, pred_action, step_idx: int):
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
    
    def visualize_plans_w_agg(self, trajs, aggregated_trajs, mode_probs, labels, current_pose=None):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].set_title("Candidate plans")
        axes[0].bar(np.arange(len(mode_probs)), mode_probs, color="tab:blue")
        axes[0].set_ylabel("prob")
        for idx, lbl in enumerate(labels):
            if lbl == 1:
                axes[0].bar(idx, mode_probs[idx], color="tab:green")
        
        trajs = np.asarray(trajs)
        if trajs.ndim == 3 and trajs.shape[-1] >= 3:
            for idx, traj in enumerate(trajs):
                axes[1].plot(traj[:, 0], alpha=0.6)
            axes[1].set_ylabel("action[0]")
            axes[1].set_xlabel("horizon")

        if current_pose is not None and len(current_pose) >= 3:
            fig.suptitle(f"tcp xyz = {np.round(current_pose[:3], 3)}")
        fig.tight_layout()
        return fig
    
    def log_obs(self, obs, action_dict):
        self.logged_steps.append({"obs": obs, "action": action_dict})
    
    def on_step(self, traj_idx, step_idx):
        return None



class DummyPolicyCallback:
    def __init__(self, logdir="./logs", experiment_name="stackcube_sim", horizon=64,
                 action_dim=4, dp_obs_horizon=2, dp_act_horizon=8, dp_pred_horizon=16):
        self.logger = type("Logger", (object,),
                          {"storage_path": logdir, "experiment_name": experiment_name})()
        self.pred_obs = None
        self.horizon = horizon
        self.action_dim = action_dim
        self.logged_steps = []
        self._selected_traj = None
        self._selected_cursor = 0
        self._selected_step = 0
        self.dp_obs_horizon = int(dp_obs_horizon)
        self.dp_act_horizon = int(dp_act_horizon)
        self.dp_pred_horizon = int(dp_pred_horizon)
        self._dp_obs_history = None
        self._worker_proc = None
        self._worker_obs_dim = None

    def on_begin_traj(self, traj_idx: int):
        self.logged_steps = []
        self._selected_traj = None
        self._selected_cursor = 0
        self._selected_step = 0
        self._dp_obs_history = None

    def get_candidate_plans_w_current_pose(self, obs, agent_choice=2, n_clusters=6):
        state = _extract_stackcube_state(obs)
        state_vec = np.asarray(state.get("state", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if state_vec.size == 0:
            raise RuntimeError("State vector is empty")

        current_pose = state.get("tcp_pose", np.zeros(7, dtype=np.float32))
        obs_dim = int(state_vec.shape[0])

        if self._dp_obs_history is None or self._dp_obs_history.shape[1] != obs_dim:
            self._dp_obs_history = np.stack([state_vec] * self.dp_obs_horizon, axis=0)
        else:
            self._dp_obs_history = np.roll(self._dp_obs_history, shift=-1, axis=0)
            self._dp_obs_history[-1] = state_vec

        obs_batch = np.repeat(self._dp_obs_history[None], n_clusters, axis=0)

        print(f"[DIFFUSION] Sending obs_batch shape {obs_batch.shape} to worker")
        _pipe_send(self._worker_proc, {"cmd": "infer", "obs_batch": obs_batch})
        response = _pipe_recv(self._worker_proc)

        if not isinstance(response, dict) or not response.get("ok"):
            err_msg = response.get('error', '?') if isinstance(response, dict) else str(response)
            raise RuntimeError(f"Worker inference error: {err_msg}")

        trajs = np.asarray(response["actions"], dtype=np.float32)
        if trajs.ndim != 3:
            trajs = trajs.reshape(n_clusters, -1, self.action_dim)
        if trajs.shape[-1] < self.action_dim:
            trajs = np.pad(trajs, ((0, 0), (0, 0), (0, self.action_dim - trajs.shape[-1])))
        elif trajs.shape[-1] > self.action_dim:
            trajs = trajs[:, :, :self.action_dim]

        aggregated_trajs = np.cumsum(trajs[:, :, :3], axis=1)
        if trajs.shape[-1] > 3:
            aggregated_trajs = np.concatenate([aggregated_trajs, trajs[:, :, 3:4]], axis=-1)

        mode_probs = np.ones(len(trajs), dtype=np.float32) / max(len(trajs), 1)
        labels = np.zeros(len(trajs), dtype=np.int64)
        return trajs, trajs.copy(), aggregated_trajs, mode_probs, labels, current_pose

    def set_traj_in_the_middle(self, selected_action, pred_action, step_idx: int):
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

    def visualize_plans_w_agg(self, trajs, aggregated_trajs, mode_probs, labels, current_pose=None):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].set_title("Candidate plans")
        axes[0].bar(np.arange(len(mode_probs)), mode_probs, color="tab:blue")
        axes[0].set_ylabel("prob")
        for idx, lbl in enumerate(labels):
            if lbl == 1:
                axes[0].bar(idx, mode_probs[idx], color="tab:green")

        trajs = np.asarray(trajs)
        if trajs.ndim == 3 and trajs.shape[-1] >= 3:
            for idx, traj in enumerate(trajs):
                axes[1].plot(traj[:, 0], alpha=0.6)
            axes[1].set_ylabel("action[0]")
            axes[1].set_xlabel("horizon")

        if current_pose is not None and len(current_pose) >= 3:
            fig.suptitle(f"tcp xyz = {np.round(current_pose[:3], 3)}")
        fig.tight_layout()
        return fig

    def log_obs(self, obs, action_dict):
        self.logged_steps.append({"obs": obs, "action": action_dict})

    def on_step(self, traj_idx, step_idx):
        return None

    def on_end_traj(self, traj_idx, traj_status):
        return False


class PolicyLoopSim:
    def __init__(self, wm_config=None, env_meta=None, callbacks=None, T=50, mode="eval",
                 logdir="./logs", answer_type="open-word", steering_mode="vlm", auto_start=False,
                 max_trajectories=1, plan_interval=50, peft_model=None, model_name=None,
                 seed=None, vlm_backend_mode="auto", vlm_server_url=None, vlm_timeout_s=45.0,
                 vlm_max_retries=1, vlm_include_pred_frames=True, allow_local_vlm_fallback=False):
        self.callbacks = list(callbacks or [])
        self.wm_config = wm_config or {}
        self.logdir = logdir
        self.answer_type = answer_type
        self.steering_mode = steering_mode
        self.auto_start = auto_start
        self.max_trajectories = max_trajectories
        self.plan_interval = max(1, int(plan_interval))
        self.T = int(T)
        self.mode = mode
        self.seed = seed
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

        # Local WM predictor is not used in this loop and can trigger heavy CUDA teardown.
        self.wm_predictor = None

        if self.steering_mode == "vlm":
            if self.model_name is None:
                self.model_name = "/data/mllama/Llama-3.2-11B-Vision-Instruct/custom"
            if self.peft_model is None:
                self.peft_model = (
                    "/data/peft_models/run_02_21_custom_wm_150k_vlm_finetuning_0.2%_imagined_step63_1"
                    "_history_16sample_size_fork_task_open-word-fork-all_18epoch_print_eval_metrics"
                    "_3class_aug_failure_by2_shuffle_key_correct_prompt_hist_no_start_from_75/peft_checkpoint_18"
                )
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

    def _build_local_vlm_backend(self):
        return LocalVLMBackend(wm_configs=self.wm_config, model_name=self.model_name,
                               peft_model=self.peft_model, answer_type=self.answer_type)

    def _init_vlm_backend(self):
        mode = self.vlm_backend_mode
        if mode not in {"auto", "local", "remote"}:
            raise ValueError(f"Invalid --vlm-backend value: {mode}")

        if self.answer_type != "open-word":
            print(f"[VLM] warning: answer_type={self.answer_type!r}; StackCube runs are typically trained with 'open-word'")

        if mode == "remote":
            if not self.vlm_server_url:
                raise ValueError("--vlm-server-url required when --vlm-backend=remote")
            self.vlm_backend = RemoteVLMBackend(
                server_url=self.vlm_server_url, timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries, include_pred_frames=self.vlm_include_pred_frames)
            print(f"[VLM] remote ({self.vlm_server_url})")
            return

        if mode == "local":
            self._local_vlm_backend = self._build_local_vlm_backend()
            self.vlm_backend = self._local_vlm_backend
            print("[VLM] local")
            return

        if self.vlm_server_url:
            self.vlm_backend = RemoteVLMBackend(
                server_url=self.vlm_server_url, timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries, include_pred_frames=self.vlm_include_pred_frames)
            print(f"[VLM] auto -> remote ({self.vlm_server_url})")
        else:
            self._local_vlm_backend = self._build_local_vlm_backend()
            self.vlm_backend = self._local_vlm_backend
            print("[VLM] auto -> local")

    def _infer_vlm_two_stage(self, obs, trajs_candidates, question_key="grasping"):
        if self.vlm_backend is None:
            return None, None, None, None
        try:
            return self.vlm_backend.infer_two_stage(
                obs, trajs_candidates, normalize=True, question_key=question_key)
        except Exception as exc:
            print(f"[VLM] primary failed: {exc}")

        if not self.allow_local_vlm_fallback:
            return None, None, None, None

        if isinstance(self.vlm_backend, RemoteVLMBackend):
            try:
                if self._local_vlm_backend is None:
                    self._local_vlm_backend = self._build_local_vlm_backend()
                print("[VLM] trying local fallback")
                return self._local_vlm_backend.infer_two_stage(
                    obs, trajs_candidates, normalize=True, question_key=question_key)
            except Exception as e:
                print(f"[VLM] local fallback failed: {e}")

        return None, None, None, None


    def _make_env(self, env_meta: Dict[str, Any]):
        env_name = env_meta.get("env_name", "StackCube-v1")
        env_kwargs = copy.deepcopy(env_meta.get("env_kwargs", {}))
        env_kwargs.setdefault("obs_mode", "state")
        env_kwargs.setdefault("control_mode", "pd_ee_delta_pos")
        max_episode_steps = env_meta.get("max_episode_steps")
        render_mode = env_meta.get("render_mode")

        try:
            from mani_skill.envs.tasks.tabletop import stack_cube as _
        except Exception as exc:
            raise ImportError("ManiSkill StackCube not available") from exc

        make_kwargs = dict(env_kwargs)
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = max_episode_steps
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode

        return gym.make(env_name, **make_kwargs)

    def _infer_action_dim(self) -> int:
        space = getattr(self.env, "action_space", None)
        if space is not None and getattr(space, "shape", None):
            return int(np.prod(space.shape))
        return 4

    @staticmethod
    def _reset_env(env, seed=None):
        try:
            result = env.reset(seed=seed) if seed is not None else env.reset()
        except TypeError:
            result = env.reset()
        return result if isinstance(result, tuple) and len(result) == 2 else (result, {})

    @staticmethod
    def _step_env(env, action):
        result = env.step(action)
        if len(result) == 5:
            return result
        obs, reward, done, info = result
        return obs, reward, done, False, info

    @staticmethod
    def _info_success(info: Any) -> bool:
        if not isinstance(info, dict):
            return False
        for key in ("success", "is_success"):
            if key in info:
                v = info[key]
                return bool(np.any(v)) if isinstance(v, np.ndarray) else bool(v)
        return False

    def _coerce_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self.action_dim:
            arr = np.pad(arr, (0, self.action_dim - arr.size))
        elif arr.size > self.action_dim:
            arr = arr[:self.action_dim]
        return arr

    def _state_vector(self, obs: Any) -> np.ndarray:
        state = _extract_stackcube_state(obs)
        if "state" in state:
            return state["state"]
        return np.concatenate(list(state.values())) if state else np.zeros(0, dtype=np.float32)

    def process_pred(self, predictions):
        class_labels = []

        keywords = {
            1: [
                "successfully", "completes", "achieves stable placement",
                "on top of", "correctly on", "stacking on the green cube",
                "securely on top", "achieves placement"
            ],

            2: [
                "fails to grasp", "unable to grasp", "cannot grasp",
                "does not manage to pick up", "grasp fails",
                "cannot achieve a firm grasp", "fails to secure"
            ],

            3: [
                "does not complete", "task remains incomplete",
                "no progress", "does not reach", "abandoned",
                "does not carry out", "no stacking motion"
            ],

            4: [
                "grasp but does not place", "does not complete placement",
                "fails to place", "does not reach the green cube",
                "holds the red cube but"
            ],

            5: [
                "falls off", "topples", "unstable", "slides off",
                "brief contact", "falls shortly after", "collapses"
            ],

            6: [
                "falls before reaching", "drops short", "falls without touching",
                "prematurely falls", "does not reach the green cube",
                "falls during the motion"
            ],

            7: [
                "does not release", "gripper remains closed",
                "keeps holding", "does not let go",
                "without releasing", "retains its grip"
            ],

            8: [
                "misplaces", "incorrectly", "regrasp", "correct", "try again",
                "second attempt", "picks it back up", "adjust"
            ],

            9: [
                "no attempt", "does not attempt", "remains stationary",
                "no movement", "does not initiate", "no interaction"
            ]
        }

        for text in predictions:
            text_l = text.lower()
            print("TEXT: ", text_l)

            label = 3  # fallback = "unknown / mixed failure"

            # priority: more specific classes first
            for cls_id in [1, 5, 6, 4, 7, 2, 8, 3, 9]:
                if any(k in text_l for k in keywords[cls_id]):
                    label = cls_id
                    break

            class_labels.append(label)

        return class_labels

    @staticmethod
    def _parse_mode_index(vlm_second_stage_output: Any, num_candidates: int) -> int | None:
        if not vlm_second_stage_output:
            return None

        text = str(vlm_second_stage_output[0]).lower()
        patterns = [
            r"behavior\s*mode\s*(\d+)",
            r"mode\s*(\d+)",
            r"index\s*(?:is|:)?\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            one_based_idx = int(match.group(1))
            if 1 <= one_based_idx <= num_candidates:
                return one_based_idx - 1
        return None

    @staticmethod
    def _is_missing_info_reply(text: str) -> bool:
        lowered = text.lower()
        markers = (
            "need more information",
            "please provide",
            "don't see",
            "do not see",
            "robot state",
            "candidate actions",
        )
        return any(marker in lowered for marker in markers)

    def _save_plan_figure(self, fig, step_idx: int):
        if fig is None:
            return
        plan_dir = os.path.join(self.logdir, "plans")
        os.makedirs(plan_dir, exist_ok=True)
        base = os.path.join(plan_dir, f"plan_step_{step_idx:04d}")
        try:
            if hasattr(fig, "savefig"):
                fig.savefig(f"{base}.png", bbox_inches="tight")
                plt.close(fig)
            elif hasattr(fig, "write_html"):
                fig.write_html(f"{base}.html")
        except Exception:
            try:
                plt.close(fig)
            except Exception:
                pass

    def _candidate_bundle(self, result):
        if isinstance(result, tuple):
            if len(result) == 6:
                return result
            if len(result) == 5:
                return (*result, None)
            if len(result) == 4:
                return (*result, None, None)
        raise ValueError("get_candidate_plans_w_current_pose must return 4-6 values")

    def generate_plans(self, obs, step_idx, question_key="grasping"):
        if not self.callbacks:
            return (None,) * 8

        callback = self.callbacks[0]
        result = callback.get_candidate_plans_w_current_pose(
            obs, agent_choice=2, n_clusters=self._default_num_candidates)
        trajs, pred_trajs, aggregated_trajs, mode_probs, labels, current_pose = \
            self._candidate_bundle(result)

        trajs_candidates = trajs
        pred_trajs_candidates = pred_trajs if pred_trajs is not None else trajs
        vlm_trajs_candidates = aggregated_trajs if aggregated_trajs is not None else trajs_candidates
        vlm_predictions = vlm_predictions_2 = pred_frames = vlm_labels = None
        text_input = None

        if self.steering_mode == "vlm" and self.vlm_backend is not None:
            print(
                f"[VLM] Infer {vlm_trajs_candidates.shape} trajectories "
                f"(q={question_key}, source={'aggregated' if aggregated_trajs is not None else 'raw'})"
            )
            t0 = time.time()
            vlm_predictions, text_input, vlm_predictions_2, pred_frames = \
                self._infer_vlm_two_stage(obs, vlm_trajs_candidates, question_key=question_key)

            if vlm_predictions is not None:
                missing_info_count = sum(
                    1 for pred in vlm_predictions if self._is_missing_info_reply(str(pred))
                )
                if missing_info_count == len(vlm_predictions):
                    print(
                        "[VLM] all stage-1 outputs request missing state/actions; "
                        "likely remote prompt/model mismatch (server answer_type/config)."
                    )
                    return (trajs_candidates, pred_trajs_candidates, None,
                            vlm_predictions_2, pred_frames, mode_probs, labels, current_pose)

                vlm_labels = self.process_pred(vlm_predictions)
                print(f"[VLM] pred={vlm_predictions}, labels={vlm_labels}")
                print(f"[VLM] pred_2={vlm_predictions_2}")
                if text_input:
                    print(f"[VLM] prompt={str(text_input)[:220]}")
                print(f"[VLM] Time: {time.time() - t0:.1f}s")

                if pred_frames is not None and isinstance(pred_frames, dict) and "pred_cam_rs" in pred_frames:
                    for choice in range(len(pred_frames["pred_cam_rs"])):
                        lbl = vlm_labels[choice] if choice < len(vlm_labels) else "unknown"
                        imageio.mimsave(
                            os.path.join(self.logdir, f"pred_{step_idx}_{choice}_{lbl}.gif"),
                            pred_frames["pred_cam_rs"][choice])
            else:
                print("[VLM] Returned None")

        vis_fn = getattr(callback, "visualize_plans_w_agg", None)
        if callable(vis_fn):
            try:
                fig = vis_fn(trajs, aggregated_trajs, mode_probs, labels, current_pose=current_pose)
                self._save_plan_figure(fig, step_idx)
            except Exception as exc:
                print(f"[PLOT] Visualization failed: {exc}")

        return (trajs_candidates, pred_trajs_candidates, vlm_labels,
                vlm_predictions_2, pred_frames, mode_probs, labels, current_pose)



    def _cleanup_resources(self):
        try:
            plt.close("all")
        except Exception:
            pass

        for cb in self.callbacks:
            if hasattr(cb, "_shutdown_worker"):
                try:
                    cb._shutdown_worker()
                except Exception:
                    pass

        if self.vlm_backend is not None:
            if hasattr(self.vlm_backend, "close"):
                try:
                    self.vlm_backend.close()
                except Exception:
                    pass
            self.vlm_backend = None

        if self._local_vlm_backend is not None:
            if hasattr(self._local_vlm_backend, "close"):
                try:
                    self._local_vlm_backend.close()
                except Exception:
                    pass
            self._local_vlm_backend = None

        if self.wm_predictor is not None:
            if hasattr(self.wm_predictor, "close"):
                try:
                    self.wm_predictor.close()
                except Exception:
                    pass
            if hasattr(self.wm_predictor, "shutdown"):
                try:
                    self.wm_predictor.shutdown()
                except Exception:
                    pass

        if self.env is not None:
            try:
                if hasattr(self.env, "close"):
                    self.env.close()
            except Exception:
                pass
            finally:
                self.env = None

    def _log_callback_obs(self, obs, action_dict):
        obs_copy = copy.deepcopy(obs)
        for cb in self.callbacks:
            if hasattr(cb, "log_obs"):
                cb.log_obs(obs_copy, copy.deepcopy(action_dict))



    def run(self):
        traj_idx = 0
        try:
            while self.max_trajectories is None or traj_idx < self.max_trajectories:
                if not self.auto_start:
                    key = input(f"Press Enter to start trajectory {traj_idx} (q to quit): ")
                    if key.strip().lower() == "q":
                        break
                else:
                    print(f"[TRAJ] Starting {traj_idx}")

                obs, info = self._reset_env(self.env, seed=self.seed)

                for cb in self.callbacks:
                    cb.pred_obs = copy.deepcopy(obs)
                    if hasattr(cb, "on_begin_traj"):
                        cb.on_begin_traj(traj_idx)

                start_time = time.time()
                steps = 0
                traj_status = "unknown"
                break_loop = False
                next_plan_step = 0
                start_index = 0
                pred_length = 64
                selected_ind = selected_traj = None
                actions_candidates = pred_actions_candidates = None
                vlm_labels = vlm_pred_2 = pred_frames = None
                mode_probs = labels = current_pose = None
                question_key = "stable"

                for step_idx in range(self.T):
                    steps += 1
                    state_vec = self._state_vector(obs)
                    action = None

                    if step_idx == next_plan_step:
                        (actions_candidates, pred_actions_candidates, vlm_labels, vlm_pred_2,
                         pred_frames, mode_probs, labels, current_pose) = \
                            self.generate_plans(obs, step_idx, question_key=question_key)
                        # Keep second-stage key within StackCube questions.json entries.
                        question_key = "retry_ok"

                        if actions_candidates is None or len(actions_candidates) == 0:
                            print("[PLAN] No candidates; stopping")
                            traj_status = "failure"
                            break

                        if vlm_labels is None:
                            print("[PLAN] VLM failed; stopping")
                            traj_status = "failure"
                            break

                        selected_ind = self._parse_mode_index(vlm_pred_2, len(vlm_labels))
                        if selected_ind is not None:
                            print(f"[PLAN] VLM stage-2 selected {selected_ind}")

                        if selected_ind is None:
                            for choice in range(len(vlm_labels)):
                                if vlm_labels[choice] == 1:
                                    selected_ind = choice
                                    print(f"[PLAN] VLM label selected {choice}")
                                    break

                        if selected_ind is None:
                            print("[PLAN] No acceptable trajectory; stopping")
                            traj_status = "failure"
                            break

                        selected_traj = actions_candidates[selected_ind]
                        for cb in self.callbacks:
                            if hasattr(cb, "set_traj_in_the_middle"):
                                cb.set_traj_in_the_middle(
                                    selected_traj,
                                    pred_actions_candidates[selected_ind]
                                    if pred_actions_candidates is not None else selected_traj,
                                    step_idx)

                        start_index = step_idx
                        next_plan_step = step_idx + self.plan_interval

                    for cb in self.callbacks:
                        if hasattr(cb, "get_action"):
                            a = cb.get_action(obs, pred_action=state_vec)
                            if a is not None:
                                action = a

                    if action is None:
                        action = self._default_zero_action.copy()

                    action = self._coerce_action(action)
                    new_obs, reward, terminated, truncated, info = self._step_env(self.env, action)

                    action_dict = {
                        "action": action, "reward": reward, "terminated": terminated,
                        "truncated": truncated, "selected_ind": selected_ind,
                        "traj_idx": traj_idx, "step_idx": step_idx, "state": state_vec,
                        "obs_keys": list(obs.keys()) if isinstance(obs, dict) else [],
                    }
                    if selected_traj is not None:
                        action_dict["selected_traj"] = np.asarray(selected_traj)
                    if mode_probs is not None:
                        action_dict["mode_probs"] = np.asarray(mode_probs)
                    if vlm_labels is not None:
                        action_dict["vlm_labels"] = np.asarray(vlm_labels)
                    if isinstance(info, dict):
                        action_dict["info"] = {
                            k: (bool(v) if isinstance(v, (np.bool_, bool)) else v)
                            for k, v in info.items()}

                    obs_for_log = copy.deepcopy(obs) if isinstance(obs, dict) else {
                        "obs_raw": _to_numpy(obs), "obs_type": type(obs).__name__}

                    if start_index <= step_idx < start_index + pred_length:
                        if pred_frames is not None:
                            for key in pred_frames.keys():
                                for choice in range(self._default_num_candidates):
                                    if choice < len(pred_frames[key]):
                                        obs_for_log[f"{key}_{choice}"] = \
                                            pred_frames[key][choice][step_idx - start_index]
                                if selected_ind is not None and selected_ind < len(pred_frames[key]):
                                    obs_for_log[key] = \
                                        pred_frames[key][selected_ind][step_idx - start_index]
                        if vlm_labels is not None:
                            obs_for_log["vlm_labels"] = vlm_labels

                    self._log_callback_obs(obs_for_log, action_dict)
                    obs = new_obs

                    if self._info_success(info):
                        traj_status = "success"
                        break
                    if terminated or truncated:
                        traj_status = "timeout" if truncated else "terminated"
                        break

                    for cb in self.callbacks:
                        if hasattr(cb, "on_step"):
                            status = cb.on_step(traj_idx, step_idx)
                            if status is not None:
                                traj_status = status
                                break_loop = True
                                break
                    if break_loop:
                        break

                if traj_status == "unknown":
                    traj_status = "timeout" if steps >= self.T else "terminated"

                elapsed = max(time.time() - start_time, 1e-6)
                print(f"[TRAJ] {traj_idx}: {traj_status}  FPS={steps/elapsed:.1f}")

                for cb in self.callbacks:
                    if hasattr(cb, "on_end_traj"):
                        break_loop = cb.on_end_traj(traj_idx, traj_status=traj_status) or break_loop

                if break_loop:
                    print("[TRAJ] Breaking loop (callback requested)")
                    break

                traj_idx += 1

        except KeyboardInterrupt:
            print("[EXIT] Keyboard interrupt")
        finally:
            self._cleanup_resources()



def _load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "defaults" not in data:
        data = {"defaults": data}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "configs", "wm_example_config_48d_state_only.yaml"))
    parser.add_argument("--logdir", default=os.path.join(
        os.path.dirname(__file__), "..", "logs", "policy_loop_sim"))
    parser.add_argument("--env-id", default="StackCube-v1")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--traj-len", type=int, default=50)
    parser.add_argument("--plan-interval", type=int, default=50)
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steering-mode", default="vlm", choices=("vlm",))
    parser.add_argument("--answer-type", default="open-word",
                        choices=("open-word", "text", "action", "category", "snippet"))
    parser.add_argument("--model-name",
                        default="/data/mllama/Llama-3.2-11B-Vision-Instruct/custom")
    parser.add_argument("--peft-model", default=None)
    parser.add_argument("--diffusion-backend", default="subprocess",
                        choices=("subprocess", "fastapi"),
                        help="Backend for diffusion policy: subprocess or fastapi")
    parser.add_argument("--diffusion-server-url", default="http://127.0.0.1:9001",
                        help="URL of fastapi diffusion server")
    parser.add_argument("--dp-obs-horizon", type=int, default=2)
    parser.add_argument("--dp-act-horizon", type=int, default=8)
    parser.add_argument("--dp-pred-horizon", type=int, default=16)
    parser.add_argument("--vlm-backend", default="auto", choices=("auto", "local", "remote"))
    parser.add_argument("--vlm-server-url", default=None)
    parser.add_argument("--vlm-timeout-s", type=float, default=45.0)
    parser.add_argument("--vlm-max-retries", type=int, default=1)
    parser.add_argument("--vlm-include-pred-frames", action="store_true")
    parser.add_argument("--allow-local-vlm-fallback", action="store_true")
    args = parser.parse_args()

    wm_config = _load_yaml_config(args.config)
    env_meta = _default_env_meta(
        env_id=args.env_id, obs_mode=args.obs_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        render_mode="human" if args.render else None)

    if args.diffusion_backend == "fastapi":
        print(f"[MAIN] FastAPI backend: {args.diffusion_server_url}")
        policy = FastAPIDiffusionPolicyBackend(
            logdir=args.logdir, horizon=64, action_dim=4,
            server_url=args.diffusion_server_url,
            dp_obs_horizon=args.dp_obs_horizon, dp_act_horizon=args.dp_act_horizon,
            dp_pred_horizon=args.dp_pred_horizon)
    else:
        print("[MAIN] Subprocess backend")
        policy = DummyPolicyCallback(
            logdir=args.logdir, horizon=64, action_dim=4,
            dp_obs_horizon=args.dp_obs_horizon, dp_act_horizon=args.dp_act_horizon,
            dp_pred_horizon=args.dp_pred_horizon)

    loop = PolicyLoopSim(
        wm_config=wm_config, env_meta=env_meta, callbacks=[policy], T=args.traj_len,
        logdir=args.logdir, answer_type=args.answer_type,
        steering_mode=args.steering_mode, auto_start=args.auto_start,
        max_trajectories=args.max_trajectories, plan_interval=args.plan_interval,
        seed=args.seed, model_name=args.model_name, peft_model=args.peft_model,
        vlm_backend_mode=args.vlm_backend, vlm_server_url=args.vlm_server_url,
        vlm_timeout_s=args.vlm_timeout_s, vlm_max_retries=args.vlm_max_retries,
        vlm_include_pred_frames=args.vlm_include_pred_frames,
        allow_local_vlm_fallback=args.allow_local_vlm_fallback)
    loop.run()


if __name__ == "__main__":
    main()
