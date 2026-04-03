import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import ruamel.yaml as yaml
import torch
from gym.spaces import Box, Dict
from robomimic.config import Config

# Keep imports local to this repo layout when running as a script.
FOREWARN_DIR = Path(__file__).resolve().parents[1]
if str(FOREWARN_DIR) not in sys.path:
    sys.path.append(str(FOREWARN_DIR))
MBIRL_DIR = FOREWARN_DIR / "model_based_irl_torch"
if str(MBIRL_DIR) not in sys.path:
    sys.path.append(str(MBIRL_DIR))

from dreamer.dreamer import Dreamer  # noqa: E402


class WMPredictorState:
    """State-only world-model latent extractor for ManiSkill HDF5 trajectories."""

    def __init__(
        self,
        config_path,
        ckpt_path,
        norm_dict_path,
        action_type="delta",
        device="cuda:0",
    ):
        self.action_type = action_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        cfg = self._load_defaults_config(config_path)
        cfg.from_ckpt = ckpt_path
        cfg.compile = False
        cfg.device = str(self.device)

        self._config = cfg
        self.wm_model = self._load_world_model(cfg)
        self.norm_dict = self._load_norm_dict(norm_dict_path)

    def _load_defaults_config(self, config_path):
        yaml_loader = yaml.YAML(typ="safe", pure=True)
        config_data = yaml_loader.load(Path(config_path).read_text())
        defaults = config_data["defaults"]
        return Config(defaults)

    def _load_world_model(self, cfg):
        action_space = Box(-1.0, 1.0, shape=tuple(cfg.action_space))
        cfg.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

        obs_space = {}
        for key, value in cfg.observation_space.items():
            shape = tuple(value)
            if key == "state" or "robot" in key:
                obs_space[key] = Box(-1.0, 1.0, shape=shape)
            elif key == "discount":
                obs_space[key] = Box(0.0, 1.0, shape=shape)
            else:
                obs_space[key] = Box(0.0, 1.0, shape=shape)

        model = Dreamer.from_pretrained(
            cfg.from_ckpt,
            Dict(obs_space),
            action_space,
            cfg,
            None,  # logger
            None,  # dataset
        ).to(torch.float32)
        model.requires_grad_(False)
        model.eval().to(self.device)
        return model

    def _load_norm_dict(self, norm_dict_path):
        with open(norm_dict_path, "r", encoding="utf-8") as f:
            norm = json.load(f)
        for key in norm:
            norm[key] = np.asarray(norm[key], dtype=np.float32)
        return norm

    def _normalize(self, arr, data_min, data_max):
        denom = np.maximum(data_max - data_min, 1e-6)
        return 2.0 * ((arr - data_min) / denom) - 1.0

    def _resolve_action_key(self, traj_group):
        if self.action_type == "abs" and "actions_abs" in traj_group:
            return "actions_abs"
        if "actions" in traj_group:
            return "actions"
        if "actions_abs" in traj_group:
            return "actions_abs"
        raise KeyError("No action dataset found (expected actions or actions_abs)")

    def _align_states_actions(self, obs, actions):
        if len(obs) == len(actions) + 1:
            states = obs[1 : len(actions) + 1]
            acts = actions
        else:
            seq_len = min(len(obs), len(actions))
            states = obs[:seq_len]
            acts = actions[:seq_len]
        return states, acts

    @torch.no_grad()
    def predict_latents(self, states, actions, latent_mode="all", latent_source="predicted"):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        seq_len = min(len(states), len(actions))
        if seq_len == 0:
            return np.zeros((0, 1), dtype=np.float32)

        states = states[:seq_len]
        actions = actions[:seq_len]

        states_n = self._normalize(states, self.norm_dict["ob_min"], self.norm_dict["ob_max"]).astype(np.float32)
        actions_n = self._normalize(actions, self.norm_dict["ac_min"], self.norm_dict["ac_max"]).astype(np.float32)

        # Reshape to (B=1, T=seq_len, ...)
        data = {
            "state": states_n[None, :, :],  # (1, T, 48)
            "action": actions_n[None, :, :],  # (1, T, 4)
            "is_first": np.array([[1] + [0] * (seq_len - 1)], dtype=np.float32),
            "is_terminal": np.zeros((1, seq_len), dtype=np.float32),
        }

        # Add dummy image tensors for all obs_keys if they're image keys
        for img_key in self._config.obs_keys:
            if "image" in img_key:
                # Create dummy images: (B=1, T=seq_len, H=64, W=64, C=3)
                data[img_key] = np.zeros((1, seq_len, 64, 64, 3), dtype=np.float32)

        actual_lengths = torch.tensor([seq_len], device=self.device)
        if latent_source == "predicted":
            latents = self.wm_model._wm.get_latent(
                data,
                mode=latent_mode,
                imagined_steps=max(seq_len - 1, 0),
                total_steps=seq_len,
                sample_size=seq_len,
                actual_lengths=actual_lengths,
            )
        elif latent_source == "gt":
            latents = self.wm_model._wm.get_latent_gt(
                data,
                mode=latent_mode,
                imagined_steps=0,
                total_steps=seq_len,
                sample_size=seq_len,
                actual_lengths=actual_lengths,
            )
        else:
            raise ValueError("latent_source must be either 'predicted' or 'gt'")

        return latents.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _sorted_demo_keys(container):
    keys = [k for k in container.keys() if isinstance(container[k], h5py.Group)]
    try:
        return sorted(keys, key=lambda x: int(x.split("_")[-1]))
    except Exception:
        return sorted(keys)


def export_latents_hdf5(
    input_h5,
    output_h5,
    predictor,
    latent_mode="all",
    latent_source="predicted",
    state_slice=None,
    max_demos=None,
):
    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        src_container = src["data"] if "data" in src else src
        out_container = dst.create_group("data")

        demo_keys = _sorted_demo_keys(src_container)
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]

        for demo_key in demo_keys:
            traj = src_container[demo_key]
            if "obs" not in traj:
                continue

            obs = np.asarray(traj["obs"], dtype=np.float32)
            if obs.ndim == 1:
                obs = obs[:, None]
            if state_slice is not None:
                obs = obs[:, state_slice]

            action_key = predictor._resolve_action_key(traj)
            actions = np.asarray(traj[action_key], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.ndim > 2 and actions.shape[1] == 1:
                actions = actions[:, 0, :]

            states, actions = predictor._align_states_actions(obs, actions)
            latents = predictor.predict_latents(
                states,
                actions,
                latent_mode=latent_mode,
                latent_source=latent_source,
            )

            out_demo = out_container.create_group(demo_key)
            out_demo.create_dataset("obs", data=states, compression="gzip")
            out_demo.create_dataset("actions", data=actions, compression="gzip")
            out_demo.create_dataset("wm_latent", data=latents, compression="gzip")

            if "label" in traj.attrs:
                out_demo.attrs["label"] = int(traj.attrs["label"])
            out_demo.attrs["source_action_key"] = action_key
            out_demo.attrs["latent_mode"] = latent_mode
            out_demo.attrs["latent_source"] = latent_source

        dst.attrs["input_h5"] = str(input_h5)


def parse_state_slice(slice_str):
    if slice_str is None:
        return None
    if ":" not in slice_str:
        idx = int(slice_str)
        return slice(idx, idx + 1)
    start_str, end_str = slice_str.split(":", 1)
    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None
    return slice(start, end)


def main():
    parser = argparse.ArgumentParser(description="Export state-only world-model latents from ManiSkill HDF5")
    parser.add_argument("--config_path", type=str, required=True, help="Path to WM config yaml, e.g., configs/wm_example_config.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained world-model checkpoint")
    parser.add_argument("--input_h5", type=str, required=True, help="Input ManiSkill HDF5 (success_data.h5 or failure_data.h5)")
    parser.add_argument("--output_h5", type=str, required=True, help="Output HDF5 with predicted latents")
    parser.add_argument("--norm_dict_path", type=str, required=True, help="Path to norm_dict_{delta|abs}.json used during training")
    parser.add_argument("--action_type", type=str, default="delta", choices=["delta", "abs"])
    parser.add_argument("--latent_mode", type=str, default="all", choices=["all", "z"])
    parser.add_argument("--latent_source", type=str, default="predicted", choices=["predicted", "gt"])
    parser.add_argument("--state_slice", type=str, default=None, help="Optional state slice, e.g. '0:48' or '12:'")
    parser.add_argument("--max_demos", type=int, default=None, help="Optional cap for quick smoke tests")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_h5)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

    state_slice = parse_state_slice(args.state_slice)
    predictor = WMPredictorState(
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        norm_dict_path=args.norm_dict_path,
        action_type=args.action_type,
        device=args.device,
    )

    export_latents_hdf5(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        predictor=predictor,
        latent_mode=args.latent_mode,
        latent_source=args.latent_source,
        state_slice=state_slice,
        max_demos=args.max_demos,
    )
    print(f"Saved latents to: {output_path}")


if __name__ == "__main__":
    main()

