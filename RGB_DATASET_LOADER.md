# ManiSkill RGB Dataset Loader Setup

## Overview

This document explains the new RGB dataset loader for ManiSkill data that has been implemented to support training world models with RGB camera images from your PickCube dataset.

## Problem Solved

The original `maniskill_privileged` loader expected a flat `obs` array, but your HDF5 dataset has a nested group structure:
```
obs/
  ├── agent/
  │   ├── qpos     (shape: T, 9)
  │   └── qvel     (shape: T, 9)
  ├── extra/
  │   ├── is_grasped
  │   ├── tcp_pose
  │   └── goal_pos
  ├── sensor_param/
  │   └── base_camera/
  └── sensor_data/
      └── base_camera/
          └── rgb  (shape: T, 128, 128, 3)
```

## Solution

Three changes were made:

### 1. New Path Resolver Function (`common/utils.py`)

Added `get_maniskill_rgb_dataset_path_and_env_meta()` to properly resolve ManiSkill RGB dataset paths:

```python
def get_maniskill_rgb_dataset_path_and_env_meta(config, env_id=None, done_mode=0):
    """Return ManiSkill RGB dataset paths for success/failure data."""
    success_path = Path(config.root_dir, config.success_data)
    failure_path = Path(config.root_dir, config.failure_data) if hasattr(config, 'failure_data') else None
    # ... validation ...
    return {"success": success_path, "failure": failure_path}, None
```

### 2. New Dataset Fill Function (`dreamer/tools.py`)

Implemented `fill_expert_dataset_maniskill_rgb()` that:
- Handles nested group structures (e.g., `agent/qpos`)
- Properly extracts RGB images from `sensor_data/base_camera/rgb`
- Extracts state from `agent/qpos` (configurable)
- Computes normalization statistics across all trajectories
- Returns data in the format expected by the Dreamer training loop

Key features:
- Supports both success and failure trajectory datasets
- Automatic normalization stats computation
- Configurable state key extraction (e.g., "agent/qpos")
- Proper sequence alignment across observations, actions, and images
- Sample frequency support

### 3. Updated Training Script (`scripts/train_wm_real_data.py`)

Updated the dataset loader selection logic to support the new RGB loader:

```python
dataset_loader = getattr(config, "dataset_loader", "real_data")
if dataset_loader == "maniskill_privileged":
    fill_dataset_fn = tools.fill_expert_dataset_maniskill_privileged
elif dataset_loader == "maniskill_rgb":
    fill_dataset_fn = tools.fill_expert_dataset_maniskill_rgb
else:
    fill_dataset_fn = tools.fill_expert_dataset_real_data
```

### 4. Updated Config (`configs/wm_pickcube_config.yaml`)

Changed state key specification to be explicit:
```yaml
# Before: state_keys: ["agent"]
# After:
state_keys: ["agent/qpos"]
```

This explicitly specifies that we want only the `qpos` component (9 dimensions) rather than both `qpos` and `qvel` (18 dimensions).

## Usage

### Command Line

To train with RGB data from your ManiSkill dataset:

```bash
python scripts/train_wm_real_data.py \
    --config_path wm_pickcube_config.yaml \
    --pretrain_joint_steps 100000 \
    --expt_name your_experiment_name \
    --dataset_loader maniskill_rgb
```

### Configuration

The `wm_pickcube_config.yaml` is already configured with:
- `dataset_loader: maniskill_rgb` (can be overridden via CLI)
- `state_keys: ["agent/qpos"]` (9-dimensional agent position)
- RGB image resolution: 128x128x3
- RGB key: `base_camera_rgb`

### Customization

If you want to load different state dimensions:

1. **Include velocities**: Change `state_keys: ["agent/qpos", "agent/qvel"]`
2. **Load from other groups**: Change `state_keys: ["extra/tcp_pose"]` (7-dim)
3. **Combine multiple sources**: Change `state_keys: ["agent/qpos", "extra/tcp_pose"]` (16-dim total)

Update the config's `observation_space` to match:
```yaml
observation_space:
  state: [16]  # Update based on total state dimension
```

## Files Modified

1. **`model_based_irl_torch/common/utils.py`**
   - Added: `get_maniskill_rgb_dataset_path_and_env_meta()`

2. **`model_based_irl_torch/dreamer/tools.py`**
   - Added: `fill_expert_dataset_maniskill_rgb()`
   - Helper functions for nested group access and normalization

3. **`scripts/train_wm_real_data.py`**
   - Updated: Dataset loader selection logic to support `maniskill_rgb`

4. **`configs/wm_pickcube_config.yaml`**
   - Changed: `state_keys: ["agent"] → ["agent/qpos"]`

## Data Format Requirements

Your HDF5 file must have this structure:

```
/data
  /traj_0
    /obs
      /agent
        /qpos (T, D_state)
        /qvel (T, D_state)
      /sensor_data
        /base_camera
          /rgb (T, H, W, 3)
    /actions (T, D_action)
  /traj_1
    ...
```

Optional attributes:
- `attrs['label']` on each trajectory group (defaults to 1 for success, 0 for failure)

## Training Output

When you run training with `--dataset_loader maniskill_rgb`, you'll see:

```
Loaded 100 RGB trajectories from ManiSkill datasets: ...
Observation shapes: {'base_camera_rgb': (128, 128, 3), 'state': (9,), ...}
```

The world model will now have access to both RGB observations and state information for learning!

## Next Steps

After training:
1. Save checkpoint with `--pretrain_joint_steps` 
2. Load with `--from_ckpt` to continue training
3. Use the trained world model for VLM integration

## Troubleshooting

**Issue**: "No trajectories selected for this split"
- Check `num_exp_trajs` and `validation_mse_trajs` config values
- Ensure you have enough trajectories in your dataset

**Issue**: State dimension mismatch
- Verify `state_keys` matches your actual data structure
- Update `observation_space.state` to match total state dimensions

**Issue**: RGB data not loading
- Check that `sensor_data/base_camera/rgb` path exists in your HDF5
- Verify RGB shape is (T, 128, 128, 3) or adjust config accordingly

