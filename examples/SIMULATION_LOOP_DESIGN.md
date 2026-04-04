# StackCube Simulation Policy Loop Design

## Overview

The `policy_loop_stackcube_sim.py` module replicates the control logic from `policy_loop.py` (robot loop using Manimo) but runs in the **ManiSkill StackCube-v1 simulation environment** instead of on real robot hardware.

Both loops use the same **World Model (WM)** and **Vision Language Model (VLM)** inference stack for plan generation and verification.

---

## Architecture

### WM (World Model)
- **Purpose**: Generate candidate action trajectories by predicting future observations
- **Initialization**: `WMPredictor(wm_config)` loads from checkpoint path in config
- **Flow**: 
  1. Callback generates candidate plans via `get_candidate_plans_w_current_pose()`
  2. WM is used internally by the callback to imagine action rollouts

### VLM (Vision Language Model)
- **Purpose**: Verify predicted trajectories and select the best one
- **Initialization**: `VLMInference(wm_configs, model_name, peft_model, answer_type)`
- **Checkpoints**:
  - `model_name`: Base LLaMA VLM (default: `/data/mllama/Llama-3.2-11B-Vision-Instruct/custom`)
  - `peft_model`: **Trained adapter checkpoint** (point here after VLM training completes)
- **Flow**:
  1. After candidate plans are generated, VLM verifies each one
  2. VLM runs `infer_two_stage()` to check trajectory quality
  3. Returns predictions + frame visualizations for analysis
  4. Selection logic: VLM verification answer → fallback to label-based selection

---

## Running with VLM

### CLI Arguments

```bash
python policy_loop_stackcube_sim.py \
  --config /path/to/wm_config.yaml \
  --steering-mode vlm \
  --model-name /data/mllama/Llama-3.2-11B-Vision-Instruct/custom \
  --peft-model /path/to/trained/peft_checkpoint_18 \
  --auto-start \
  --max-trajectories 10
```

### Where to Update After Training

After VLM fine-tuning completes, update the `--peft-model` argument to point to your trained checkpoint directory:

```bash
--peft-model /data/peft_models/your_training_run/peft_checkpoint_<epoch>
```

Or set it as default in code:
```python
if peft_model is None:
    peft_model = "/path/to/your/new/peft_checkpoint"
```

---

## Steering Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `heuristic` | State-based scoring (default fallback) | Baseline; no VLM needed |
| `classifier` | Reserved for future classifier-based selection | - |
| `vlm` | Uses trained VLM for verification | Production; after VLM training |

---

## Key Differences vs Robot Loop

| Aspect | Robot Loop (`policy_loop.py`) | Sim Loop (`policy_loop_stackcube_sim.py`) |
|--------|-------------------------------|------------------------------------------|
| Environment | Manimo (real robot) | ManiSkill StackCube-v1 |
| Observation | Real camera + state | Simulation state dict |
| Action space | Real joint commands | Simulation delta EE position |
| Plan horizon | 64 steps | 64 steps (configurable) |
| WM/VLM | Yes (required) | Yes (same models) |

---

## Data Flow in VLM Mode

1. **Observation** → Current state of environment
2. **Plan Generation** → Callback generates 6 candidate trajectories
3. **WM Prediction** → Trajectories translated to predicted observations
4. **VLM Inference** → Two-stage:
   - Stage 1: Classify each trajectory (grasping quality)
   - Stage 2: Verify best selection (placing quality)
5. **Selection** → Pick trajectory with best VLM score
6. **Execution** → Roll out selected trajectory in sim
7. **Logging** → Save prediction GIFs + labels to `logdir/`

---

## Configuration Files

- **WM Config**: `safety/forewarn/configs/wm_example_config_48d_state_only.yaml`
  - Defines task, observation space, checkpoint paths
  - Referenced via `--config` argument
  
- **VLM Checkpoint**: Post-training output
  - Point via `--peft-model` argument after training

---

## Example: Full VLM Eval Run

```bash
cd /home/rakasheh/master/cassandra

# After training completes and checkpoint is saved:
python -m safety.forewarn.examples.policy_loop_stackcube_sim \
  --config safety/forewarn/configs/wm_example_config_48d_state_only.yaml \
  --steering-mode vlm \
  --peft-model /data/peft_models/my_run_20260404/peft_checkpoint_18 \
  --auto-start \
  --max-trajectories 100 \
  --logdir ./logs/vlm_eval
```

---

## Troubleshooting

### "No model_namesuccessful trajectory from VLM"
- VLM predictions are not agreeing with any trajectory
- Check that predictions make sense for your task
- May need to retrain VLM with more diverse examples

### "Could not select a candidate trajectory"
- Fallback heuristic selection failed
- Set `--steering-mode heuristic` to use state-based scoring instead

### PEFT checkpoint path not found
- Verify checkpoint directory exists after training
- Check path format matches your training output
- Ensure PEFT adapter files are saved (*.bin, adapter_config.json)


