# 🌟 Forewarn

**Paper:** *From Foresight to Forethought: VLM-In-the-Loop Policy Steering via Latent Alignment*  
**arXiv:** https://arxiv.org/abs/2502.01828  
**Project website:** https://yilin-wu98.github.io/forewarn/  

**Authors:** Yilin Wu¹, Ran Tian², Gokul Swamy¹, Andrea Bajcsy¹  
¹ Carnegie Mellon University · ² UC Berkeley

---

## 📌 Overview

This repository contains the official code for **Forewarn**, a framework that combines:

- 🌍 **World Model** (Dreamer-v3 style) trained on real robot rollouts & demonstrations
- 🦙 **VLM verifier** (modified Llama-3.2-11B-Vision-Instruct) that consumes **WM latents** for:
  - ✍️ **Stage 1:** behavior description generation
  - ✅ **Stage 2:** behavior selection under different scenarios

For VLM-specific setup and commands, see: **[`vlm/README.md`](./vlm/README.md)**.

---

## 🗂️ Repository Structure

The main code lives in **`model_based_irl_torch/`** (world model) and **`vlm/llama-recipes/`** (VLM).

### Key paths

- 📦 **World Model datasets (WM data):** ` /data/wm_data/ `  
  Place the WM datasets for **cup / bag / fork** tasks here (default).  
  You can change this via `root_dir` in the YAML configs.

- 📦 **VLM datasets (VLM data):**  
  `./vlm/llama-recipes/recipes/quickstart/finetuning/datasets/`  
  Place VLM datasets for **cup / bag / fork** tasks here (and dataset-loading code lives here too).

- 🧩 **World Model code:** [`model_based_irl_torch/`](./model_based_irl_torch/)  
  Dreamer-v3 world model + utilities.

- 🧠 **VLM code (modified Llama):** [`vlm/llama-recipes/`](./vlm/llama-recipes/)  
  Modified Llama-3.2-11B-Vision-Instruct and training/inference scripts.  
  Details: **[`vlm/README.md`](./vlm/README.md)**

- 🔥 **World Model training entry:** [`scripts/train_wm_real_data.py`](./scripts/train_wm_real_data.py)

- 🔥 **VLM finetuning entry:**  
  [`vlm/llama-recipes/src/llama_recipes/finetuning_wm.py`](./vlm/llama-recipes/src/llama_recipes/finetuning_wm.py)

- 🔎 **VLM inference entry:**  
  [`vlm/llama-recipes/recipes/quickstart/inference/local_inference/llama_wm_infer.py`](./vlm/llama-recipes/recipes/quickstart/inference/local_inference/llama_wm_infer.py)

---

## ⬇️ Installation

### 1) Clone the repo (with submodules)

```bash
git clone --recurse-submodules git@github.com:CMU-IntentLab/Forewarn.git
cd Forewarn
```

### 2) Create the conda environment

**create env + pip install**
```bash
conda env create -f environment.yaml
conda activate forewarn
pip install -r requirements.txt
```
**install dreamer wm**
```bash
cd model_based_irl_torch
pip install -e .
```

**install llama-recipes for vlm**
```bash
cd vlm/llama-recipes
pip install -e .
```


---

## 📦 Prepare Datasets

### Option 1: Download the provided datasets (recommended)

#### ✅ World Model data (WM data)

We provide WM data on Hugging Face: https://huggingface.co/datasets/yilin-wu/Forewarn_WM_data

Download and save under `/data/wm_data`:

```bash
huggingface-cli download yilin-wu/Forewarn_WM_data   --local-dir /data/wm_data   --repo-type dataset
```

> If you change the local directory, also update `root_dir` in the YAML config(s) under [`configs/`](./configs/).

#### ✅ VLM dataset (VLM data)

Please follow **[`vlm/README.md`](./vlm/README.md)** for VLM dataset downloading and placement.

---

### Option 2: Create your own dataset

In our experiments, we use **two sources**:
- ~100 demonstrations
- ~200 policy rollouts

#### Data format

- Save rollouts + demonstrations into an **HDF5** file with a format similar to RoboMimic.
- Add an additional attribute field:
  - `data['demo_x'].attrs['label']` stores the **mode** of the episode.

#### Helpful scripts

- 🔁 Relabel after creation: [`scripts/relabel_hdf5.py`](./scripts/relabel_hdf5.py)
- ✂️ Split into train/test for VLM finetuning: [`scripts/split_hdf5.py`](./scripts/split_hdf5.py)
- 📏 Compute normalization stats (state/action): [`scripts/compute_norm_dict.py`](./scripts/compute_norm_dict.py)  
  Saves a JSON file in the data folder used for WM preprocessing.

#### Hooking in your dataset loader

To load custom data properly:

1) Add a dataset fill function (similar to `fill_expert_dataset_real_data`) in:  
   [`model_based_irl_torch/dreamer/tools.py`](./model_based_irl_torch/dreamer/tools.py)

2) Add a dataset path + env meta resolver (similar to `get_real_dataset_path_and_env_meta`) in:  
   [`model_based_irl_torch/common/utils.py`](./model_based_irl_torch/common/utils.py)

3) Update the corresponding function call inside:  
   [`scripts/train_wm_real_data.py`](./scripts/train_wm_real_data.py)

---

## 🌍 World Model Training

### Reproduce tasks from the paper

You can load configs from [`configs/`](./configs/) and train from scratch:

- ☕ Cup:
```bash
python scripts/train_wm_real_data.py --config_path wm_cup_config.yaml
```

- 👜 Bag:
```bash
python scripts/train_wm_real_data.py --config_path wm_bag_config.yaml
```

- 🍴 Fork:
```bash
python scripts/train_wm_real_data.py --config_path wm_fork_config.yaml
```

### Evaluate world model quality (video rollouts)

After training, evaluate video quality by loading checkpoints via `from_ckpt` in the YAML config.
The evaluation logic is in the `evaluate` function inside:  
[`scripts/train_wm_real_data.py`](./scripts/train_wm_real_data.py)

### Download pretrained world model checkpoints (optional)

We provide pretrained checkpoints on Hugging Face: https://huggingface.co/yilin-wu/Forewarn_WMs

Download them into `logs/dreamer_cont/`, then set `from_ckpt` in your config file to point to the downloaded checkpoint:

- Cup example:
  - `from_ckpt: /path_to_project/logs/dreamer_cont/cup_ckpt/1229/222731/pretrain_joint.pt`
- Bag example:
  - `from_ckpt: /path_to_project/logs/dreamer_cont/bag_ckpt/0106/234547/pretrain_joint_150000.pt`
- Fork example:
  - `from_ckpt: /path_to_project/logs/dreamer_cont/fork_ckpt/0127/215329/pretrain_joint_150000.pt`

---

## 🧪 Train on Your Own Task (World Model)

1) Start from an existing config (e.g., [`configs/wm_cup_config.yaml`](./configs/wm_cup_config.yaml)) and modify:
- `task` (your env/task name)
- `root_dir` (dataset root directory)
- `obs_keys` (camera views)
- `state_keys` (robot states, gripper states)
- `batch_length` (temporal horizon of snippets; increase for longer context)
- `num_exp_trajs`, `validation_mse_traj` (train/eval trajectories)
- experiment name/output directories

2) Train:
```bash
python scripts/train_wm_real_data.py --config_path <your_config.yaml>
```

3) After training, set:
- `from_ckpt: <path_to_checkpoint>`
to evaluate or continue training.

---

## 🦙 VLM Finetuning + Inference

Please see **[`vlm/README.md`](./vlm/README.md)** for:
- VLM base model download
- VLM dataset download
- finetuning scripts
- inference commands (Stage 1 / Stage 2)

---


##  🌟 Policy Inference
We also provide a simplified examples for integrating world model and VLM for policy inference. Please see **[`examples`](./examples/)**.
- wm_pred_fork.py is an example to load the world model and vlm togehter and create a class called VLMInference
- policy_loop.py is an example to create a policy loop where it takes samples from the diffusion policy and pass in those action samples into VLMInference and gets the selected action sample to execute. 

In order to get complete exeuction of policy inference, you need to integrate this with your real robot controller and your base policy code to get action samples and observations to pass in. 

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{wu2025forewarn,
  title={From Foresight to Forethought: VLM-In-the-Loop Policy Steering via Latent Alignment},
  author={Wu, Yilin and Tian, Ran and Swamy, Gokul and Bajcsy, Andrea},
  journal={arXiv preprint arXiv:2502.01828},
  year={2025}
}
```
