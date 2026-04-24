#!/usr/bin/env bash
set -euo pipefail

# Server-side training launch for custom PickCube RGB + State dataset.
# Gemma4 + WM finetuning with PEFT/LoRA and wandb enabled.

DATA_PATH="/home/rakasheh/rgb_pick_cube/data_misc"
MODEL_PATH="/home/rakasheh/models/Llama-3.2-11B-Vision-Instruct/custom"
PEFT_OUT="/home/rakasheh/checkpoints/pickcube_rgb_ckpt"
WM_CONFIG_PATH="/home/rakasheh/rgb_pick_cube/forewarn/configs/wm_pickcube_config.yaml"
DATASET_FILE="/home/rakasheh/rgb_pick_cube/forewarn/vlm/gemma4/pickcube_wm_latent.py"
GEMMA4_DIR="/home/rakasheh/rgb_pick_cube/forewarn/vlm/gemma4"
DATASET_DIR="/home/rakasheh/rgb_pick_cube/forewarn/vlm/llama-recipes/recipes/quickstart/finetuning/datasets"
LLAMA_RECIPES_SRC="/home/rakasheh/rgb_pick_cube/forewarn/vlm/llama-recipes/src"

mkdir -p /home/rakasheh/rgb_pick_cube/forewarn/vlm/slurm_logs "${PEFT_OUT}"

#export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export PYTHONPATH="${GEMMA4_DIR}:${LLAMA_RECIPES_SRC}:${DATASET_DIR}:${PYTHONPATH:-}"
#source /opt/conda/etc/profile.d/conda.sh
#conda activate dreamer

cd "${GEMMA4_DIR}"

python finetuning_gemma_wm.py \
  --dataset custom_dataset \
  --custom_dataset.file "${DATASET_FILE}" \
  --custom_dataset.data_path "${DATA_PATH}" \
  --custom_dataset.train_split train \
  --custom_dataset.test_split test \
  --custom_dataset.answer_type open-word \
  --custom_dataset.num_images 16 \
  --custom_dataset.sample_size 16 \
  --custom_dataset.num_history_images 1 \
  --custom_dataset.imagined_steps 15 \
  --custom_dataset.latent_mode all \
  --custom_dataset.start_index 0 \
  --model_name "${MODEL_PATH}" \
  --output_dir "${PEFT_OUT}" \
  --use_wm True \
  --wm_config_path "${WM_CONFIG_PATH}" \
  --batch_size_training 1 \
  --gradient_accumulation_steps 1 \
  --batching_strategy padding \
  --num_epochs 10 \
  --use_peft \
  --peft_method lora \
  --target_modules "['linear']" \
  --use_fast_kernels \
  --use_wandb \
  --dist_checkpoint_root_folder "${PEFT_OUT}/dist_ckpt" \
  --dist_checkpoint_folder fine-tuned \
  --num_workers_dataloader 0

