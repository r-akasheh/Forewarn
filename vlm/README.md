# 🦙 Forewarn VLM (Llama VLM) Code

This folder (`vlm/`) contains the code for our **Llama Vision-Language Model (VLM)** setup used in Forewarn.
We **remove the original vision encoder** and instead **project world-model (WM) latents into the language space** for finetuning and inference.

---

## 📁 Code Structure

The main code lives in `llama-recipes/`:

- 🧩 **Llama Recipes**: [`llama-recipes/`](https://github.com/CMU-IntentLab/llama-recipes/tree/5f0fddf9a89ea638f027324f08a02f149e267426)  
  Scripts and utilities related to Llama VLM training/inference.

- 📦 **Datasets**: [`recipes/quickstart/finetuning/datasets/`](https://github.com/CMU-IntentLab/llama-recipes/tree/5f0fddf9a89ea638f027324f08a02f149e267426/recipes/quickstart/finetuning/datasets)  
  Place the datasets for **cup / bag / fork** tasks here, along with the dataset-loading code.

- 🧠 **Model**: [`mllama_model.py`](https://github.com/CMU-IntentLab/llama-recipes/tree/5f0fddf9a89ea638f027324f08a02f149e267426/src/llama_recipes/models/mllama_model.py)  
  Modified Llama model that:
  - removes the original vision encoder
  - projects WM latents into the LLM language embedding space

- 🔥 **Training**: [`finetuning_wm.py`](https://github.com/CMU-IntentLab/llama-recipes/tree/5f0fddf9a89ea638f027324f08a02f149e267426/src/llama_recipes/finetuning_wm.py)  
  Script to run finetuning.

- 🔎 **Inference**: [`llama_wm_infer.py`]([./llama-recipes](https://github.com/CMU-IntentLab/llama-recipes/tree/5f0fddf9a89ea638f027324f08a02f149e267426/recipes/quickstart/inference/local_inference/llama_wm_infer.py)  
  Script to run inference for:
  - behavior description generation (Stage 1)
  - behavior selection (Stage 2)

---

## ⬇️ Dataset Download

Download the dataset from Hugging Face and put it under:

`./llama-recipes/recipes/quickstart/finetuning/datasets/`

```bash
huggingface-cli download yilin-wu/Forewarn_VLM_data   --local-dir ./llama-recipes/recipes/quickstart/finetuning/datasets
```

---

## ⬇️ Model Download

### 1) Download the modified base model (required)

Download the modified **Llama-3.2-11B-Vision-Instruct** base model from:

- https://huggingface.co/yilin-wu/Forewarn_VLMs/tree/main/Llama-3.2-11B-Vision-Instruct

This base model is modified to:
- remove the original vision encoder
- add a randomly-initialized projection layer

Put it under: `llama-recipes/mllama/`

```bash
mkdir -p llama-recipes/mllama
huggingface-cli download yilin-wu/Forewarn_VLMs   --local-dir ./llama-recipes/mllama   --include "Llama-3.2-11B-Vision-Instruct/**"
```

### 2) Download finetuned PEFT checkpoints (optional)

If you want to try our finetuned PEFT models (cup/bag/fork), download the `*ckpt` folders and store them anywhere.
Example:

```bash
mkdir -p /data/finetuned_models
huggingface-cli download yilin-wu/Forewarn_VLMs   --local-dir /data/finetuned_models   --include "*ckpt/**"
```

✅ When running inference, pass your local path via:  
`--peft_model_name <peft_model_path>`

---

## 🏋️ Training

After downloading the dataset and base model:

```bash
cd llama-recipes
```

Run one of the scripts below:

- ☕ Cup task
```bash
bash run_exp_cup.sh
```

- 👜 Bag task
```bash
bash run_exp_bag.sh
```

- 🍴 Fork task
```bash
bash run_exp_fork_all.sh
```

> Update paths inside these `.sh` files to match your local environment (dataset/model/checkpoint dirs).

---

## 🚀 Inference

We use the same entrypoint:  
`recipes/quickstart/inference/local_inference/llama_wm_infer.py`

There are **two modes**:

1) ✍️ **Stage 1**: Generate behavior descriptions (`answer_type="open-word"`)  
2) ✅ **Stage 2**: Select behavior descriptions (`answer_type="text"`) with a scenario key

---

### ✍️ Stage 1: Generate behavior descriptions (open-word)

**Template command:**

```bash
CUDA_VISIBLE_DEVICES=0 python recipes/quickstart/inference/local_inference/llama_wm_infer.py   --dataset "custom_dataset"   --custom_dataset.file <file_path>   --custom_dataset.data_path <data_path>   --custom_dataset.answer_type "open-word"   --custom_dataset.num_images 16   --custom_dataset.sample_size 16   --custom_dataset.num_history_images 1   --custom_dataset.imagined_steps 63   --custom_dataset.latent_mode "all"   --model_name "mllama/Llama-3.2-11B-Vision-Instruct/custom"   --batch_size_training 10   --custom_dataset.test_split "test"   --custom_dataset.start_index <start_idx>   --peft_model_name <peft_model_path>   --use_sentence True   --print-labels-predictions True
```

**Fork (stage 1):**
```bash
--custom_dataset.file "recipes/quickstart/finetuning/datasets/realfork_dataset_latent.py" 
--custom_dataset.data_path "realfork_data" 
--custom_dataset.start_index 60 
--peft_model_name /data/finetuned_models/Forewarn_VLMs/fork_ckpt
```

**Bag (stage 1):**
```bash
--custom_dataset.file "recipes/quickstart/finetuning/datasets/realbag_dataset_latent.py" 
--custom_dataset.data_path "realbag_data" 
--custom_dataset.start_index 0 
--peft_model_name /data/finetuned_models/Forewarn_VLMs/bag_ckpt
```

**Cup (stage 1):**
```bash
--custom_dataset.file "recipes/quickstart/finetuning/datasets/realcup_dataset_latent.py" 
--custom_dataset.data_path "realcup_data" --custom_dataset.start_index 35 
--peft_model_name /data/finetuned_models/Forewarn_VLMs/cup_ckpt
```

---

### ✅ Stage 2: Select behavior descriptions (text answer)

In this mode, you set a scenario via `--custom_dataset.question_key`.

**Template command:**

```bash
CUDA_VISIBLE_DEVICES=0 python recipes/quickstart/inference/local_inference/llama_wm_infer.py   --temperature 0.01   --top_p 0.9   --dataset "custom_dataset"   --custom_dataset.file <file_path>   --custom_dataset.data_path <data_path>   --custom_dataset.answer_type "text"   --custom_dataset.num_images 16   --custom_dataset.sample_size 16   --custom_dataset.num_history_images 1   --custom_dataset.imagined_steps 63   --custom_dataset.latent_mode "all"   --model_name "mllama/Llama-3.2-11B-Vision-Instruct/custom"   --batch_size_training 10   --custom_dataset.test_split "test"   --custom_dataset.start_index 0   --peft_model_name <peft_model_path>   --use_sentence True   --custom_dataset.question_key <key>   --print-labels-predictions True
```

**Cup (stage 2):** scenarios = `handle` / `interior`
```bash
--custom_dataset.file "recipes/quickstart/finetuning/datasets/realcup_dataset_text.py" 
--custom_dataset.data_path "realcup_data" --custom_dataset.question_key "handle" 
--peft_model_name /data/finetuned_models/Forewarn_VLMs/cup_ckpt
```

**Bag (stage 2):** scenarios = `edge` / `middle`
```bash
--custom_dataset.file "recipes/quickstart/finetuning/datasets/realbag_dataset_text.py"
--custom_dataset.data_path "realbag_data" --custom_dataset.question_key "middle" 
--peft_model_name /data/finetuned_models/Forewarn_VLMs/bag_ckpt
```

**Fork (stage 2):** scenarios = `grasp-handle` / `grasp-tines` / `place-bowl`
```bash
--custom_dataset.file "recipes/quickstart/finetuning/datasets/realfork_dataset_text.py"
--custom_dataset.data_path "realfork_data"
--custom_dataset.question_key "grasp-handle" --peft_model_name /data/finetuned_models/Forewarn_VLMs/fork_ckpt
```

---

## 🧯 Common Gotchas

- Confirm `--peft_model_name` points to the correct `*_ckpt` directory on your machine.
- If you relocate datasets/models, update paths in both:
  - `run_exp_*.sh`
  - inference commands above
