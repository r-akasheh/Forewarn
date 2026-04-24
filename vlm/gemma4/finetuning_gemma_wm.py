from __future__ import annotations

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import csv
import dataclasses
import json
import os
import random
import re
from pathlib import Path
from warnings import warn

import fire
import numpy as np
import torch
import torch.optim as optim

# set_submodule was added in PyTorch 2.0; patch older envs for bitsandbytes 4-bit.
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = _set_submodule
import ruamel.yaml as yaml
from accelerate.utils import is_xpu_available
from peft import PeftModel, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4Config, Gemma4ForConditionalGeneration

from llama_recipes.configs import (
	fsdp_config as FSDP_CONFIG,
	quantization_config as QUANTIZATION_CONFIG,
	train_config as TRAIN_CONFIG,
)
from llama_recipes.model_checkpointing import save_peft_checkpoint
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
	check_fsdp_config,
	generate_dataset_config,
	generate_peft_config,
	get_dataloader_kwargs,
	update_config,
)
from llama_recipes.utils.dataset_utils import (
	get_custom_data_collator,
	get_custom_eval_data_collator,
	get_preprocessed_dataset,
)
from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
	clear_gpu_cache,
	eval,
	freeze_transformer_layers,
	get_policies,
	print_model_size,
	setup,
	setup_environ_flags,
	train,
)

try:
	from .model import Gemma4ForConditionalGenerationWM, Gemma4WMImageProcessor, Gemma4WMProcessor
except ImportError:  # pragma: no cover - convenience for running as a script
	from model import Gemma4ForConditionalGenerationWM, Gemma4WMImageProcessor, Gemma4WMProcessor

import dreamer.dreamer
from dreamer.dreamer import Dreamer

try:
	from transformers.models.gemma4.modeling_gemma4 import Gemma4DecoderLayer
except ImportError:  # pragma: no cover
	try:
		from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer as Gemma4DecoderLayer
	except ImportError:  # pragma: no cover
		from transformers.models.llama.modeling_llama import LlamaDecoderLayer as Gemma4DecoderLayer


def process_config(task_name="PickCube", wm_config_path=None):
	yml = yaml.YAML(typ="safe", pure=True)
	if wm_config_path is not None:
		return yml.load(Path(wm_config_path).read_text())

	if "Cup" in task_name:
		return yml.load((Path(dreamer.__file__).parent / "../../configs/wm_cup_config.yaml").read_text())
	if "Bag" in task_name:
		return yml.load((Path(dreamer.__file__).parent / "../../configs/wm_bag_config.yaml").read_text())
	if "Fork" in task_name:
		return yml.load((Path(dreamer.__file__).parent / "../../configs/wm_fork_config.yaml").read_text())
	if "PickCube" in task_name:
		return yml.load((Path(dreamer.__file__).parent / "../../configs/wm_pickcube_config.yaml").read_text())
	raise ValueError(
		f"Could not infer WM config from task_name='{task_name}'. Provide --wm_config_path /absolute/path/to/wm_config.yaml"
	)


def setup_wandb(train_config, fsdp_config, **kwargs):
	try:
		import wandb
	except ImportError as exc:  # pragma: no cover
		raise ImportError("wandb is not installed. Please install it with pip install wandb") from exc

	from llama_recipes.configs import wandb_config as WANDB_CONFIG

	wandb_config = WANDB_CONFIG()
	update_config(wandb_config, **kwargs)
	run = wandb.init(**dataclasses.asdict(wandb_config))
	run.config.update(train_config)
	run.config.update(fsdp_config, allow_val_change=True)
	return run


def _infer_task_name(data_path: str, wm_config_path: str | None = None) -> str:
	data_path_lower = (data_path or "").lower()
	if "pickcube" in data_path_lower or "cube" in data_path_lower:
		return "PickCube"
	if "cup" in data_path_lower:
		return "GraspCup"
	if "bag" in data_path_lower:
		return "GraspBag"
	if "fork" in data_path_lower:
		return "GraspFork"
	if wm_config_path is None:
		raise ValueError("Could not infer task from dataset path. Please pass --wm_config_path for custom datasets.")
	return "Custom"


def main(**kwargs):
	train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
	update_config((train_config, fsdp_config), **kwargs)

	if is_xpu_available():
		torch.xpu.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	np.random.seed(train_config.seed)

	local_rank = rank = world_size = None
	device_id = torch.device("cpu")
	if is_xpu_available():
		device_id = torch.xpu.current_device()
	elif torch.cuda.is_available():
		device_id = torch.cuda.current_device()

	if train_config.enable_fsdp:
		setup()
		local_rank = int(os.environ["LOCAL_RANK"])
		rank = int(os.environ["RANK"])
		world_size = int(os.environ["WORLD_SIZE"])

	if torch.distributed.is_initialized():
		if is_xpu_available():
			torch.xpu.set_device(local_rank)
		elif torch.cuda.is_available():
			torch.cuda.set_device(local_rank)
		clear_gpu_cache(local_rank)
		setup_environ_flags(rank)

	wandb_run = None
	if train_config.use_wandb and (not train_config.enable_fsdp or rank == 0):
		wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

	bnb_config = None
	if train_config.quantization:
		if type(train_config.quantization) == type(True):
			warn(
				"Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit'.",
				FutureWarning,
			)
			train_config.quantization = "8bit"
		if train_config.quantization == "8bit" and train_config.enable_fsdp:
			raise ValueError("8bit quantization is not supported with FSDP, please use 4bit quantization")

		quant_config = QUANTIZATION_CONFIG()
		update_config(quant_config, **kwargs)
		bnb_config = quant_config.create_bnb_config(train_config.quantization)

	use_cache = False if train_config.enable_fsdp else None
	dataset_config = generate_dataset_config(train_config, kwargs)
	task_name = _infer_task_name(getattr(dataset_config, "data_path", ""), kwargs.get("wm_config_path"))
	wm_configs = process_config(task_name=task_name, wm_config_path=kwargs.get("wm_config_path"))

	config = Gemma4Config.from_pretrained(train_config.model_name)
	config.wm_config = wm_configs

	# device_map="auto" silently falls back to CPU for custom model subclasses
	# because accelerate cannot inspect their architecture.  Use an explicit map
	# so every layer is loaded directly onto the GPU with no CPU copy.
	_device_map = {"": device_id} if isinstance(device_id, int) else "cpu"

	model = Gemma4ForConditionalGenerationWM.from_pretrained(
		train_config.model_name,
		config=config,
		use_safetensors=True,
		quantization_config=bnb_config,
		torch_dtype=torch.float16 if train_config.quantization else torch.bfloat16,
		device_map=_device_map,
		low_cpu_mem_usage=True
	)
	# Verify placement — all parameter devices should be cuda, not cpu.
	_param_devices = {p.device.type for p in model.parameters()}
	print(f"Model parameter devices after loading: {_param_devices}")
	# Release safetensors mmap objects and any CPU-side loading buffers so the
	# ~10 GB of model-file pages are freed before dataset generation begins.
	import gc
	gc.collect()
	torch.cuda.empty_cache()

	tokenizer = AutoTokenizer.from_pretrained(
		train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name
	)
	if not tokenizer.pad_token_id:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
		print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
		model.resize_token_embeddings(len(tokenizer))

	print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

	if train_config.enable_fsdp and fsdp_config.pure_bf16 and not train_config.quantization:
		model.to(torch.bfloat16)

	if train_config.use_wm:
		model.initialize_vision_model()
		if torch.cuda.is_available():
			# Move WM model to explicit device. Keep in original float32 dtype;
			# process_batch_vision handles dtype casting of inputs to match WM model.
			wm_device = f"cuda:{device_id}" if isinstance(device_id, int) else device_id
			model.wm_model.to(wm_device)
			# Verify all parameters and buffers are actually on device
			for param in model.wm_model.parameters():
				assert param.device.type == 'cuda', f"WM param on {param.device}, expected cuda"
			for buf in model.wm_model.buffers():
				assert buf.device.type == 'cuda', f"WM buffer on {buf.device}, expected cuda"

	processor = Gemma4WMProcessor(
		image_processor=Gemma4WMImageProcessor(config=config.wm_config),
		tokenizer=tokenizer,
	)

	# prepare_model_for_kbit_training (PEFT) upcasts every non-quantized parameter
	# (embeddings, lm_head, layer norms) from fp16 → fp32 on GPU, which requires
	# a temporary second copy and OOMs on large vocab models (Gemma4: 256k tokens).
	# We replicate only the essential steps: freeze base weights, install the input-
	# gradient hook so LoRA adapters receive gradients, and enable activation
	# checkpointing.  Skipping the fp32 upcast is safe for LoRA fine-tuning.
	if train_config.quantization and train_config.use_peft and not train_config.enable_fsdp:
		for param in model.parameters():
			param.requires_grad = False
		model.enable_input_require_grads()
		model.gradient_checkpointing_enable()

	if train_config.use_peft:
		if train_config.from_peft_checkpoint:
			model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
			peft_config = model.peft_config
		else:
			peft_config = generate_peft_config(train_config, kwargs)
			model = get_peft_model(model, peft_config)

		if hasattr(model, "multi_modal_projector"):
			model.multi_modal_projector.requires_grad_(True)
		if wandb_run:
			wandb_run.config.update(peft_config)
		model.print_trainable_parameters()

	hsdp_device_mesh_plan = None
	if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
		hsdp_device_mesh_plan = hsdp_device_mesh(
			replica_group_size=fsdp_config.replica_group_size,
			sharding_group_size=fsdp_config.sharding_group_size,
		)
		print("HSDP device mesh is ready")

	is_vision = True
	if train_config.enable_fsdp:
		check_fsdp_config(fsdp_config)
		if not train_config.use_peft and train_config.freeze_layers:
			freeze_transformer_layers(model, train_config.num_freeze_layers)

		mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
		my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [Gemma4DecoderLayer])

		model = FSDP(
			model,
			auto_wrap_policy=(my_auto_wrapping_policy if train_config.use_peft else wrapping_policy),
			cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
			mixed_precision=(mixed_precision_policy if not fsdp_config.pure_bf16 else None),
			sharding_strategy=fsdp_config.sharding_strategy,
			device_mesh=hsdp_device_mesh_plan,
			device_id=device_id,
			limit_all_gathers=True,
			sync_module_states=train_config.low_cpu_fsdp,
			param_init_fn=(
				(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
				if train_config.low_cpu_fsdp and rank != 0
				else None
			),
		)
		if fsdp_config.fsdp_activation_checkpointing:
			model.enable_input_require_grads()
			model.gradient_checkpointing_enable()
			apply_fsdp_checkpointing(model)
	elif not train_config.quantization and not train_config.enable_fsdp:
		if is_xpu_available():
			model.to("xpu:0")
		elif torch.cuda.is_available():
			model.to("cuda")

	if not train_config.enable_fsdp and not train_config.quantization:
		model.enable_input_require_grads()
		model.gradient_checkpointing_enable()

	print("dataset config", dataset_config.num_history_images)
	model.init_dataset_config(dataset_config)
	processor.init_dataset_config(dataset_config)

	dataset_processer = processor
	dataset_train = get_preprocessed_dataset(dataset_processer, dataset_config, split="train")
	dataset_val = get_preprocessed_dataset(dataset_processer, dataset_config, split="test")

	if train_config.batching_strategy == "packing":
		raise ValueError("Packing is not supported for Gemma4 WM runs; use padding.")

	train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, dataset_processer, "train")
	custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
	custom_eval_data_collator = get_custom_eval_data_collator(dataset_processer, dataset_config)
	if custom_data_collator:
		train_dl_kwargs["collate_fn"] = custom_data_collator
	pin_memory = torch.cuda.is_available() and not train_config.enable_fsdp
	train_dataloader = torch.utils.data.DataLoader(
		dataset_train,
		num_workers=train_config.num_workers_dataloader,
		pin_memory=False,
		**train_dl_kwargs,
	)
	print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

	metrics_train_dataloader = train_dataloader
	if custom_eval_data_collator:
		metrics_train_dl_kwargs = dict(train_dl_kwargs)
		metrics_train_dl_kwargs["collate_fn"] = custom_eval_data_collator
		metrics_train_dataloader = torch.utils.data.DataLoader(
			dataset_train,
			num_workers=train_config.num_workers_dataloader,
			pin_memory=False,
			**metrics_train_dl_kwargs,
		)

	eval_dataloader = None
	metrics_eval_dataloader = None
	val_dl_kwargs = {}
	if train_config.run_validation:
		val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, dataset_processer, "val")
		if custom_data_collator:
			val_dl_kwargs["collate_fn"] = custom_data_collator

		eval_dataloader = torch.utils.data.DataLoader(
			dataset_val,
			num_workers=train_config.num_workers_dataloader,
			pin_memory=False,
			**val_dl_kwargs,
		)
		metrics_val_dl_kwargs = dict(val_dl_kwargs)
		if custom_eval_data_collator:
			metrics_val_dl_kwargs["collate_fn"] = custom_eval_data_collator
		metrics_eval_dataloader = torch.utils.data.DataLoader(
			dataset_val,
			num_workers=train_config.num_workers_dataloader,
			pin_memory=False,
			**metrics_val_dl_kwargs,
		)
		if len(eval_dataloader) == 0:
			raise ValueError(f"The eval set size is too small for dataloader to load even one batch. ({len(eval_dataloader)=})")
		print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

	if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
		optimizer = AnyPrecisionAdamW(
			model.parameters(),
			lr=train_config.lr,
			momentum_dtype=torch.bfloat16,
			variance_dtype=torch.bfloat16,
			use_kahan_summation=False,
			weight_decay=train_config.weight_decay,
		)
	else:
		optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
	scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

	results = train(
		model,
		train_dataloader,
		eval_dataloader,
		tokenizer,
		optimizer,
		scheduler,
		train_config.gradient_accumulation_steps,
		train_config,
		fsdp_config if train_config.enable_fsdp else None,
		local_rank if train_config.enable_fsdp else None,
		rank if train_config.enable_fsdp else None,
		wandb_run,
		processor=processor,
		dataset_config=dataset_config,
		val_dl_kwargs=val_dl_kwargs,
		dataset=dataset_val,
	)
	if not train_config.enable_fsdp or rank == 0:
		[print(f"Key: {k}, Value: {v}") for k, v in results.items()]
		if train_config.use_wandb:
			for k, v in results.items():
				wandb_run.summary[k] = v

	if train_config.enable_fsdp:
		import torch.distributed as dist

		dist.barrier()

	if train_config.use_peft:
		if train_config.enable_fsdp:
			if rank == 0:
				print("we are about to save the PEFT modules")
		else:
			print("we are about to save the PEFT modules")
		save_peft_checkpoint(model, train_config.output_dir + "/last_peft/")

	results, overall_acc = eval(model, metrics_train_dataloader, device_id, processor, dataset_config, split="train")
	eval_results, eval_overall_acc = None, None
	if metrics_eval_dataloader is not None:
		eval_results, eval_overall_acc = eval(model, metrics_eval_dataloader, device_id, processor, dataset_config, split="test")

	file_name = "no_hist_results_test_update_test.csv"

	def _to_python(value):
		if isinstance(value, torch.Tensor):
			return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
		return value

	def _metric_sort_key(metric_key):
		match = re.match(r"^class_(\d+)_", metric_key)
		if match:
			return (0, int(match.group(1)), metric_key)
		return (1, metric_key)

	def _metric_columns(*metric_dicts):
		metric_keys = set()
		for metric_dict in metric_dicts:
			if not metric_dict:
				continue
			for key in metric_dict.keys():
				if key == "matrix":
					continue
				metric_keys.add(key)
		return sorted(metric_keys, key=_metric_sort_key)

	metric_columns = _metric_columns(results, eval_results)

	def _build_row(group, metrics, group_overall_acc):
		metrics = metrics or {}
		scalar_metrics = {k: _to_python(v) for k, v in metrics.items()}
		row = {
			"group": group,
			"start_index": dataset_config.start_index,
			"type": "hist",
			"history_length": dataset_config.start_index,
			"sample_size": 16,
			"answer_type": dataset_config.answer_type,
			"overall_acc": _to_python(group_overall_acc),
			"metrics_json": json.dumps(scalar_metrics, ensure_ascii=True),
		}
		for key in metric_columns:
			row[key] = scalar_metrics.get(key, "")
		return row

	fieldnames = ["group", "start_index", "type", "history_length", "sample_size", "answer_type", "overall_acc", *metric_columns, "metrics_json"]

	reset_file = dataset_config.start_index == 0 or not os.path.exists(file_name)
	mode = "w" if reset_file else "a"
	with open(file_name, mode, newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if reset_file:
			writer.writeheader()
		writer.writerow(_build_row("train", results, overall_acc))
		if eval_results is not None:
			writer.writerow(_build_row("test", eval_results, eval_overall_acc))


if __name__ == "__main__":
	fire.Fire(main)

