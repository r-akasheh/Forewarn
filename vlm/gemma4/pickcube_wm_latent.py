import copy
import json
import os
from dataclasses import dataclass

import datasets
import h5py
import numpy as np
import torch
from PIL import Image
from datasets import SplitGenerator, load_dataset


def _check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def _replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def _tokenize_dialogs(dialogs, images, states, actions, is_first, is_terminal, lengths, processor, labels=None):
    # Custom processor doesn't have apply_chat_template; use tokenizer instead
    text_prompt = processor.tokenizer.apply_chat_template(dialogs, tokenize=False)
    processor_kwargs = dict(
        states=states,
        actions=actions,
        is_first=is_first,
        is_terminal=is_terminal,
        lengths=lengths,
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    )
    if images is not None:
        processor_kwargs["images"] = images
    batch = processor(**processor_kwargs)
    if labels is not None:
        batch["labels"] = labels
        return batch

    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        label_tokens = copy.copy(dialog_tokens)
        eot_indices = [j for j, n in enumerate(label_tokens) if n == 128009]
        last_idx = 0
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for idx in eot_indices:
            current_seq = label_tokens[last_idx : idx + 1]
            if _check_header(prompt_header_seqs, current_seq):
                label_tokens[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
        assistant_header_seq = [128006, 78191, 128007]
        label_tokens = _replace_target(assistant_header_seq, label_tokens)

        for j in range(len(label_tokens)):
            if label_tokens[j] == processor.tokenizer.pad_token_id or label_tokens[j] == 128256:
                label_tokens[j] = -100
        label_list.append(label_tokens)

    batch["labels"] = torch.tensor(label_list)
    return batch


def _load_questions(data_dir):
    candidates = ["question.json", "questions.json"]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No question file found in {data_dir}. Expected one of: {candidates}")


def _resolve_question(questions, answer_type, question_key):
    entry = questions.get(answer_type, None)
    if entry is None:
        # fallback to first key when the file is a flat map
        if len(questions) == 1:
            entry = list(questions.values())[0]
        else:
            raise KeyError(f"Question file does not contain key '{answer_type}'")

    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if question_key in entry:
            return entry[question_key]
        # fallback to first question template if key is missing
        return next(iter(entry.values()))

    raise ValueError("Unsupported question template format")


def _load_norm_dict(data_dir):
    candidates = [
        "norm_dict_delta.json",
    ]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}
    raise FileNotFoundError(f"No norm dict found in {data_dir}. Tried: {candidates}")


def _normalize(x, x_min, x_max):
    denom = np.maximum(x_max - x_min, 1e-6)
    return 2.0 * ((x - x_min) / denom) - 1.0


class HDF5PickCubeRGBDataset:
    VERSION = datasets.Version("1.0.0")

    def __init__(
        self,
        answer_type,
        num_images,
        latent_mode,
        imagined_steps,
        num_history_images,
        stride_size,
        start_index,
        question_key,
        *args,
        **kwargs,
    ):
        self.num_images = num_images
        self.latent_mode = latent_mode
        self.imagined_steps = imagined_steps
        self.num_history_images = num_history_images
        self.answer_type = answer_type
        self.stride_size = stride_size
        self.start_index = start_index
        self.question_key = question_key
        super().__init__(*args, **kwargs)

    def _info(self):
        steps = self.num_history_images + self.imagined_steps
        return datasets.DatasetInfo(
            description="PickCube RGB + State WM-latent dataset from HDF5 files.",
            features=datasets.Features(
                {
                    "states": datasets.Array2D(shape=(steps, 9), dtype="float32"),
                    "actions": datasets.Array2D(shape=(steps, 7), dtype="float32"),
                    "is_first": datasets.Array2D(shape=(steps, 1), dtype="float32"),
                    "is_terminal": datasets.Array2D(shape=(steps, 1), dtype="float32"),
                    "length": datasets.Value("int32"),
                    "images": datasets.Array2D(shape=(steps, 128 * 128 * 3), dtype="uint8"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        data_files = self.config.data_files
        split_name = "train" if "train" in data_files else "test"

        file_dir = os.path.dirname(data_files[split_name][0])
        question_path = _load_questions(file_dir)
        answer_path = os.path.join(file_dir, "answers.json")

        return [
            SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "hdf5_paths": data_files[split_name],
                    "questions": question_path,
                    "answer_path": answer_path,
                    "answer_type": self.answer_type,
                    "num_history_images": self.num_history_images,
                    "imagined_steps": self.imagined_steps,
                    "start_index": self.start_index,
                    "question_key": self.question_key,
                },
            )
        ]

    def _generate_examples(
        self,
        hdf5_paths,
        questions,
        answer_path,
        answer_type="open-word",
        num_history_images=1,
        imagined_steps=0,
        start_index=0,
        question_key="default",
    ):
        with open(answer_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
        answer_bank = answers.get(answer_type, {})

        for hdf5_path in hdf5_paths:
            data_dir = os.path.dirname(hdf5_path)
            norm_dict = _load_norm_dict(data_dir)
            question = _resolve_question(questions, answer_type, question_key)

            with h5py.File(hdf5_path, "r") as f:
                container = f["data"] if "data" in f else f
                trajectories = sorted(container.keys(), key=lambda x: int(x.split("_")[-1]))

                for traj_idx, traj in enumerate(trajectories):
                    group = container[traj]
                    label = int(group.attrs.get("label", 1))

                    # Extract state from obs/agent/qpos
                    obs = group["obs"]
                    states = np.asarray(obs["agent"]["qpos"], dtype=np.float32)

                    # Extract RGB images from obs/sensor_data/base_camera/rgb
                    rgb_data = np.asarray(obs["sensor_data"]["base_camera"]["rgb"], dtype=np.uint8)

                    # Downsample to 128×128 if needed (some trajectories may have higher resolution)
                    if rgb_data.shape[-3:-1] != (128, 128):
                        from PIL import Image
                        h, w = rgb_data.shape[-3:-1]
                        downsampled = []
                        for frame in rgb_data:
                            img = Image.fromarray(frame).resize((128, 128), Image.BILINEAR)
                            downsampled.append(np.array(img, dtype=np.uint8))
                        rgb_data = np.stack(downsampled, axis=0)

                    # Extract actions
                    actions = np.asarray(group["actions"], dtype=np.float32)

                    if states.ndim == 1:
                        states = states[:, None]
                    if actions.ndim == 1:
                        actions = actions[:, None]

                    seq_len = min(len(states), len(actions), len(rgb_data))
                    if seq_len == 0:
                        continue

                    states = states[:seq_len]
                    actions = actions[:seq_len]
                    rgb_data = rgb_data[:seq_len]

                    window_steps = num_history_images + imagined_steps
                    begin = min(start_index, max(0, seq_len - 1))
                    end = min(seq_len, begin + window_steps)

                    states_w = states[begin:end].copy()
                    actions_w = actions[begin:end].copy()
                    # .copy() is critical: a numpy slice is a view that keeps the
                    # entire backing trajectory array alive in RAM until the
                    # datasets Arrow writer releases the batch (~100 examples).
                    rgb_w = rgb_data[begin:end].copy()

                    if len(states_w) < window_steps:
                        pad = window_steps - len(states_w)
                        states_w = np.concatenate([states_w, np.repeat(states_w[-1:], pad, axis=0)], axis=0)
                        actions_w = np.concatenate([actions_w, np.repeat(actions_w[-1:], pad, axis=0)], axis=0)
                        rgb_w = np.concatenate([rgb_w, np.repeat(rgb_w[-1:], pad, axis=0)], axis=0)

                    states_w = _normalize(states_w, norm_dict["ob_min"], norm_dict["ob_max"]).astype(np.float32)
                    actions_w = _normalize(actions_w, norm_dict["ac_min"], norm_dict["ac_max"]).astype(np.float32)

                    is_first = np.zeros((window_steps, 1), dtype=np.float32)
                    is_first[0, 0] = 1.0
                    is_terminal = np.zeros((window_steps, 1), dtype=np.float32)

                    label_key = str(label)
                    answer_candidates = answer_bank.get(label_key, [f"Label {label}"])
                    answer = answer_candidates[traj_idx % len(answer_candidates)]

                    yield f"{hdf5_path}-{traj_idx}", {
                        "images": rgb_w.reshape(window_steps, 128 * 128 * 3),
                        "states": states_w,
                        "actions": actions_w,
                        "is_first": is_first,
                        "is_terminal": is_terminal,
                        "question": question,
                        "answer": answer,
                        "length": window_steps,
                        "label": label,
                    }


@dataclass
class DataCollector:
    processor: object
    data_config: object = None

    def __post_init__(self):
        self.processor.tokenizer.padding_side = "right"
        self.num_images = self.data_config.num_images if self.data_config is not None else 1

    def __call__(self, samples):
        dialogs, images, states, actions, is_first, is_terminal, lengths = [], [], [], [], [], [], []
        for ex in samples:
            dialog = [
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"].strip()}]},
            ]
            for _ in range(self.num_images - 1):
                dialog[0]["content"].append({"type": "image"})
            dialog[0]["content"].append({"type": "text", "text": ex["question"].strip()})
            dialogs.append(dialog)
            # Arrow stores images as (T, H*W*C) uint8; restore spatial dims here.
            images.append(np.array(ex["images"]).reshape(self.num_images, 128, 128, 3))
            states.append(ex["states"])
            actions.append(ex["actions"])
            is_first.append(ex["is_first"])
            is_terminal.append(ex["is_terminal"])
            lengths.append(ex["length"])
        image_batch = images if any(img.size > 0 for img in images) else None
        return _tokenize_dialogs(dialogs, image_batch, states, actions, is_first, is_terminal, lengths, self.processor)


@dataclass
class EvalDataCollector:
    processor: object
    data_config: object = None

    def __post_init__(self):
        self.processor.tokenizer.padding_side = "right"
        self.num_images = self.data_config.num_images if self.data_config is not None else 1

    def __call__(self, samples):
        dialogs, images, states, actions, is_first, is_terminal, lengths, answers = [], [], [], [], [], [], [], []
        for ex in samples:
            dialog = [{"role": "user", "content": [{"type": "image"}]}]
            for _ in range(self.num_images - 1):
                dialog[0]["content"].append({"type": "image"})
            dialog[0]["content"].append({"type": "text", "text": ex["question"].strip()})
            dialogs.append(dialog)
            images.append(np.array(ex["images"]).reshape(self.num_images, 128, 128, 3))
            states.append(ex["states"])
            actions.append(ex["actions"])
            is_first.append(ex["is_first"])
            is_terminal.append(ex["is_terminal"])
            lengths.append(ex["length"])
            answers.append(ex["answer"])
        image_batch = images if any(img.size > 0 for img in images) else None
        return _tokenize_dialogs(
            dialogs,
            image_batch,
            states,
            actions,
            is_first,
            is_terminal,
            lengths,
            self.processor,
            labels=answers,
        )


def _resolve_split_file(data_path, split):
    candidates = [
        f"{split}.h5",
        f"{split}.hdf5",
    ]
    for name in candidates:
        p = os.path.join(data_path, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find split file for '{split}' in {data_path}. Tried: {candidates}"
    )


def get_custom_dataset(dataset_config, tokenizer, split, split_ratio=0.9):
    import datasets
    import os

    # 1. Resolve paths and load metadata
    split_path = _resolve_split_file(dataset_config.data_path, split)
    file_dir = dataset_config.data_path

    # Load questions using the helper function already in your script
    questions = _load_questions(file_dir)
    # Define the path to the answers JSON
    answer_path = os.path.join(file_dir, "answers.json")

    # 2. Define the schema features.
    # Use fixed shapes (not None) so Arrow can stream-write examples as they are
    # generated instead of buffering the entire dataset before schema finalisation.
    _win = dataset_config.num_history_images + dataset_config.imagined_steps
    features = datasets.Features({
        "states": datasets.Array2D(shape=(_win, 9), dtype="float32"),
        "actions": datasets.Array2D(shape=(_win, 7), dtype="float32"),
        "is_first": datasets.Array2D(shape=(_win, 1), dtype="float32"),
        "is_terminal": datasets.Array2D(shape=(_win, 1), dtype="float32"),
        "length": datasets.Value("int32"),
        # Array4D has a reshape bug in this datasets version (flat buffer gets a
        # spurious ×_win factor).  Array2D with spatial dims flattened avoids the
        # bug and still uses pyarrow's buffer protocol (no Python list overhead).
        "images": datasets.Array2D(shape=(_win, 128 * 128 * 3), dtype="uint8"),
        "question": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "label": datasets.Value("int32"),
    })

    # 3. Define the generator wrapper
    def gen_func():
        # Use the logic class (make sure you removed 'datasets.GeneratorBasedBuilder' from its definition)
        logic = HDF5PickCubeRGBDataset(
            answer_type=dataset_config.answer_type,
            num_images=dataset_config.num_images,
            latent_mode=dataset_config.latent_mode,
            imagined_steps=dataset_config.imagined_steps,
            num_history_images=dataset_config.num_history_images,
            stride_size=dataset_config.stride_size,
            start_index=dataset_config.start_index,
            question_key=dataset_config.question_key,
        )

        # Unpack the (id, data) tuple and only yield the data
        for example_id, example_data in logic._generate_examples(
                hdf5_paths=[split_path],
                questions=questions,
                answer_path=answer_path,
                answer_type=dataset_config.answer_type,
                num_history_images=dataset_config.num_history_images,
                imagined_steps=dataset_config.imagined_steps,
                start_index=dataset_config.start_index,
                question_key=dataset_config.question_key,
        ):
            yield example_data

    # 4. Create the dataset
    # cache_dir points to /tmp so stale Arrow files from prior schema versions
    # (e.g. float32 images with Sequence/Array4D) are never reused.
    cache_dir = f"/tmp/hf_pickcube_cache_{split}"
    return datasets.Dataset.from_generator(gen_func, features=features, writer_batch_size=100, cache_dir=cache_dir)

def get_data_collator(processor, data_config=None):
    return DataCollector(processor, data_config)


def get_eval_data_collator(processor, data_config=None):
    return EvalDataCollector(processor, data_config)


