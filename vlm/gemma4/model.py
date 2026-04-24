from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from dreamer.dreamer import Dreamer
from gym.spaces import Box, Dict
from PIL import Image
from robomimic.config.base_config import Config
from torch import nn
from transformers import Gemma4Config, Gemma4ForConditionalGeneration
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, to_numpy_array
from transformers.models.gemma4.modeling_gemma4 import Gemma4PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


def _normalize_image_batches(images: ImageInput | None) -> list[list[Image.Image | np.ndarray]]:
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        if len(images) > 0 and isinstance(images[0], (list, tuple)):
            return [list(sample) for sample in images]
        return [list(images)]
    if isinstance(images, np.ndarray) and images.ndim == 5:
        return [[sample[t] for t in range(sample.shape[0])] for sample in images]
    return [[images]]


class MultiModalProjector(nn.Module):
    def __init__(self, embed_output_dim: int, text_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(embed_output_dim, text_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Gemma4ForConditionalGenerationWM(Gemma4ForConditionalGeneration):
    """Gemma4 conditional generation model with a frozen Dreamer world model prefix."""

    def __init__(self, config: Gemma4Config):
        super().__init__(config)

        if not hasattr(config, "wm_config"):
            raise ValueError("Gemma4ForConditionalGenerationWM requires config.wm_config")

        self.wm_config = config.wm_config
        self.embed_output_dim = config.wm_config["defaults"]["dyn_deter"] + config.wm_config["defaults"]["dyn_stoch"]
        self.text_hidden_size = config.text_config.hidden_size

        self.vision_model = None
        self.multi_modal_projector = MultiModalProjector(self.embed_output_dim, self.text_hidden_size)

        self.post_init()

    def initialize_vision_model(self):
        wm_config = Config(self.wm_config["defaults"])

        def _set_int_default(key, default):
            value = getattr(wm_config, key, None)
            if isinstance(value, Config) or value is None:
                value = default
            setattr(wm_config, key, int(value))

        _set_int_default("reward_ensemble_size", 1)
        _set_int_default("reward_ensemble_subsample", 1)
        _set_int_default("cont_ensemble_size", 1)
        _set_int_default("cont_ensemble_subsample", 1)
        _set_int_default("critic_ensemble_size", 1)
        _set_int_default("critic_ensemble_subsample", 2)

        action_space = Box(-1, 1, shape=wm_config.action_space)
        wm_config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

        obs_space = {}
        for key, value in wm_config.observation_space.items():
            obs_space[key] = Box(-1, 1, shape=value) if "robot" in key else Box(0, 1, shape=value)
        obs_space = Dict(obs_space)

        print("loading world model from ckpt path", wm_config.from_ckpt)
        self.wm_model = Dreamer.from_pretrained(
            path=wm_config.from_ckpt,
            obs_space=obs_space,
            act_space=action_space,
            config=wm_config,
            dataset=None,
            logger=None,
            expert_dataset=None,
        ).to(torch.bfloat16)
        self.wm_model.requires_grad_(False)
        self.wm_model.eval()

    def init_dataset_config(self, config):
        self.latent_mode = config.latent_mode
        self.num_history_images = config.num_history_images
        self.imagined_steps = config.imagined_steps
        self.num_images = config.num_images
        self.sample_size = config.sample_size
        self.start_index = config.start_index

    def process_batch_vision(self, images, states, actions, is_first, is_terminal, actual_lengths):
        wm_inputs = {}

        if states is None or actions is None or is_first is None or is_terminal is None:
            raise ValueError("states/actions/is_first/is_terminal are required for WM latent extraction")

        B, T = states.shape[:2]
        if T > self.num_history_images + self.imagined_steps:
            chunk = 6
            if images is not None:
                _, _, H, W, C = images.shape
                images = images.reshape(-1, T // chunk, H, W, C)
            states = states.reshape(-1, T // chunk, states.shape[-1])
            actions = actions.reshape(-1, T // chunk, actions.shape[-1])
            is_first = is_first.reshape(-1, T // chunk, 1)
            is_terminal = is_terminal.reshape(-1, T // chunk, 1)
            actual_lengths = actual_lengths.reshape(B * chunk)

        img_keys = self.wm_config["defaults"].get("obs_keys", [])
        if images is not None and len(img_keys) > 0:
            _, _, H, _, _ = images.shape
            if len(img_keys) == 1:
                wm_inputs[img_keys[0]] = images
            elif len(img_keys) == 2:
                wm_inputs[img_keys[0]] = images[:, :, : H // 2,]
                wm_inputs[img_keys[1]] = images[:, :, H // 2 :,]
            else:
                raise ValueError(f"Unsupported number of obs image keys: {len(img_keys)}")
        elif len(img_keys) > 0:
            obs_space = self.wm_config["defaults"].get("observation_space", {})
            for key in img_keys:
                shape = obs_space.get(key, [64, 64, 3])
                h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
                wm_inputs[key] = torch.zeros((states.shape[0], states.shape[1], h, w, c), device=states.device)

        wm_inputs["state"] = states
        wm_inputs["action"] = actions
        wm_inputs["is_first"] = is_first[:, :, 0]
        wm_inputs["is_terminal"] = is_terminal[:, :, 0]

        wm_dtype = next(self.wm_model.parameters()).dtype
        wm_inputs = {k: v.to(wm_dtype) if isinstance(v, torch.Tensor) else v for k, v in wm_inputs.items()}

        # Disable cuDNN for WM forward to work around CUDNN_STATUS_NOT_INITIALIZED
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            batch_embeds = self.wm_model._wm.get_latent(
                wm_inputs,
                mode=self.latent_mode,
                imagined_steps=self.imagined_steps,
                actual_lengths=actual_lengths,
                sample_size=self.sample_size,
                total_steps=self.num_images,
            )
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled

        B2, T_hat, _ = batch_embeds.shape
        if T > self.num_history_images + self.imagined_steps:
            batch_embeds = batch_embeds.reshape(B2 // 6, T_hat * 6, -1)
        return batch_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        actions=None,
        states=None,
        is_first=None,
        is_terminal=None,
        lengths=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :].clone(memory_format=torch.contiguous_format)

        model_inputs = {"input_ids": input_ids, "inputs_embeds": None}
        if inputs_embeds is not None and cache_position is not None and cache_position[0] == 0:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": inputs_embeds}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        if cache_position is not None and cache_position[0] == 0:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "actions": actions,
                    "states": states,
                    "is_first": is_first,
                    "is_terminal": is_terminal,
                    "lengths": lengths,
                }
            )

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        is_first: Optional[torch.BoolTensor] = None,
        is_terminal: Optional[torch.BoolTensor] = None,
        lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify at least one of input_ids or inputs_embeds")

        lm_input_ids = input_ids
        lm_inputs_embeds = inputs_embeds

        add_vision_prefix = (
            pixel_values is not None
            or (states is not None and actions is not None and is_first is not None and is_terminal is not None)
        ) and (past_key_values is None or cache_position is None or cache_position[0] == 0)

        if add_vision_prefix:
            text_embeds = inputs_embeds
            if text_embeds is None:
                text_embeds = self.model.get_input_embeddings()(input_ids)

            vision_outputs = self.process_batch_vision(
                images=pixel_values,
                actions=actions,
                states=states,
                is_first=is_first,
                is_terminal=is_terminal,
                actual_lengths=lengths,
            )
            latent_tokens = self.multi_modal_projector(vision_outputs.to(text_embeds.dtype))
            text_embeds = torch.cat([latent_tokens, text_embeds], dim=1)
            lm_input_ids = None
            lm_inputs_embeds = text_embeds

            if attention_mask is None:
                attention_mask = torch.ones(text_embeds.shape[:2], device=text_embeds.device, dtype=torch.long)
            else:
                latent_mask = torch.ones(
                    attention_mask.shape[0], latent_tokens.shape[1],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([latent_mask, attention_mask], dim=1)

            if labels is not None and labels.shape[1] != text_embeds.shape[1]:
                latent_labels = torch.full(
                    (labels.shape[0], latent_tokens.shape[1]),
                    -100,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                labels = torch.cat([latent_labels, labels], dim=1)

        outputs = self.model(
            input_ids=lm_input_ids,
            inputs_embeds=lm_inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        return outputs


class Gemma4WMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "states", "actions", "is_first", "is_terminal", "lengths", "num_tiles"]

    def __init__(self, do_convert_rgb: bool = True, do_resize: bool = True, config: Config | None = None, **kwargs):
        super().__init__(**kwargs)
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.max_image_tiles = 1
        self.config = config

    def init_dataset_config(self, config) -> None:
        self.latent_mode = config.latent_mode
        self.num_history_images = config.num_history_images
        self.imagined_steps = config.imagined_steps
        self.num_images = config.num_images
        self.start_index = config.start_index

    def __call__(self, images, states, actions, is_first, is_terminal, lengths, **kwargs) -> BatchFeature:
        return self.preprocess(images, states, actions, is_first, is_terminal, lengths, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        states: np.ndarray,
        actions: np.ndarray,
        is_first: np.ndarray,
        is_terminal: np.ndarray,
        lengths: np.ndarray,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        image_batches = _normalize_image_batches(images)
        if self.do_convert_rgb:
            image_batches = [
                [img.convert("RGB") if isinstance(img, PIL.Image.Image) else img for img in sample]
                for sample in image_batches
            ]

        pixel_values = [
            [to_numpy_array(img) for img in sample]
            for sample in image_batches
        ]
        pixel_values = [
            [a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32) for a in sample]
            for sample in pixel_values
        ]
        num_tiles = [[1 for _ in sample] for sample in pixel_values]

        encoded_inputs = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "states": states,
                "actions": actions,
                "is_first": is_first,
                "is_terminal": is_terminal,
                "lengths": lengths,
            },
            tensor_type=return_tensors,
        )
        encoded_inputs["num_tiles"] = num_tiles
        return encoded_inputs


class Gemma4WMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Gemma4WMImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor, tokenizer):
        self.image_token = "<|image|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.bos_token = tokenizer.bos_token
        self.chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer)

    def init_dataset_config(self, config) -> None:
        self.latent_mode = config.latent_mode
        self.num_history_images = config.num_history_images
        self.imagined_steps = config.imagined_steps
        self.num_images = config.num_images
        self.start_index = config.start_index

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        is_first: Optional[np.ndarray] = None,
        is_terminal: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        return_tensors = kwargs.pop("return_tensors", None)
        text_kwargs = kwargs.pop("text_kwargs", {})
        images_kwargs = kwargs.pop("images_kwargs", {})

        data = {}
        n_images_in_text = [0]

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            n_images_in_text = [t.count(self.image_token) for t in text]
            encoding = self.tokenizer(text, **{**text_kwargs, "return_tensors": return_tensors})
            data.update(encoding)

        n_images_in_images = [0]
        if images is not None:
            images = _normalize_image_batches(images)
            n_images_in_images = [len(sample) for sample in images]

        if images is None and text is not None and max(n_images_in_text) > 0:
            placeholder = np.zeros((128, 64, 3), dtype=np.uint8)
            images = [[PIL.Image.fromarray(placeholder, "RGB") for _ in range(n)] for n in n_images_in_text]
            n_images_in_images = [len(sample) for sample in images]

        if text is not None and images is not None:
            if any(batch_img == 0 for batch_img in n_images_in_text) and not all(batch_img == 0 for batch_img in n_images_in_text):
                raise ValueError("If a batch of text is provided, there should be either no images or at least one image per sample")
            if n_images_in_images[0] != self.num_history_images + self.imagined_steps + self.start_index and n_images_in_images[0] != self.num_history_images + self.imagined_steps and n_images_in_images[0] % (self.num_history_images + self.imagined_steps) != 0 and sum(n_images_in_images) != sum(n_images_in_text) and n_images_in_images[0] != self.num_history_images and n_images_in_images[0] != self.start_index + self.num_history_images:
                raise ValueError(
                    f"The number of image tokens (either {n_images_in_text[0]} or {self.num_history_images} or {self.num_history_images+self.imagined_steps} or {self.num_history_images + self.imagined_steps + self.start_index} or {self.start_index + self.num_history_images}) should be the same as in the number of provided images ({n_images_in_images[0]})"
                )
            if n_images_in_text[0] != self.num_images and n_images_in_text[0] % self.num_images != 0:
                raise ValueError(
                    f"The number of image tokens ({n_images_in_text[0]}) in text should be the same as the number of image latent to generate({self.num_images})"
                )

        if images is not None:
            image_features = self.image_processor(
                images,
                states,
                actions,
                is_first,
                is_terminal,
                lengths,
                **{**images_kwargs, "return_tensors": return_tensors},
            )
            data.update(image_features)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(self.tokenizer.model_input_names + self.image_processor.model_input_names)
from transformers import Gemma4ForConditionalGeneration, Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from torch import nn
import torch
import os
import warnings
from dreamer.dreamer import Dreamer
from typing import List, Optional, Tuple, Union
import numpy as np
from gym.spaces import Box, Discrete, Dict
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)
from pathlib import Path

from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_valid_image,
    is_vision_available,
    to_numpy_array,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, logging
from robomimic.config.base_config import Config
from PIL import Image
import PIL
from types import ModuleType
import sys
import importlib


class MultiModalProjector(nn.Module):
    """Projects world model embeddings to text hidden size."""
    def __init__(self, embed_output_dim, text_hidden_size):
        super().__init__()
        self.linear = nn.Linear(embed_output_dim, text_hidden_size, bias=True)

    def forward(self, x):
        return self.linear(x)


class Gemma4ForConditionalGenerationWM(Gemma4ForConditionalGeneration):
    """
    Gemma4 model with World Model vision encoder replacement.

    Instead of using the standard vision encoder, this model:
    1. Extracts latent embeddings from a Dreamer world model
    2. Projects them to text embedding dimension via a linear projector
    3. Concatenates them with text embeddings
    4. Feeds the concatenated sequence to the language model
    """

    def __init__(self, config: Gemma4Config):
        super().__init__(config)

        # Store config values for later use
        self.embed_output_dim = config.wm_config['defaults']['dyn_deter'] + config.wm_config['defaults']['dyn_stoch']
        self.text_hidden_size = config.text_config.hidden_size if hasattr(config, 'text_config') else config.hidden_size

        # Replace vision model with world model
        self.vision_model = None

        # Single linear projector (per paper)
        self.multi_modal_projector = MultiModalProjector(
            self.embed_output_dim,
            self.text_hidden_size
        )

        self.wm_config = config.wm_config
        self.post_init()

    def initialize_vision_model(self):
        """Load and initialize the Dreamer world model."""
        wm_config = Config(self.wm_config['defaults'])

        # robomimic Config may return a nested Config placeholder for missing keys.
        # Coerce ensemble knobs to ints so Dreamer comparisons are always valid.
        def _set_int_default(key, default):
            value = getattr(wm_config, key, None)
            if isinstance(value, Config) or value is None:
                value = default
            setattr(wm_config, key, int(value))

        _set_int_default("reward_ensemble_size", 1)
        _set_int_default("reward_ensemble_subsample", 1)
        _set_int_default("cont_ensemble_size", 1)
        _set_int_default("cont_ensemble_subsample", 1)
        _set_int_default("critic_ensemble_size", 1)
        _set_int_default("critic_ensemble_subsample", 2)

        action_space = Box(-1, 1, shape=wm_config.action_space)
        wm_config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

        obs_space = {}
        for key, value in wm_config.observation_space.items():
            if 'robot' in key:
                obs_space[key] = Box(-1, 1, shape=value)
            else:
                obs_space[key] = Box(0, 1, shape=value)
        obs_space = Dict(obs_space)

        print('loading world model from ckpt path', wm_config.from_ckpt)
        self.wm_model = Dreamer.from_pretrained(
            path=wm_config.from_ckpt,
            obs_space=obs_space,
            act_space=action_space,
            config=wm_config,
            dataset=None,
            logger=None,
            expert_dataset=None
        ).to(torch.bfloat16)

        self.wm_model.requires_grad_(requires_grad=False)
        self.wm_model.eval()

    def process_batch_vision(self, images, states, actions, is_first, is_terminal, actual_lengths):
        """
        Process batch inputs through world model to extract latent embeddings.

        Args:
            images: Image observations (B, T, H, W, C)
            states: State observations (B, T, state_dim)
            actions: Actions (B, T, action_dim)
            is_first: Episode start flags (B, T, 1)
            is_terminal: Episode end flags (B, T, 1)
            actual_lengths: Actual sequence lengths (B,)

        Returns:
            batch_embeds: World model latent embeddings (B, T, embed_dim)
        """
        wm_inputs = {}

        if states is None or actions is None or is_first is None or is_terminal is None:
            raise ValueError("states/actions/is_first/is_terminal are required for WM latent extraction")

        B, T = states.shape[:2]
        if T > self.num_history_images + self.imagined_steps:
            # Keep existing chunking behavior used by the original loader path.
            chunk = 6
            if images is not None:
                _, _, H, W, C = images.shape
                images = images.reshape(-1, T // chunk, H, W, C)
            states = states.reshape(-1, T // chunk, states.shape[-1])
            actions = actions.reshape(-1, T // chunk, actions.shape[-1])
            is_first = is_first.reshape(-1, T // chunk, 1)
            is_terminal = is_terminal.reshape(-1, T // chunk, 1)
            actual_lengths = actual_lengths.reshape(B * chunk)

        # Get image keys from world model config
        img_keys = self.wm_config['defaults'].get('obs_keys', [])
        if images is not None and len(img_keys) > 0:
            _, _, H, _, _ = images.shape
            if len(img_keys) == 1:
                wm_inputs[img_keys[0]] = images
            elif len(img_keys) == 2:
                wm_inputs[img_keys[0]] = images[:, :, :H // 2,]
                wm_inputs[img_keys[1]] = images[:, :, H // 2:,]
            else:
                raise ValueError(f"Unsupported number of obs image keys: {len(img_keys)}")
        elif len(img_keys) > 0:
            # State-only runtime can still use WM configs with image keys by injecting zero-images.
            obs_space = self.wm_config['defaults'].get('observation_space', {})
            for key in img_keys:
                shape = obs_space.get(key, [64, 64, 3])
                h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
                wm_inputs[key] = torch.zeros((states.shape[0], states.shape[1], h, w, c), device=states.device)

        wm_inputs['state'] = states
        wm_inputs['action'] = actions
        wm_inputs['is_first'] = is_first[:, :, 0]
        wm_inputs['is_terminal'] = is_terminal[:, :, 0]

        wm_dtype = next(self.wm_model.parameters()).dtype
        wm_inputs = {k: v.to(wm_dtype) if isinstance(v, torch.Tensor) else v for k, v in wm_inputs.items()}

        # Disable cuDNN for WM forward to work around CUDNN_STATUS_NOT_INITIALIZED
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            batch_embeds = self.wm_model._wm.get_latent(
                wm_inputs,
                mode=self.latent_mode,
                imagined_steps=self.imagined_steps,
                actual_lengths=actual_lengths,
                sample_size=self.sample_size,
                total_steps=self.num_images,
            )
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled

        B2, T_hat, _ = batch_embeds.shape
        if T > self.num_history_images + self.imagined_steps:
            batch_embeds = batch_embeds.reshape(B2 // 6, T_hat * 6, -1)

        return batch_embeds

    def init_dataset_config(self, config):
        """Initialize dataset-specific configuration for world model processing."""
        self.latent_mode = config.latent_mode
        self.num_history_images = config.num_history_images
        self.imagined_steps = config.imagined_steps
        self.num_images = config.num_images
        self.sample_size = config.sample_size
        self.start_index = config.start_index

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        is_first: Optional[torch.BoolTensor] = None,
        is_terminal: Optional[torch.BoolTensor] = None,
        lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Forward pass for Gemma4 with World Model vision encoder.

        Args:
            input_ids: Text token IDs (B, T_text)
            pixel_values: Image observations (B, T_img, H, W, C)
            states: State observations (B, T_img, state_dim)
            actions: Actions (B, T_img, action_dim)
            is_first: Episode start flags (B, T_img, 1)
            is_terminal: Episode end flags (B, T_img, 1)
            lengths: Actual sequence lengths (B,)
            attention_mask: Attention mask for text tokens
            position_ids: Position IDs
            past_key_values: Cached key/values from previous forward passes
            inputs_embeds: Pre-computed input embeddings (alternative to input_ids)
            labels: Target token IDs for loss computation
            use_cache: Whether to return cached key/values
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            cache_position: Cache position for incremental decoding
            num_logits_to_keep: Number of logits to keep (optimization)

        Returns:
            CausalLMOutputWithPast or tuple depending on return_dict
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Process world model inputs if provided
        if pixel_values is not None or (
            states is not None and actions is not None and is_first is not None and is_terminal is not None
        ):
            # 1. Get world model latents
            vision_outputs = self.process_batch_vision(
                images=pixel_values,
                actions=actions,
                states=states,
                is_first=is_first,
                is_terminal=is_terminal,
                actual_lengths=lengths,
            )

            # 2. Project latents → text dim (single linear projection)
            latent_tokens = self.multi_modal_projector(vision_outputs.to(self.dtype))
            # shape: (B, T_img, text_hidden_size)

            # 3. Get text embeddings
            text_embeds = self.model.get_input_embeddings()(input_ids)
            # shape: (B, T_text, text_hidden_size)

            # 4. Concatenate latent tokens and text embeddings change
            # latent_tokens: (B, T_img, text_hidden_size)
            # text_embeds: (B, T_text, text_hidden_size)
            inputs_embeds = torch.cat([latent_tokens, text_embeds], dim=1)

            # 5. Extend attention mask to cover latent tokens
            latent_mask = torch.ones(
                attention_mask.shape[0], latent_tokens.shape[1],
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([latent_mask, attention_mask], dim=1)

            input_ids = None  # We're providing inputs_embeds instead

        # 6. Forward through language model
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            **kwargs,
        )

        return outputs


class Gemma4WMImageProcessor(BaseImageProcessor):
    """
    Image processor for Gemma4 with World Model inputs.

    Handles preprocessing of images along with state/action/episode flags.
    """

    model_input_names = ["pixel_values", "num_tiles", "states", "actions", "is_first", "is_terminal", "lengths"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        config: Config = None,
        device: str = 'auto',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.max_image_tiles = 1

    def __call__(self, images, states, actions, is_first, is_terminal, lengths, **kwargs) -> BatchFeature:
        """Preprocess images with actions, states, and episode flags."""
        return self.preprocess(images, states, actions, is_first, is_terminal, lengths, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        states: np.ndarray,
        actions: np.ndarray,
        is_first: np.ndarray,
        is_terminal: np.ndarray,
        lengths: np.ndarray,
        do_convert_rgb: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Preprocess a batch of images with state/action information.

        Args:
            images: Image observations
            states: State observations
            actions: Actions
            is_first: Episode start flags
            is_terminal: Episode end flags
            lengths: Actual sequence lengths
            do_convert_rgb: Whether to convert to RGB
            do_resize: Whether to resize
            return_tensors: Tensor type to return

        Returns:
            BatchFeature with preprocessed data
        """
        images_list = images if isinstance(images, list) else [images]

        # Convert to RGB if needed
        if self.do_convert_rgb:
            images_list = [
                [convert_to_rgb(image) if not isinstance(image, np.ndarray) else image for image in img_batch]
                for img_batch in images_list
            ]

        # Convert to numpy arrays, normalizing uint8 to [0, 1]
        images_list = [
            [to_numpy_array(image) for image in img_batch]
            for img_batch in images_list
        ]
        images_list = [
            [a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32) for a in img_batch]
            for img_batch in images_list
        ]

        num_tiles = [[1 for _ in img_batch] for img_batch in images_list]

        encoded_inputs = BatchFeature(
            data={
                "pixel_values": images_list,
                'states': states,
                'actions': actions,
                'is_first': is_first,
                'is_terminal': is_terminal,
                'lengths': lengths,
            },
            tensor_type=return_tensors,
        )
        encoded_inputs["num_tiles"] = num_tiles

        return encoded_inputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        new_type=None,
        **kwargs,
    ):
        """Load image processor from pretrained model."""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
        if new_type is not None:
            image_processor_dict['image_processor_type'] = new_type
            image_processor_dict['config'] = kwargs.get('config')
            image_processor_dict['device'] = kwargs.get('device_map')
        return cls.from_dict(image_processor_dict, **kwargs)


class Gemma4WMProcessor(ProcessorMixin):
    """
    Processor for Gemma4 with World Model inputs.

    Combines image processor and tokenizer for handling images + text with
    world model state/action information.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Gemma4WMImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor, tokenizer):
        self.image_token = "<|image|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.bos_token = tokenizer.bos_token
        self.chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer)

    def init_dataset_config(self, config) -> None:
        """Initialize dataset-specific configuration."""
        self.latent_mode = config.latent_mode
        self.num_history_images = config.num_history_images
        self.imagined_steps = config.imagined_steps
        self.num_images = config.num_images
        self.start_index = config.start_index

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        is_first: Optional[np.ndarray] = None,
        is_terminal: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare text and images for the model.

        Args:
            images: Image observations
            states: State observations
            actions: Actions
            is_first: Episode start flags
            is_terminal: Episode end flags
            lengths: Actual sequence lengths
            text: Text prompts
            **kwargs: Additional arguments

        Returns:
            BatchFeature with processed inputs
        """
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        data = {}

        # Process text
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            encoding = self.tokenizer(text, return_tensors=kwargs.get('return_tensors', 'pt'))
            data.update(encoding)

        # Process images with world model inputs
        if images is not None:
            image_features = self.image_processor(
                images, states, actions, is_first, is_terminal, lengths,
                return_tensors=kwargs.get('return_tensors', 'pt')
            )
            num_tiles = image_features.pop("num_tiles", None)
            data.update(image_features)

        return_tensors = kwargs.get('return_tensors', None)
        batch_feature = BatchFeature(data=data, tensor_type=return_tensors)

        return batch_feature

    def batch_decode(self, *args, **kwargs):
        """Decode batch of token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names + image_processor_input_names)


def convert_to_rgb(image):
    """Convert PIL image to RGB."""
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    return image

