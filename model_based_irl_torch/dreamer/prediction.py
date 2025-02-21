import textwrap
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.models.base_nets import Module
from robomimic.models.vae_nets import *
from robomimic.models.obs_nets import ObservationGroupEncoder, PositionalEncoding
from dreamer.transformers import GPT_Backbone
import robomimic.models.base_nets as BaseNets
from robomimic.config.base_config import Config
import time
from tqdm import trange
from collections import defaultdict
import json
class ObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    """
    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
        image_output_activation=None,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self.image_output_activation = image_output_activation
        
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            
            if "image" in k:
                if self.image_output_activation is None:
                    self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)
                else:
                    layers = [nn.Linear(self.input_feat_dim, layer_out_dim)]
                    layers.append(self.image_output_activation())
                    self.nets[k] = nn.Sequential(*layers)
            else:
                self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.obs_shapes[k]) for k in self.obs_shapes }

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg
    
class MIMO_Transformer_Dyn(Module):
    """
    Extension to Transformer (based on GPT architecture) to accept multiple observation 
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as 
    a dictionary of observation dictionaries, with each key corresponding to an observation group.
    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating 
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """
    def __init__(
        self,
        input_dim,
        output_shapes,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        encoder_kwargs=None,
        image_output_activation=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            transformer_embed_dim (int): dimension for embeddings used by transformer
            transformer_num_layers (int): number of transformer blocks to stack
            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is 
                computed over this many partitions of the embedding dimension separately.
            transformer_context_length (int): expected length of input sequences
            transformer_activation: non-linearity for input and output layers used in transformer
            transformer_emb_dropout (float): dropout probability for embedding inputs in transformer
            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block
            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            encoder_kwargs (dict): observation encoder config
        """
        super(MIMO_Transformer_Dyn, self).__init__()
        
        # assert isinstance(input_obs_group_shapes, OrderedDict)
        # assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        # assert isinstance(output_shapes, OrderedDict)

        # self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # flat encoder output dimension
        transformer_input_dim = input_dim # TODO: need to check
        self.nets["embed_encoder"] = nn.Linear(
            transformer_input_dim, transformer_embed_dim
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(transformer_embed_dim)
        elif transformer_nn_parameter_for_timesteps:
            assert (
                not transformer_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, transformer_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(max_timestep, transformer_embed_dim)

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(transformer_embed_dim)
        
        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(transformer_emb_dropout)

        # GPT transformer
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_length=transformer_context_length,
            attn_dropout=transformer_attn_dropout,
            block_output_dropout=transformer_block_output_dropout,
            activation=transformer_activation,
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=transformer_embed_dim,
            image_output_activation=image_output_activation,
        )

        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_nn_parameter_for_timesteps = transformer_nn_parameter_for_timesteps

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.transformer_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()

        if self.transformer_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            time_embeddings = self.nets["embed_timestep"](
                timesteps
            )  # these are NOT fed into transformer, only added to the inputs.
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.transformer_embed_dim
            time_embeddings = torch.cat([time_embeddings for _ in range(num_replicates)], -1)
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        embeddings = self.nets["embed_encoder"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    def forward(self, transformer_inputs, key_padding_mask=None):
        transformer_encoder_outputs = None
        if transformer_encoder_outputs is None:
            transformer_embeddings = self.input_embedding(transformer_inputs)
            # pass encoded sequences through transformer
            inputs  = dict(batch = transformer_embeddings, key_padding_mask = key_padding_mask)
            transformer_encoder_outputs = self.nets["transformer"].forward(inputs)

        transformer_outputs = transformer_encoder_outputs
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs 

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        # msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\ntransformer={}".format(self.nets["transformer"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class MIMO_Transformer(Module):
    """
    Extension to Transformer (based on GPT architecture) to accept multiple observation 
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as 
    a dictionary of observation dictionaries, with each key corresponding to an observation group.
    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating 
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        encoder_kwargs=None,
        image_output_activation=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            transformer_embed_dim (int): dimension for embeddings used by transformer
            transformer_num_layers (int): number of transformer blocks to stack
            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is 
                computed over this many partitions of the embedding dimension separately.
            transformer_context_length (int): expected length of input sequences
            transformer_activation: non-linearity for input and output layers used in transformer
            transformer_emb_dropout (float): dropout probability for embedding inputs in transformer
            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block
            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            encoder_kwargs (dict): observation encoder config
        """
        super(MIMO_Transformer, self).__init__()
        
        # print("transformer_context_length", transformer_context_length); exit()
        
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
        )

        # flat encoder output dimension
        transformer_input_dim = self.nets["encoder"].output_shape()[0]
        self.nets["embed_encoder"] = nn.Linear(
            transformer_input_dim, transformer_embed_dim
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(transformer_embed_dim)
        elif transformer_nn_parameter_for_timesteps:
            assert (
                not transformer_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, transformer_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(max_timestep, transformer_embed_dim)

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(transformer_embed_dim)
        
        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(transformer_emb_dropout)

        # GPT transformer
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_length=transformer_context_length,
            attn_dropout=transformer_attn_dropout,
            block_output_dropout=transformer_block_output_dropout,
            activation=transformer_activation,
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=transformer_embed_dim,
            image_output_activation=image_output_activation,
        )

        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_nn_parameter_for_timesteps = transformer_nn_parameter_for_timesteps

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.transformer_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()

        if self.transformer_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            time_embeddings = self.nets["embed_timestep"](
                timesteps
            )  # these are NOT fed into transformer, only added to the inputs.
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.transformer_embed_dim
            time_embeddings = torch.cat([time_embeddings for _ in range(num_replicates)], -1)
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        embeddings = self.nets["embed_encoder"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    
    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                if inputs[obs_group][k] is None:
                    continue
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        inputs = inputs.copy()

        transformer_encoder_outputs = None
        transformer_inputs = TensorUtils.time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        assert transformer_inputs.ndim == 3  # [B, T, D]

        if transformer_encoder_outputs is None:
            transformer_embeddings = self.input_embedding(transformer_inputs)
            # pass encoded sequences through transformer
            transformer_encoder_outputs = self.nets["transformer"].forward(transformer_embeddings)

        transformer_outputs = transformer_encoder_outputs
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs

    def forward_part1(self, **inputs):
        """
        Process each set of inputs in its own observation group.
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                if inputs[obs_group][k] is None:
                    continue
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        inputs = inputs.copy()
    
        transformer_inputs = TensorUtils.time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        assert transformer_inputs.ndim == 3  # [B, T, D]
        
        return transformer_inputs
        
    def forward_part2(self, transformer_inputs):
        transformer_encoder_outputs = None
        if transformer_encoder_outputs is None:
            transformer_embeddings = self.input_embedding(transformer_inputs)
            # pass encoded sequences through transformer
            transformer_encoder_outputs = self.nets["transformer"].forward(transformer_embeddings)

        transformer_outputs = transformer_encoder_outputs
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs 

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\ntransformer={}".format(self.nets["transformer"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg
        
## reward transformer from the firework paper
class PredTransformer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 context_length=1500,
                 algo_config = None,
                 
                ):
        super().__init__()
        
        # self.algo_configs = algo_configs
        # self.algo_config = Config()
        # self.algo_config.enabled =  True
        # self.algo_config.supervise_all_steps = True,
        # self.algo_config.pred_future_acs = False
        # self.algo_config.causal =  True
        # self.algo_config.num_layers = 1 # #  6
        # self.algo_config.embed_dim = 16 # 64 #256 #512
        # self.algo_config.num_heads = 1# 4 #8
        # self.algo_config.attn_dropout = 0.05#0.1 
        # self.algo_config.emb_dropout = 0.05 #0.1
        # self.algo_config.block_output_dropout = 0.05 #0.1
        # self.algo_config.activation = "gelu"
        # self.algo_config.nn_parameter_for_timesteps = False
        # self.algo_config.sinusoidal_embedding = True
        self.algo_config = Config(algo_config)
        # self.wm_configs = wm_configs
        
        # if self.wm_configs.rew.use_action:
        #     hidden_dim = self.wm_configs.rew.action.hidden_dim
        #     action_network_dim = self.wm_configs.rew.action.output_dim
        #     num_layers = self.wm_configs.rew.action.num_layers
            
        #     self._action_network = MLP(
        #         input_dim=action_dim,
        #         output_dim=action_network_dim,
        #         layer_dims=[hidden_dim] * num_layers,
        #         activation=nn.ELU,
        #         output_activation=None,
        #         normalization=True,
        #     )
        #     input_dim = embed_dim 
        # else:
        input_dim = embed_dim

        # Three classses prediction
        output_shapes = OrderedDict(
            prediction=(2,)
        )
        
        algo_configs_transformer = deepcopy(self.algo_config)
        algo_configs_transformer.context_length = context_length # TODO: hack for now
    
        self._transformer = MIMO_Transformer_Dyn(
            input_dim=input_dim,
            output_shapes=output_shapes,
            **BaseNets.transformer_args_from_config(algo_configs_transformer),
            )

    def forward(self, embed, key_padding_mask = None):
        # if self.wm_configs.rew.use_action:
            # action = self._action_network(action)
            
        # pred = torch.cat([embed, action], dim=-1)    
        pred = embed    
        pred = self._transformer(pred, key_padding_mask = key_padding_mask)["prediction"]
        return pred 

class ClassifierTrainer:
    def __init__(self, wm, logger, config, device, num_training_steps, context_length=1500):
        self.wm = wm
        self.logger = logger
        self.device = device
        self.chunk_size = config.batch_length
        self.nets = nn.ModuleDict()
        self.nets['classifier'] = PredTransformer(
            embed_dim = config.dyn_stoch + config.dyn_deter, context_length=context_length, algo_config = config.transformer)
        self.context_length = context_length
        self.padding_mode = config.classifier_padding_mode
        self.nets = self.nets.float().to(self.device)
        self.latent_mode = config.classifier_latent_mode
        self.optim_params = {}
        self.optimizers = {}
        self.lr_schedulers = {}
        self._loss_name = 'pred_loss'
        self.optim_params['classifier'] = dict(optimizer_type = "adam", 
                                 learning_rate = dict(
                                     initial = 0.0001,
                                     decay_factor = 0.1,
                                     epoch_schedule = [],
                                     scheduler_type = 'constant',
                                     ),
                                 regularization = dict(
                                     L2 = 0.0
                                 ))
        self.max_grad_norm = 100.0
        self.optimizers['classifier'] = TorchUtils.optimizer_from_optim_params(
                            net_optim_params=self.optim_params['classifier'],
                            net=self.nets['classifier'])
                       
        
        self.lr_schedulers['classifier'] =  TorchUtils.lr_scheduler_from_optim_params(
                            net_optim_params=self.optim_params['classifier'],
                            net=self.nets['classifier'],
                            optimizer=self.optimizers['classifier'],
                            #num_training_steps=num_training_steps,
                        )
                        
        
    def confusion_matrix(self, y_true, y_pred, num_classes):
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_no_nan = torch.where(torch.isnan(conf_matrix_norm), torch.zeros_like(conf_matrix_norm), conf_matrix_norm)
        return torch.diag(conf_matrix_no_nan), conf_matrix_no_nan

    def load_batch_labels(self, batch):
        gt_labels = torch.Tensor(batch['label'][:, 0]).to(self.device).long()
        # if gt_labels == 1:
            # gt_labels = torch.zeros_like(gt_labels)
        return gt_labels
    
    
    def process_batch(self, batch_trajectories, mode='whole'):
        # batch_size = len(batch_trajectories)
        
        # Get the real lengths of each trajectory before padding
        # real_lengths = torch.tensor([trajectory.size(0) for trajectory in batch_trajectories])
        real_lengths = torch.Tensor(batch_trajectories['actual_length'][:, 0])
        batch_size = batch_trajectories['actual_length'].shape[0]

        # Pad all trajectories to the maximum sequence length in the batch
        max_len = self.context_length
        embed_chunks = []
        if mode == 'whole':
            for i in range(batch_size):
                chunked_inputs_dict = {
                    key: padded_inputs[i][None,:]  # Shape: (1, T, D)
                    for key, padded_inputs in batch_trajectories.items()
                }
                latent_chunk = self.wm.get_latent(chunked_inputs_dict)
                embed_chunks.append(latent_chunk)
            batch_embed = torch.cat(embed_chunks, dim=0)
        elif mode == 'split':
            # Process the sequence in chunks of size chunk_size
            for start_idx in range(0, max_len, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, max_len)
                
                # Create a chunked dictionary that will be passed to the world model
                # t = time.time()
                chunked_inputs_dict = {
                    key: padded_inputs[:, start_idx:end_idx]  # Shape: (batch_size, chunk_size, D)
                    for key, padded_inputs in batch_trajectories.items()
                }
                # print('Time taken for chunked_inputs_dict', time.time()-t)
                # t = time.time()
                # Process the chunked dictionary through the world model
                latent_chunk = self.wm.get_latent(chunked_inputs_dict, mode = self.latent_mode)
                # print('Time taken for get_latent', time.time()-t)
                embed_chunks.append(latent_chunk)
            ## Concatenate the embeddings from all chunks at the first dim
            batch_embed = torch.cat(embed_chunks, dim=1)
        elif mode == 'split_varied':
            ## test cuttng off the success trajectories
            # real_lengths = torch.Tensor(batch_trajectories['actual_length'][:, 0] - 200)
            print('real_length', real_lengths)
            # real_lengths = real_lengths - 300
            # Process the sequence in chunks of size chunk_size
            for start_idx in range(0, real_lengths.long(), self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, real_lengths.long())
                
                # Create a chunked dictionary that will be passed to the world model
                # t = time.time()
                chunked_inputs_dict = {
                    key: padded_inputs[:, start_idx:end_idx]  # Shape: (batch_size, chunk_size, D)
                    for key, padded_inputs in batch_trajectories.items()
                }
                # print('Time taken for chunked_inputs_dict', time.time()-t)
                # t = time.time()
                # Process the chunked dictionary through the world model
                latent_chunk = self.wm.get_latent(chunked_inputs_dict, mode = self.latent_mode)
                # print('Time taken for get_latent', time.time()-t)
                embed_chunks.append(latent_chunk)
            ## Concatenate the embeddings from all chunks at the first dim
            batch_embed = torch.cat(embed_chunks, dim=1)
            return batch_embed, None, real_lengths
        else: 
            raise NotImplementedError
    

        # Create the mask (True for padded positions, False for real positions)
        batch_mask = torch.arange(max_len).expand(batch_size, max_len) >= real_lengths.unsqueeze(1)
        batch_mask = batch_mask.to(self.device)

        # Pass the concatenated embeddings through the transformer with the mask
        # output = transformer(embeddings, mask=mask)
        if self.padding_mode == 'last':
            return batch_embed, None, real_lengths
        return batch_embed, batch_mask, real_lengths
        
    # # Function to sample batches and process them through the world model and transformer
    # def process_batch(self, batch_trajectories, chunk_size=16):
    #     # batch_size = len(batch_trajectories)
        
    #     # Get the real lengths of each trajectory before padding
    #     # real_lengths = torch.tensor([trajectory.size(0) for trajectory in batch_trajectories])
    #     real_lengths = batch_trajectories['actual_length'][:, 0]
    #     batch_size = batch_trajectories['actual_length'].shape[0]

    #     # Pad all trajectories to the maximum sequence length in the batch
    #     max_len = self.context_length #real_lengths.max().item()  # Get the maximum length in the batch
    #     # padded_trajectories = nn.utils.rnn.pad_sequence(batch_trajectories, batch_first=True)

    #     # Reshape the padded trajectories into chunks of size chunk_size
    #     num_chunks = (max_len + chunk_size - 1) // chunk_size  # Calculate the number of chunks needed
    #     # padded_trajectories = batch_trajectories.unfold(1, chunk_size, chunk_size)  # Shape: [batch_size, num_chunks, chunk_size, obs_dim]

    #     # Flatten so we can pass all chunks at once to the world model
    #     # flat_chunks = padded_trajectories.reshape(-1, chunk_size, padded_trajectories.size(-1))  # Shape: [batch_size * num_chunks, chunk_size, obs_dim]
    #     flat_chunks = batch_trajectories
    #     # Process all chunks at once through the world model
    #     all_chunk_embeddings = self.wm.get_latent(flat_chunks)  # Shape: [batch_size * num_chunks, chunk_size, latent_dim]

    #     # Reshape back into the original batch format: [batch_size, num_chunks, chunk_size, latent_dim]
    #     all_chunk_embeddings = all_chunk_embeddings.view(batch_size, num_chunks, chunk_size, -1)

    #     # Now flatten the chunks along the sequence dimension: [batch_size, total_seq_len, latent_dim]
    #     total_seq_len = num_chunks * chunk_size
    #     batch_embeddings = all_chunk_embeddings.view(batch_size, total_seq_len, -1)

    #     # Create the mask (True for padded positions, False for real positions)
    #     batch_mask = torch.arange(total_seq_len).expand(batch_size, total_seq_len) >= real_lengths.unsqueeze(1)

    #     # Pass the concatenated embeddings through the transformer with the mask
    #     # output = transformer(embeddings, mask=mask)
        
    #     return batch_embeddings, batch_mask

    def _compute_losses(self, predictions):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        loss_dict = OrderedDict()
        for key in predictions:
            if "loss" in key:
                loss_dict[key] = predictions[key]
                
        return loss_dict

    def forward_training(self, embed, mask, gt_labels, real_lengths):
        # embed = self.wm.get_latent(batch)
        # embed, mask = self.process_batch(batch)
        embed = TensorUtils.detach(embed)
        if mask is not None:
            embed_mask =  TensorUtils.detach(mask)
        else: 
            embed_mask = None
        pred_logits = self.nets['classifier'].forward(embed = embed, key_padding_mask=embed_mask ) 
        
        batch_size = embed.size(0)  
        ## option 1 create sparse labels for each sequence at the last step
        pred_logits_traj = pred_logits[torch.arange(batch_size), real_lengths.long()-1, :]
        ## option 2 create dense labels for each sequence from the first step to the real length step
        # pred_logits_traj = torch.concat(pred_logits[i, : real_lengths[i].long(), :] for i in range(batch_size))
        # breakpoint()
        # pred_reward_for_loss = torch.permute(pred_reward, (0,2,1))
        # reward_labels_for_loss = reward_labels
        
        pred_loss_func = nn.CrossEntropyLoss(reduction="none")
        pred_loss = pred_loss_func(pred_logits_traj, gt_labels).mean()
        
        pred_classes = torch.argmax(pred_logits_traj, dim=-1)
        overall_acc = (pred_classes == gt_labels).float().mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc, matrix = self.confusion_matrix(y_true=gt_labels, y_pred=pred_classes, num_classes=2)
        info = {'pred_loss': pred_loss, 'overall_acc': overall_acc, 'y_true': gt_labels, 'y_pred': pred_classes,
                "class_acc_fail": class_acc[0], 'class_acc_success': class_acc[1], 
                'class_acc_fnr':matrix[1,0], 'class_acc_fpr':matrix[0,1]}
        # print('info', info)
        return info
    
    def train_on_batch(self, batch, mask, gt_labels, real_lengths, validate = False):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """
        info = OrderedDict()  
        with TorchUtils.maybe_no_grad(no_grad=validate):
            

            predictions = self.forward_training(batch, mask=mask, gt_labels=gt_labels, real_lengths=real_lengths) 
                
            losses = self._compute_losses(predictions)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                    
                # gradient step
                
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["classifier"],
                    optim=self.optimizers["classifier"],
                    loss=losses[self._loss_name],
                    max_grad_norm=self.max_grad_norm,
                )
                info["policy_grad_norms"] = policy_grad_norms

                # step through optimizers
                for k in self.lr_schedulers:
                    if self.lr_schedulers[k] is not None:
                        self.lr_schedulers[k].step()
        return info
    
    def evaluate_whole_dataset(self, data_loader, epoch = None, classifier_mode = 'whole'):
        self.nets['classifier'].eval()

        # step_log_all = []
        # timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[])
        # start_time = time.time()
        data_loader_iter = data_loader #iter(data_loader)
        info_all = defaultdict(list)
        iter = 0
        while True:
            try:
                batch = next(data_loader_iter)
            except RuntimeError:
                break
            input_batch, input_mask, real_lengths = self.process_batch(batch, mode=classifier_mode)
            gt_labels = self.load_batch_labels(batch)
            with TorchUtils.maybe_no_grad(no_grad=True):
                info = self.forward_training(input_batch, input_mask, gt_labels=gt_labels, real_lengths = real_lengths)
    
            del batch  # dont need to keep these
        # torch.cuda.empty_cache()
            # timing_stats["Train_Batch"].append(time.time() - t)
            # print('Evaluating time', timing_stats["Train_Batch"][-1])
            ## log
            print('Iteration', iter)
            iter += 1
            if info['y_pred'] != info['y_true']:
                print('info', info)
            info_all['y_pred'].extend(info['y_pred'].cpu())
            info_all['y_true'].extend(info['y_true'].cpu()) 
        overall_acc = (torch.Tensor(info_all['y_pred']) == torch.Tensor(info_all['y_true'])).float().mean()
        class_acc, matrix = self.confusion_matrix(y_true=torch.Tensor(info_all['y_true']), y_pred=torch.Tensor(info_all['y_pred']), num_classes=2)
        self.logger.scalar(f"Eval/overall_acc", overall_acc)
        self.logger.scalar(f"Eval/class_acc_fail", class_acc[0])
        self.logger.scalar(f"Eval/class_acc_success", class_acc[1])
        self.logger.scalar(f"Eval/class_acc_fnr", matrix[1,0])
        self.logger.scalar(f"Eval/class_acc_fpr", matrix[0,1])
        print('Eval', 'overall_acc', overall_acc)
        print('Eval', 'class_acc_fail', class_acc[0])
        print('Eval', 'class_acc_success', class_acc[1])
        print('Eval', 'class_acc_fnr', matrix[1,0])
        print('Eval', 'class_acc_fpr', matrix[0,1])
        #     for key in info.keys():
        #         if isinstance(info[key], dict):
        #             for sub_key in info[key].keys():
        #                 info_all[key+'/'+sub_key].append(info[key][sub_key].cpu().numpy())
        #                 # self.logger.scalar(f"Eval/{key}/{sub_key}", info[key][sub_key])
        #         # else:
        #         else: 
        #             info_all[key].append(info[key].cpu().numpy())
        #             # self.logger.scalar(f"Eval/{key}", info[key])
        #     # for key in timing_stats.keys():
        #         # self.logger.scalar(f"timing/{key}", np.mean(timing_stats[key]))
        # for key in info_all.keys():
        #     if isinstance(info_all[key], dict):
        #         for sub_key in info_all[key].keys():
        #             self.logger.scalar(f"Eval/{key}/{sub_key}", np.mean(info_all[key][sub_key]))
        #             print('Eval', key, sub_key, np.mean(info_all[key][sub_key]))
        #     self.logger.scalar(f"Eval/{key}", np.mean(info_all[key]))
        #     print('Eval', key, np.mean(info_all[key]))
        self.logger.write(step=epoch)
        return 
    
    def train_classifier(self, validate = False, num_steps = None,  epoch = None, data_loader = None, classifier_mode = 'split'):
        # epoch_timestamp = time.time()
        if validate:
            self.nets['classifier'].eval()
        else:
            self.nets['classifier'].train()
        if num_steps is None:
            num_steps = len(data_loader)

        # step_log_all = []
        timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[])
        # start_time = time.time()

        data_loader_iter = data_loader #iter(data_loader)
        if not validate:
            ## training loop through the whole training set
            for _step in trange(
                num_steps,
                desc="Classifier training",
                ncols=0,
                leave=False,
            ):

                # load next batch from data loader
                try:
                    t = time.time()
                    batch = next(data_loader_iter)
                except StopIteration:
                    # reset for next dataset pass
                    data_loader_iter = iter(data_loader)
                    t = time.time()
                    batch = next(data_loader_iter)
                timing_stats["Data_Loading"].append(time.time() - t)
                print('loading time', timing_stats["Data_Loading"][-1])
                # process batch for training
                t = time.time()
                # input_batch = model.process_batch_for_training(batch)
                input_batch, input_mask, real_lengths= self.process_batch(batch, mode=classifier_mode)
                gt_labels = self.load_batch_labels(batch)
                # input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
                timing_stats["Process_Batch"].append(time.time() - t)
                print('processing time', timing_stats["Process_Batch"][-1])
                # forward and backward pass
                t = time.time()
                
            
                info = self.train_on_batch(input_batch, input_mask, gt_labels=gt_labels, real_lengths = real_lengths,validate=validate)
                timing_stats["Train_Batch"].append(time.time() - t)
                print('training time', timing_stats["Train_Batch"][-1])
                for key in info.keys():
                    
                    if isinstance(info[key], dict):
                        for sub_key in info[key].keys():
                            if 'y_pred' in sub_key or 'y_true' in sub_key:
                                continue
                            self.logger.scalar(f"{key}/{sub_key}", info[key][sub_key])
                    else:
                        self.logger.scalar(key, info[key])
                    
                for key in timing_stats.keys():
                    self.logger.scalar(f"timing/{key}", np.mean(timing_stats[key]))
                self.logger.write(step=_step+ epoch * num_steps)
            return num_steps * epoch + num_steps -1
        else: 
            ## evaluate sample one batch from the validation set
            # self.evaluate_whole_dataset(data_loader=data_loader, epoch = epoch, classifier_mode = classifier_mode)
            t = time.time()
            batch = next(data_loader_iter)
            timing_stats["Data_Loading"].append(time.time() - t)
            print('loading time', timing_stats["Data_Loading"][-1])
            # process batch for eval
            t = time.time() 
            input_batch, input_mask, real_lengths = self.process_batch(batch)
            gt_labels = self.load_batch_labels(batch)
            timing_stats["Process_Batch"].append(time.time() - t)
            print('processing time', timing_stats["Process_Batch"][-1])
            # forward pass
            t = time.time()
            # info = self.forward_training(input_batch, input_mask, gt_labels=gt_labels, real_lengths = real_lengths)
            info = self.train_on_batch(input_batch, input_mask, gt_labels=gt_labels, real_lengths = real_lengths,validate=validate)
            timing_stats["Train_Batch"].append(time.time() - t)
            print('Evaluating time', timing_stats["Train_Batch"][-1])
            ## log
            for key in info.keys():
                if isinstance(info[key], dict):
                    for sub_key in info[key].keys():
                        
                        if 'y_pred' in sub_key or 'y_true' in sub_key:
                            continue
                        self.logger.scalar(f"Eval/{key}/{sub_key}", info[key][sub_key])
                else:

                    self.logger.scalar(f"Eval/{key}", info[key])
            for key in timing_stats.keys():
                self.logger.scalar(f"timing/{key}", np.mean(timing_stats[key]))
            self.logger.write(step=num_steps)
            return None
            
            

class ClassifierLatentTrainer:
    def __init__(self, wm, logger, config, device, num_training_steps, context_length=16):
        self.wm = wm
        self.logger = logger
        self.device = device
        self.chunk_size = config.batch_length
        self.nets = nn.ModuleDict()
        self.nets['classifier'] = PredTransformer(
            embed_dim = config.dyn_stoch + config.dyn_deter, context_length=context_length, algo_config = config.transformer)
        self.context_length = context_length
        self.padding_mode = config.classifier_padding_mode
        self.nets = self.nets.float().to(self.device)
        self.latent_mode = config.classifier_latent_mode
        self.optim_params = {}
        self.optimizers = {}
        self.lr_schedulers = {}
        self._loss_name = 'pred_loss'
        self.optim_params['classifier'] = dict(optimizer_type = "adam", 
                                 learning_rate = dict(
                                     initial = 0.0001,
                                     decay_factor = 0.1,
                                     epoch_schedule = [],
                                     scheduler_type = 'constant',
                                     ),
                                 regularization = dict(
                                     L2 = 0.0
                                 ))
        self.max_grad_norm = 100.0
        self.optimizers['classifier'] = TorchUtils.optimizer_from_optim_params(
                            net_optim_params=self.optim_params['classifier'],
                            net=self.nets['classifier'])
                       
        
        self.lr_schedulers['classifier'] =  TorchUtils.lr_scheduler_from_optim_params(
                            net_optim_params=self.optim_params['classifier'],
                            net=self.nets['classifier'],
                            optimizer=self.optimizers['classifier'],
                            #num_training_steps=num_training_steps,
                        )
                        
        
    def confusion_matrix(self, y_true, y_pred, num_classes):
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_no_nan = torch.where(torch.isnan(conf_matrix_norm), torch.zeros_like(conf_matrix_norm), conf_matrix_norm)
        return torch.diag(conf_matrix_no_nan), conf_matrix_no_nan

    def load_batch_labels(self, batch):
        ## 0 and 3 for 1, 1 and 2 for 0  
        gt_labels = torch.Tensor(batch['label'][:, 0]).to(self.device).long()
        gt_labels = torch.where((gt_labels == 0) | (gt_labels == 3), 1, 
                                torch.where((gt_labels == 1) | (gt_labels == 2), 0, gt_labels))

        ## 0 and 3 for 0, 1 and 2 for 1
        
        # if gt_labels == 1:
            # gt_labels = torch.zeros_like(gt_labels)
        return gt_labels
    
    
    def process_batch(self, batch_trajectories, mode='whole'):
        # batch_size = len(batch_trajectories)
        
        # Get the real lengths of each trajectory before padding
        # real_lengths = torch.tensor([trajectory.size(0) for trajectory in batch_trajectories])
        # real_lengths = torch.Tensor(batch_trajectories['actual_length'][:, 0])
        batch_size = batch_trajectories['actual_length'].shape[0]
        # shape = batch_trajectories['actual_length'].shape
        actual_lengths = torch.ones((batch_size,))
        actual_lengths[:] = 64
        actual_lengths = actual_lengths.to(self.device)
        assert batch_trajectories['action'].shape[1] == 64, batch_trajectories['actions'].shape
        # Pad all trajectories to the maximum sequence length in the batch
        # max_len = self.context_length
        # embed_chunks = []
        
        latents = self.wm.get_latent(batch_trajectories, imagined_steps = 63, sample_size=16, actual_lengths = actual_lengths)
        ## downsample to 16 
        # downsampled_latents = latents[:, ::16, :]
        # if mode == 'whole':
        #     for i in range(batch_size):
        #         chunked_inputs_dict = {
        #             key: padded_inputs[i][None,:]  # Shape: (1, T, D)
        #             for key, padded_inputs in batch_trajectories.items()
        #         }
        #         latent_chunk = self.wm.get_latent(chunked_inputs_dict)
        #         embed_chunks.append(latent_chunk)
        #     batch_embed = torch.cat(embed_chunks, dim=0)
        # elif mode == 'split':
        #     # Process the sequence in chunks of size chunk_size
        #     for start_idx in range(0, max_len, self.chunk_size):
        #         end_idx = min(start_idx + self.chunk_size, max_len)
                
        #         # Create a chunked dictionary that will be passed to the world model
        #         # t = time.time()
        #         chunked_inputs_dict = {
        #             key: padded_inputs[:, start_idx:end_idx]  # Shape: (batch_size, chunk_size, D)
        #             for key, padded_inputs in batch_trajectories.items()
        #         }
        #         # print('Time taken for chunked_inputs_dict', time.time()-t)
        #         # t = time.time()
        #         # Process the chunked dictionary through the world model
        #         latent_chunk = self.wm.get_latent(chunked_inputs_dict, mode = self.latent_mode)
        #         # print('Time taken for get_latent', time.time()-t)
        #         embed_chunks.append(latent_chunk)
        #     ## Concatenate the embeddings from all chunks at the first dim
        #     batch_embed = torch.cat(embed_chunks, dim=1)
        # elif mode == 'split_varied':
        #     ## test cuttng off the success trajectories
        #     # real_lengths = torch.Tensor(batch_trajectories['actual_length'][:, 0] - 200)
        #     print('real_length', real_lengths)
        #     # real_lengths = real_lengths - 300
        #     # Process the sequence in chunks of size chunk_size
        #     for start_idx in range(0, real_lengths.long(), self.chunk_size):
        #         end_idx = min(start_idx + self.chunk_size, real_lengths.long())
                
        #         # Create a chunked dictionary that will be passed to the world model
        #         # t = time.time()
        #         chunked_inputs_dict = {
        #             key: padded_inputs[:, start_idx:end_idx]  # Shape: (batch_size, chunk_size, D)
        #             for key, padded_inputs in batch_trajectories.items()
        #         }
        #         # print('Time taken for chunked_inputs_dict', time.time()-t)
        #         # t = time.time()
        #         # Process the chunked dictionary through the world model
        #         latent_chunk = self.wm.get_latent(chunked_inputs_dict, mode = self.latent_mode)
        #         # print('Time taken for get_latent', time.time()-t)
        #         embed_chunks.append(latent_chunk)
        #     ## Concatenate the embeddings from all chunks at the first dim
        #     batch_embed = torch.cat(embed_chunks, dim=1)
        #     return batch_embed, None, real_lengths
        # else: 
        #     raise NotImplementedError
    

        # Create the mask (True for padded positions, False for real positions)
        # batch_mask = torch.arange(max_len).expand(batch_size, max_len) >= real_lengths.unsqueeze(1)
        ## 1 for all positions as batched_embed
        batch_mask = torch.ones_like(latents)
        assert latents.shape[1] == 16, latents.shape
        batch_mask = batch_mask.to(self.device)
        batch_embed = latents

        # Pass the concatenated embeddings through the transformer with the mask
        # output = transformer(embeddings, mask=mask)
        if self.padding_mode == 'last':
            return batch_embed, None
        return batch_embed, batch_mask
        
    # # Function to sample batches and process them through the world model and transformer
    # def process_batch(self, batch_trajectories, chunk_size=16):
    #     # batch_size = len(batch_trajectories)
        
    #     # Get the real lengths of each trajectory before padding
    #     # real_lengths = torch.tensor([trajectory.size(0) for trajectory in batch_trajectories])
    #     real_lengths = batch_trajectories['actual_length'][:, 0]
    #     batch_size = batch_trajectories['actual_length'].shape[0]

    #     # Pad all trajectories to the maximum sequence length in the batch
    #     max_len = self.context_length #real_lengths.max().item()  # Get the maximum length in the batch
    #     # padded_trajectories = nn.utils.rnn.pad_sequence(batch_trajectories, batch_first=True)

    #     # Reshape the padded trajectories into chunks of size chunk_size
    #     num_chunks = (max_len + chunk_size - 1) // chunk_size  # Calculate the number of chunks needed
    #     # padded_trajectories = batch_trajectories.unfold(1, chunk_size, chunk_size)  # Shape: [batch_size, num_chunks, chunk_size, obs_dim]

    #     # Flatten so we can pass all chunks at once to the world model
    #     # flat_chunks = padded_trajectories.reshape(-1, chunk_size, padded_trajectories.size(-1))  # Shape: [batch_size * num_chunks, chunk_size, obs_dim]
    #     flat_chunks = batch_trajectories
    #     # Process all chunks at once through the world model
    #     all_chunk_embeddings = self.wm.get_latent(flat_chunks)  # Shape: [batch_size * num_chunks, chunk_size, latent_dim]

    #     # Reshape back into the original batch format: [batch_size, num_chunks, chunk_size, latent_dim]
    #     all_chunk_embeddings = all_chunk_embeddings.view(batch_size, num_chunks, chunk_size, -1)

    #     # Now flatten the chunks along the sequence dimension: [batch_size, total_seq_len, latent_dim]
    #     total_seq_len = num_chunks * chunk_size
    #     batch_embeddings = all_chunk_embeddings.view(batch_size, total_seq_len, -1)

    #     # Create the mask (True for padded positions, False for real positions)
    #     batch_mask = torch.arange(total_seq_len).expand(batch_size, total_seq_len) >= real_lengths.unsqueeze(1)

    #     # Pass the concatenated embeddings through the transformer with the mask
    #     # output = transformer(embeddings, mask=mask)
        
    #     return batch_embeddings, batch_mask

    def _compute_losses(self, predictions):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        loss_dict = OrderedDict()
        for key in predictions:
            if "loss" in key:
                loss_dict[key] = predictions[key]
                
        return loss_dict

    def forward_training(self, embed, mask, gt_labels):
        # embed = self.wm.get_latent(batch)
        # embed, mask = self.process_batch(batch)
        embed = TensorUtils.detach(embed)
        if mask is not None:
            embed_mask =  TensorUtils.detach(mask)
        else: 
            embed_mask = None
        pred_logits = self.nets['classifier'].forward(embed = embed, key_padding_mask=embed_mask ) 
        
        # batch_size = embed.size(0)  
        ## option 1 create sparse labels for each sequence at the last step
        ## get the last step logits
        pred_logits_traj = pred_logits[:, -1, :]
        # pred_logits_traj = pred_logits[torch.arange(batch_size), real_lengths.long()-1, :]
        ## option 2 create dense labels for each sequence from the first step to the real length step
        # pred_logits_traj = torch.concat(pred_logits[i, : real_lengths[i].long(), :] for i in range(batch_size))
        # breakpoint()
        # pred_reward_for_loss = torch.permute(pred_reward, (0,2,1))
        # reward_labels_for_loss = reward_labels
        
        pred_loss_func = nn.CrossEntropyLoss(reduction="none")
        pred_loss = pred_loss_func(pred_logits_traj, gt_labels).mean()
        
        pred_classes = torch.argmax(pred_logits_traj, dim=-1)
        overall_acc = (pred_classes == gt_labels).float().mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc, matrix = self.confusion_matrix(y_true=gt_labels, y_pred=pred_classes, num_classes=2)
        info = {'pred_loss': pred_loss, 'overall_acc': overall_acc, 'y_true': gt_labels, 'y_pred': pred_classes,
                "class_acc_fail": class_acc[0], 'class_acc_success': class_acc[1], 
                'class_acc_fnr':matrix[1,0], 'class_acc_fpr':matrix[0,1]}
        # print('info', info)
        return info
    
    def train_on_batch(self, batch, mask, gt_labels, validate = False):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """
        info = OrderedDict()  
        with TorchUtils.maybe_no_grad(no_grad=validate):
            

            predictions = self.forward_training(batch, mask=mask, gt_labels=gt_labels) 
                
            losses = self._compute_losses(predictions)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                    
                # gradient step
                
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["classifier"],
                    optim=self.optimizers["classifier"],
                    loss=losses[self._loss_name],
                    max_grad_norm=self.max_grad_norm,
                )
                info["policy_grad_norms"] = policy_grad_norms

                # step through optimizers
                for k in self.lr_schedulers:
                    if self.lr_schedulers[k] is not None:
                        self.lr_schedulers[k].step()
        return info

    def evaluate_whole_dataset_in_training(self, data_loader, epoch = None, classifier_mode = 'whole'):
        self.nets['classifier'].eval()

        # step_log_all = []
        # timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[])
        # start_time = time.time()
        data_loader_iter = data_loader #iter(data_loader)
        info_all = defaultdict(list)
        # iter = 0
        while True:
            try:
                batch = next(data_loader_iter)
            except RuntimeError:
                break
            input_batch, input_mask= self.process_batch(batch, mode=classifier_mode)
            gt_labels = self.load_batch_labels(batch)
            with TorchUtils.maybe_no_grad(no_grad=True):
                info = self.forward_training(input_batch, input_mask, gt_labels=gt_labels)
    
            del batch  # dont need to keep these
        # torch.cuda.empty_cache()
            # timing_stats["Train_Batch"].append(time.time() - t)
            # print('Evaluating time', timing_stats["Train_Batch"][-1])
            ## log
            # print('Iteration', iter)
            # iter += 1
            # if info['y_pred'] != info['y_true']:
                # print('info', info)
            info_all['y_pred'].extend(info['y_pred'].cpu())
            info_all['y_true'].extend(info['y_true'].cpu()) 
        overall_acc = (torch.Tensor(info_all['y_pred']) == torch.Tensor(info_all['y_true'])).float().mean()
        class_acc, matrix = self.confusion_matrix(y_true=torch.Tensor(info_all['y_true']), y_pred=torch.Tensor(info_all['y_pred']), num_classes=2)
        self.logger.scalar(f"Eval/overall_acc", overall_acc)
        self.logger.scalar(f"Eval/class_acc_fail", class_acc[0])
        self.logger.scalar(f"Eval/class_acc_success", class_acc[1])
        self.logger.scalar(f"Eval/class_acc_fnr", matrix[1,0])
        self.logger.scalar(f"Eval/class_acc_fpr", matrix[0,1])
        print('Eval', 'overall_acc', overall_acc)
        print('Eval', 'class_acc_fail', class_acc[0])
        print('Eval', 'class_acc_success', class_acc[1])
        print('Eval', 'class_acc_fnr', matrix[1,0])
        print('Eval', 'class_acc_fpr', matrix[0,1])
        #     for key in info.keys():
        #         if isinstance(info[key], dict):
        #             for sub_key in info[key].keys():
        #                 info_all[key+'/'+sub_key].append(info[key][sub_key].cpu().numpy())
        #                 # self.logger.scalar(f"Eval/{key}/{sub_key}", info[key][sub_key])
        #         # else:
        #         else: 
        #             info_all[key].append(info[key].cpu().numpy())
        #             # self.logger.scalar(f"Eval/{key}", info[key])
        #     # for key in timing_stats.keys():
        #         # self.logger.scalar(f"timing/{key}", np.mean(timing_stats[key]))
        # for key in info_all.keys():
        #     if isinstance(info_all[key], dict):
        #         for sub_key in info_all[key].keys():
        #             self.logger.scalar(f"Eval/{key}/{sub_key}", np.mean(info_all[key][sub_key]))
        #             print('Eval', key, sub_key, np.mean(info_all[key][sub_key]))
        #     self.logger.scalar(f"Eval/{key}", np.mean(info_all[key]))
        #     print('Eval', key, np.mean(info_all[key]))
        self.logger.write(step=epoch)
        return 
    def process_data(self, data):
        norm_path = '/data/wm_data/GraspCup_data/norm_dict_abs.json'
        with open(norm_path, 'r') as f:
            norm_dict = json.load(f)
        for key in norm_dict.keys():
            norm_dict[key] = np.array(norm_dict[key])
        state = data['obs']['state'][35:36]
        state = (state - norm_dict['ob_min']) / (norm_dict['ob_max'] - norm_dict['ob_min'])
        state = 2* state - 1
        actions = data['actions_abs'][35: 35 + 64]
        actions = (actions - norm_dict['ac_min']) / (norm_dict['ac_max'] - norm_dict['ac_min'])
        actions = 2* actions - 1
        front_image = data['obs']['cam_front_view_image'][35: 36][None]
        wrist_image = data['obs']['cam_wrist_view_image'][35: 36][None]
        input = {}
        size = 64
        input['action'] = actions[None]
        input['state'] = state[None]
        input['cam_front_view_image'] = front_image
        input['cam_wrist_view_image'] = wrist_image
        input['is_first'] = np.array([1] + [0]*(64-1), dtype=np.bool_)[None]
        # input['is_last'] = np.array([0]*size, dtype=np.bool_)
        input['is_terminal'] = np.array([0]*size, dtype=np.bool_)[None]
        input['label'] = data.attrs['label']
        return input
    def predict_labels(self, data):
        self.nets['classifier'].eval()
        with TorchUtils.maybe_no_grad(no_grad=True):
            # input_batch, input_mask = self.process_batch(data, mode='whole')
            input = self.process_data(data)
            embed = self.wm.get_latent(input, imagined_steps = 63, sample_size=16, actual_lengths = torch.ones((1,)).to(self.device)*64)
            pred_logits = self.nets['classifier'].forward(embed = embed, key_padding_mask=None)
            pred_logits_traj = pred_logits[:, -1, :]
            pred_classes = torch.argmax(pred_logits_traj, dim=-1)
            label = 1 if data.attrs['label'] in [0, 3] else 0
            print('pred_classes', pred_classes)
            print('label', label)
            print('pred_logits', pred_logits_traj)
        return label, pred_classes
    def evaluate_whole_dataset(self, data_loader, epoch = None, classifier_mode = 'whole'):
        self.nets['classifier'].eval()

        # step_log_all = []
        # timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[])
        # start_time = time.time()
        data_loader_iter = data_loader #iter(data_loader)
        info_all = defaultdict(list)
        iter = 0
        while True:
            try:
                batch = next(data_loader_iter)
            except RuntimeError:
                break
            input_batch, input_mask= self.process_batch(batch, mode=classifier_mode)
            gt_labels = self.load_batch_labels(batch)
            with TorchUtils.maybe_no_grad(no_grad=True):
                info = self.forward_training(input_batch, input_mask, gt_labels=gt_labels)
    
            del batch  # dont need to keep these
        # torch.cuda.empty_cache()
            # timing_stats["Train_Batch"].append(time.time() - t)
            # print('Evaluating time', timing_stats["Train_Batch"][-1])
            ## log
            print('Iteration', iter)
            iter += 1
            if info['y_pred'] != info['y_true']:
                print('info', info)
            info_all['y_pred'].extend(info['y_pred'].cpu())
            info_all['y_true'].extend(info['y_true'].cpu()) 
        overall_acc = (torch.Tensor(info_all['y_pred']) == torch.Tensor(info_all['y_true'])).float().mean()
        class_acc, matrix = self.confusion_matrix(y_true=torch.Tensor(info_all['y_true']), y_pred=torch.Tensor(info_all['y_pred']), num_classes=2)
        self.logger.scalar(f"Eval/overall_acc", overall_acc)
        self.logger.scalar(f"Eval/class_acc_fail", class_acc[0])
        self.logger.scalar(f"Eval/class_acc_success", class_acc[1])
        self.logger.scalar(f"Eval/class_acc_fnr", matrix[1,0])
        self.logger.scalar(f"Eval/class_acc_fpr", matrix[0,1])
        print('Eval', 'overall_acc', overall_acc)
        print('Eval', 'class_acc_fail', class_acc[0])
        print('Eval', 'class_acc_success', class_acc[1])
        print('Eval', 'class_acc_fnr', matrix[1,0])
        print('Eval', 'class_acc_fpr', matrix[0,1])
        #     for key in info.keys():
        #         if isinstance(info[key], dict):
        #             for sub_key in info[key].keys():
        #                 info_all[key+'/'+sub_key].append(info[key][sub_key].cpu().numpy())
        #                 # self.logger.scalar(f"Eval/{key}/{sub_key}", info[key][sub_key])
        #         # else:
        #         else: 
        #             info_all[key].append(info[key].cpu().numpy())
        #             # self.logger.scalar(f"Eval/{key}", info[key])
        #     # for key in timing_stats.keys():
        #         # self.logger.scalar(f"timing/{key}", np.mean(timing_stats[key]))
        # for key in info_all.keys():
        #     if isinstance(info_all[key], dict):
        #         for sub_key in info_all[key].keys():
        #             self.logger.scalar(f"Eval/{key}/{sub_key}", np.mean(info_all[key][sub_key]))
        #             print('Eval', key, sub_key, np.mean(info_all[key][sub_key]))
        #     self.logger.scalar(f"Eval/{key}", np.mean(info_all[key]))
        #     print('Eval', key, np.mean(info_all[key]))
        self.logger.write(step=epoch)
        return 
    
    def train_classifier(self, validate = False, num_steps = None,  epoch = None, data_loader = None, classifier_mode = 'split'):
        # epoch_timestamp = time.time()
        if validate:
            self.nets['classifier'].eval()
        else:
            self.nets['classifier'].train()
        if num_steps is None:
            num_steps = len(data_loader)

        # step_log_all = []
        timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[])
        # start_time = time.time()

        data_loader_iter = data_loader #iter(data_loader)
        if not validate:
            ## training loop through the whole training set
            for _step in trange(
                num_steps,
                desc="Classifier training",
                ncols=0,
                leave=False,
            ):

                # load next batch from data loader
                try:
                    t = time.time()
                    batch = next(data_loader_iter)
                except StopIteration:
                    # reset for next dataset pass
                    data_loader_iter = iter(data_loader)
                    t = time.time()
                    batch = next(data_loader_iter)
                timing_stats["Data_Loading"].append(time.time() - t)
                print('loading time', timing_stats["Data_Loading"][-1])
                # process batch for training
                t = time.time()
                # input_batch = model.process_batch_for_training(batch)
                input_batch, input_mask= self.process_batch(batch, mode=classifier_mode)
                gt_labels = self.load_batch_labels(batch)
                # input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
                timing_stats["Process_Batch"].append(time.time() - t)
                print('processing time', timing_stats["Process_Batch"][-1])
                # forward and backward pass
                t = time.time()
                
            
                info = self.train_on_batch(input_batch, input_mask, gt_labels=gt_labels,validate=validate)
                timing_stats["Train_Batch"].append(time.time() - t)
                print('training time', timing_stats["Train_Batch"][-1])
                for key in info.keys():
                    
                    if isinstance(info[key], dict):
                        for sub_key in info[key].keys():
                            if 'y_pred' in sub_key or 'y_true' in sub_key:
                                continue
                            self.logger.scalar(f"{key}/{sub_key}", info[key][sub_key])
                    else:
                        self.logger.scalar(key, info[key])
                    
                for key in timing_stats.keys():
                    self.logger.scalar(f"timing/{key}", np.mean(timing_stats[key]))
                self.logger.write(step=_step+ epoch * num_steps)
            return num_steps * epoch + num_steps -1
        else: 
            ## evaluate sample one batch from the validation set
            self.evaluate_whole_dataset(data_loader=data_loader, epoch = epoch, classifier_mode = classifier_mode)
            # t = time.time()
            # batch = next(data_loader_iter)
            # timing_stats["Data_Loading"].append(time.time() - t)
            # print('loading time', timing_stats["Data_Loading"][-1])
            # # process batch for eval
            # t = time.time() 
            # input_batch, input_mask= self.process_batch(batch)
            # gt_labels = self.load_batch_labels(batch)
            # timing_stats["Process_Batch"].append(time.time() - t)
            # print('processing time', timing_stats["Process_Batch"][-1])
            # # forward pass
            # t = time.time()
            # # info = self.forward_training(input_batch, input_mask, gt_labels=gt_labels, real_lengths = real_lengths)
            # info = self.train_on_batch(input_batch, input_mask, gt_labels=gt_labels,validate=validate)
            # timing_stats["Train_Batch"].append(time.time() - t)
            # print('Evaluating time', timing_stats["Train_Batch"][-1])
            # ## log
            # for key in info.keys():
            #     if isinstance(info[key], dict):
            #         for sub_key in info[key].keys():
                        
            #             if 'y_pred' in sub_key or 'y_true' in sub_key:
            #                 continue
            #             self.logger.scalar(f"Eval/{key}/{sub_key}", info[key][sub_key])
            #     else:

            #         self.logger.scalar(f"Eval/{key}", info[key])
            # for key in timing_stats.keys():
            #     self.logger.scalar(f"timing/{key}", np.mean(timing_stats[key]))
            # self.logger.write(step=num_steps)
            return None
            
            
            