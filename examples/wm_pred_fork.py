from robomimic.config import Config
import torch 
from gym.spaces import Box, Dict
from dreamer.dreamer import Dreamer
import numpy as np
import cv2 

import argparse
import os
import sys
import wandb
import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
# from transformers import MllamaForConditionalGeneration, MllamaProcessor
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from llama_recipes.models.mllama_model import MllamaForConditionalGenerationWM, MllamaWMProcessor, MllamaWMImageProcessor, MllamaWMConfig
from llama_recipes.models.mllama_model import MllamaConfig
from llama_recipes.utils.dataset_utils import (
    get_custom_data_collator,
    get_custom_eval_data_collator,
    get_preprocessed_dataset,
)
import json
from llama_recipes.utils.config_utils import generate_dataset_config, update_config,get_dataloader_kwargs
from llama_recipes.inference.model_utils import load_model, load_peft_model
from transformers.configuration_utils import PretrainedConfig
from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    # quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)
import torch.multiprocessing as mp
import yaml
from pathlib import Path
import fire
import dreamer.dreamer
from dreamer import tools
import ruamel.yaml as yaml
import time
import argparse
import copy 
from PIL import Image
class WMPredictor:
    def __init__(self, wm_config):
        self.wm_config = wm_config
        wm_config = Config(self.wm_config['defaults'])
        self.task_name = wm_config.task
        action_space = Box(-1, 1, shape = wm_config.action_space)
        wm_config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]
        self.img_keys = self.wm_config['defaults']['obs_keys']
        self.state_keys = [ "eef_pos", "eef_rot","eef_gripper_width",]
        obs_space = {}
        for key, value in wm_config.observation_space.items():
            if 'robot' in key:
                obs_space[key] = Box(-1, 1, shape = value)
            else: 
                obs_space[key] = Box(0, 1, shape = value)
        obs_space = Dict(obs_space)
        print('loading world model from ckpt path', wm_config.from_ckpt)
        self.wm_model = Dreamer.from_pretrained(path = wm_config.from_ckpt, obs_space = obs_space,
                                                act_space = action_space,
                                                config = wm_config,
                                                dataset = None,#success_val_dataset,
                                                logger = None,
                                                expert_dataset= None).to(torch.float32)
        self.wm_model.requires_grad_(requires_grad=False)
        self.wm_model.eval().cuda()
        ## normalizing the actions
        with open(f'/data/wm_data/{self.task_name}_data/norm_dict_abs.json', 'r') as f:
            print('loading norm_dict from', f'/data/wm_data/{self.task_name}_data/norm_dict_abs.json')
            norm_dict = json.load(f)
        self.norm_dict = norm_dict
        for key in self.norm_dict:
            self.norm_dict[key] = np.array(self.norm_dict[key])

    def _proc_image(self, img, size=(256, 256)):
        bgr_img = img[:, :, :3]

        return torch.from_numpy(bgr_img[None][None]).float().cuda()


    def _get_images_and_states(self, obs):
        """
        Return images and states given observations
        """
        images = {}
        state = np.empty(0)
        if 'state' in obs:
            state = obs['state']
        else: 
            for key in self.state_keys:
                state = np.append(state, obs[key])
        # state = obs['state']


        # cam_keys = [key for key in obs.keys() if "cam" in key]
        cam_keys = ['cam_rs', 'cam_zed_right']
        for i, key in enumerate(cam_keys):
            # if i in self.cam_indices:
            if isinstance(obs[key], tuple):
                img, ts = obs[key]
            else: 
                img = obs[key]
            print(key)
            # cv2.imshow('zed image', obs['cam_zed_right'][0])
            # cv2.waitKey(0)
            if 'zed' in key:
                for new_key in self.img_keys:
                    if 'front' in new_key:
                        images[new_key] = self._proc_image(img)
            # cur_img = self._proc_image(img)
            elif 'rs' in key:
                for new_key in self.img_keys:
                    if 'wrist' in new_key:
                        images[new_key] = self._proc_image(img)
            else: 
                raise ValueError('Invalid camera key')
        
        #tot_state = torch.from_numpy(tot_state.astype(np.float32))[None].cuda()
        state = state.astype(np.float32)#[None][None]
        
        return images, state #tot_state


    def process_data(self, obs, action_seq, normalize=True):
        
        data = {}

        images, states = self._get_images_and_states(obs)
        for key in images:
            data[key] = images[key]

        print('normalize',normalize)
        if normalize:
            states = (states - self.norm_dict['ob_min']) / (self.norm_dict['ob_max'] - self.norm_dict['ob_min'])
            states = 2 * states - 1
            states = states.astype(np.float32)
            print('states', states)
        data['state'] = torch.from_numpy(states)[None][None].cuda()
        B, T, _ = data['state'].shape

        actions = np.array(action_seq, dtype = np.float32)

        if normalize:
            actions = (actions - self.norm_dict['ac_min']) / (self.norm_dict['ac_max'] - self.norm_dict['ac_min'])
            actions = 2 * actions - 1
            actions = actions.astype(np.float32)  
            print('actions', actions)  
        data['action'] = actions[None]
        data['is_first'] = np.zeros((B, T))
        data['is_terminal'] = np.zeros((B, T))

        return data
    
    def predict(self, obs, action, normalize=True):
        data = self.process_data(obs, action, normalize=normalize) 
        pred_frames = self.wm_model._wm.pred_video_frames(data)
        ## key remapping for the pred_frames
        ## map key that has wrist to pred_cam_rs
        ## map key that has front to pred_cam_zed_right
        keys = list(pred_frames.keys())
        for key in keys:
            if 'wrist' in key:
                pred_frames['pred_cam_rs'] = pred_frames[key][0] ## remove the batch dimension
            elif 'front' in key:
                pred_frames['pred_cam_zed_right'] = pred_frames[key][0] ## remove the batch_dimension
            else:
                raise ValueError('Invalid key in pred_frames')
            ## delete the original key
            del pred_frames[key]
        ## this is passed to the log_obs to log the pred_cam_zed_right, and pred_cam_rs
        ## in logger also create a function to plot those pred and truth videos together
        return pred_frames
# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct/custom"
class VLMInference:
    def __init__(self, wm_configs, model_name = None, peft_model = None, answer_type = 'snippet'):
        self.wm_config = wm_configs
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        dataset_config = Config()
        dataset_config.latent_mode = "all"
        dataset_config.imagined_steps = 63
        dataset_config.num_images = 16
        dataset_config.num_history_images = 1
        dataset_config.sample_size = 16
        dataset_config.start_index = 35
        dataset_config.answer_type = answer_type    
        self.answer_type = answer_type
        self.img_keys = self.wm_config['defaults']['obs_keys']
        self.state_keys = [ "eef_pos", "eef_rot","eef_gripper_width",]
        self.dataset_config = dataset_config
        current_path = os.path.abspath(__file__)
        print('wm_config task', self.wm_config['defaults']['task'])
        if 'cup' in self.wm_config['defaults']['task']:
            self.task_name = 'GraspCup'
        elif 'bag' in self.wm_config['defaults']['task']:
            self.task_name = 'GraspBag'
        elif 'Fork' in self.wm_config['defaults']['task']:
            self.task_name = 'GraspFork'
        else: 
            raise ValueError('Invalid task name')
        if self.task_name == 'GraspCup':
            self.question_path = os.path.join(os.path.dirname(current_path), '../../../failure_detection/vlm/llama-recipes/recipes/quickstart/finetuning/datasets/realcup_data/questions.json')
        elif self.task_name == 'GraspBag':
            self.question_path = os.path.join(os.path.dirname(current_path), '../../../failure_detection/vlm/llama-recipes/recipes/quickstart/finetuning/datasets/realbag_data/questions.json')
        elif self.task_name == 'GraspFork':
            self.question_path = os.path.join(os.path.dirname(current_path), '../../../failure_detection/vlm/llama-recipes/recipes/quickstart/finetuning/datasets/realfork_data/questions.json')
        else: 
            raise ValueError('Invalid task name')
        
        if "meta-llama" in model_name:
            model, processor = self.load_original_model_and_processor(model_name, peft_model)
            self.num_images = 1
        else: 
            model, processor = self.load_model_and_processor(model_name, peft_model)
            model.init_dataset_config(self.dataset_config)
            self.num_images = 16

            processor.init_dataset_config(self.dataset_config)
       
        with open(f'/data/wm_data/{self.task_name}_data/norm_dict_abs.json', 'r') as f:
            norm_dict = json.load(f)
            print('self.task name', self.task_name)
        self.norm_dict = norm_dict
        for key in self.norm_dict:
            self.norm_dict[key] = np.array(self.norm_dict[key])
        self.model = model 
        self.processor = processor
 
 
    def load_original_model_and_processor(self, model_name: str, peft_model: str = None):
        """
         Load the model and processor based on the 11B or 90B model.
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
    
        model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                device_map=self.device,
            )
        if peft_model is not None:
            model = load_peft_model(model, peft_model)
        
        processor = MllamaProcessor.from_pretrained(model_name, use_safetensors=True)

        model, processor = self.accelerator.prepare(model, processor)
        return model, processor
    def load_model_and_processor(self, model_name: str, peft_model: str = None):
        """
        Load the model and processor based on the 11B or 90B model.
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        config = MllamaConfig.from_pretrained(model_name)

        config.wm_config = self.wm_config

    
        model = MllamaForConditionalGenerationWM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                config = config,
                use_safetensors=True,
                device_map=self.device,
            )

        if peft_model is not None:
            model = load_peft_model(model, peft_model)
    
        processor = MllamaWMProcessor.from_pretrained(model_name, device_map = self.device,  use_safetensors=True)

        processor.image_processor = MllamaWMImageProcessor.from_pretrained(model_name,device_map = self.device, use_safetensors=True, 
                                                                        config = config.wm_config ,new_type='MllamaWMImageProcessor')
        model, processor = self.accelerator.prepare(model, processor)
   
        model.initialize_vision_model()
        model.wm_model = model.wm_model.to(self.device)
        return model, processor


    def generate_text_from_image(self,
        model, processor, batch, temperature: float, top_p: float, use_sentence = False
    ):
        """
        Generate text from an image using the model and processor.
        """
        inputs = batch.to(self.device)
        temperature = 0.0
        start_time = time.time()
        outputs = model.generate(
            **inputs, 

            max_new_tokens=50 
        )
        print("the inference time is", time.time() - start_time)

        start_header_id = "<|start_header_id|>assistant<|end_header_id|>"
        input_start_header_id = "<|start_header_id|>user<|end_header_id|>"
        end_header_id = "<|eot_id|>"
        ## check if the start_header_id is in the output
        
        ## save prediction as a list
        predictions = []
        for i,output in enumerate(outputs):
            raw_output = processor.decode(output)
            raw_input = processor.decode(inputs['input_ids'][i])
            if input_start_header_id in raw_input:
                start_index = raw_input.find(input_start_header_id)
                assistant_input = raw_input[start_index + len(input_start_header_id):].strip()
                end_index = assistant_input.find(end_header_id)
                assistant_input = assistant_input[:end_index].strip().lower()

                print('Assistant Input:    ', assistant_input)
            if start_header_id in raw_output:
                start_index = raw_output.find(start_header_id)
                assistant_output = raw_output[start_index + len(start_header_id):].strip()
                end_index = assistant_output.find(end_header_id)
                cleaned_output = assistant_output[:end_index].strip().lower()
                cleaned_output = cleaned_output.replace(".", "")
                if use_sentence:
                    print('cleaned_output', cleaned_output)

                    
            predictions.append(cleaned_output)
        return predictions

    def _proc_image(self, img, size=(256, 256)):
        bgr_img = img[:, :, :3]

        return torch.from_numpy(bgr_img[None]).float()

            
    def _get_images_and_states(self, obs):
        """
        Return images and states given observations
        """
        images = {}
        state = np.empty(0)
        if 'state' in obs:
            state = obs['state']
        else: 
            for key in self.state_keys:
                state = np.append(state, obs[key])


        cam_keys = ['cam_rs', 'cam_zed_right']
        for i, key in enumerate(cam_keys):
            # if i in self.cam_indices:
            if isinstance(obs[key], tuple):
                img, ts = obs[key]
            else: 
                img = obs[key]
            print(key)

            if 'zed' in key:
                for new_key in self.img_keys:
                    if 'front' in new_key:
                        images[new_key] = self._proc_image(img)
            # cur_img = self._proc_image(img)
            elif 'rs' in key:
                for new_key in self.img_keys:
                    if 'wrist' in new_key:
                        images[new_key] = self._proc_image(img)
            else: 
                raise ValueError('Invalid camera key')
     
        state = torch.from_numpy(state.astype(np.float32))[None]
        
        return images, state 
    
    def process_data(self, obs, action_seq, normalize=False, question_key = 'grasp'):
        data = {}

        images, states = self._get_images_and_states(obs)
        T, _ = states.shape
        for key in images:
            if 'front' in key:
                front_images = images[key]
            elif 'wrist' in key:
                wrist_images = images[key]
            
        images = np.concatenate((front_images,wrist_images), axis=1)
        
        
            # data[key] = images[key]
        states = np.array(states, dtype=np.float32)
        if normalize:
            states = (states - self.norm_dict['ob_min']) / (self.norm_dict['ob_max'] - self.norm_dict['ob_min'])
            states = 2 * states - 1
        if self.answer_type !='action':
            data['is_first'] = np.zeros((64,1), dtype = np.float32)
            data['is_terminal'] = np.zeros((64,1), dtype = np.float32)
        
            data['length'] = 64
            ## add padding
            padded_length = 64 - T
            images = np.concatenate((images, np.repeat(images[-1:], padded_length, axis=0)), axis=0)
            states = np.concatenate((states, np.repeat(states[-1:], padded_length, axis=0)), axis=0)
        if self.question_path is not None:
            with open(self.question_path) as file:
                questions = json.load(file)
            if question_key is None:
                data['question'] = questions[self.answer_type]
            else:
                data['question'] = questions[f"{self.answer_type}-{question_key}"]
        ## here start with category
        if self.answer_type == 'category':
            all_images = []
            all_states = []
            all_actions = []
            all_is_first = []
            all_is_terminal = []
            all_lengths = []
            for i in range(len(action_seq)):
                all_images.extend(images)
                all_states.extend(states)
                all_is_first.extend(data['is_first'])
                all_is_terminal.extend(data['is_terminal'])
                all_lengths.append(data['length'])
                actions = np.array(action_seq[i][:64], dtype = np.float32)
                if normalize:
                    actions = (actions - self.norm_dict['ac_min']) / (self.norm_dict['ac_max'] - self.norm_dict['ac_min'])
                    actions = 2 * actions - 1
                    actions = actions.astype(np.float32)
                all_actions.append(actions)
            data['images'] = [Image.fromarray(img.astype('uint8'), 'RGB').resize((128,64)) for img in all_images]
            data['states'] = np.array(all_states).astype(np.float32)
            data['actions'] = np.array(all_actions).astype(np.float32)
            data['is_first'] = np.array(all_is_first).astype(np.float32)
            data['is_terminal'] = np.array(all_is_terminal).astype(np.float32)
            data['length'] = np.array(all_lengths).astype(np.float32)
            return [data]
        pil_images = [Image.fromarray(img.astype('uint8'), 'RGB').resize((128,64)) for img in images]

        data['images'] = pil_images
        
        if self.answer_type == 'text':
            data['question'] = data['question']['handle-new']
            print('question', data['question'])
        if 'STATE' in data['question']:
            formatted_state = f"Current Robot Gripper State: {','.join(map(str, states[0]))} \n"
            data['question'] = data['question'].replace('STATE', formatted_state)
        else: 
            data['states'] = states.astype(np.float32)
        ## convert all the keys in the data to numpy float32

        data_all = []
        for i in range(len(action_seq)):
            data_copy = copy.deepcopy(data)
            actions = np.array(action_seq[i][:64], dtype = np.float32)
            if normalize:
                actions = (actions - self.norm_dict['ac_min']) / (self.norm_dict['ac_max'] - self.norm_dict['ac_min'])
                actions = 2 * actions - 1
            
            if self.answer_type == 'action':
                formatted_action_sequence = "\n".join([f"Step {i+1}: {', '.join(map(str, row))}" for i, row in enumerate(actions)])
                data_copy['question'] = data_copy['question'].replace('ACTION_SEQUENCE', formatted_action_sequence)
            else: 
                data_copy['actions'] = actions.astype(np.float32)
            data_all.append(data_copy)
        return data_all
  
    
    def check_header(self, targets,seq):
        for i in range(len(seq)-3):
            if seq[i:i+3] in targets:
                return True
        return False
    def replace_target(self, target,seq):
        for i in range(len(seq)-3):
            if seq[i:i+3] == target:
                seq[i],seq[i+1],seq[i+2] = -100,-100,-100
          
        return seq
    
    def generate_second_dialogue(self, predictions, key='grasp-new'):
 
        with open(self.question_path) as file:
            questions = json.load(file)
        question_template = questions['text']
        question_key = key##
        question = question_template[question_key]

        # Substitute placeholders in the question
        for i, prediction in enumerate(predictions):
            question = question.replace(f'BEHAVIOR_MODE{i+1}', prediction)
        # question = question.replace('BEHAVIOR_MODE1', predictions[0]).replace('BEHAVIOR_MODE2', predictions[1])
        dialog = []
        answers = []
        # length = len(example['images'])            
        current_dialog = [ {"role":"user","content":[]},]
        current_dialog[0]["content"].append({"type": "text", "text": question.strip()})
        dialog.append(current_dialog)
        answers.append([""])
        return dialog, answers
  
    def generate_second_stage_response(self,
        model, processor, predictions, temperature: float, top_p: float, use_sentence = False, dataset_config = None,
          key = 'grasp-handle',
    ):
        """
        Generate text from an image using the model and processor.
        """
        dialog, answers = self.generate_second_dialogue(predictions, key=key)
        processor.num_images = 0
        batch = self.tokenize_dialogs(dialog, labels = answers)
        inputs = batch.to(self.device)
        temperature = 0.0
        # top_p = 0.9
        start_time = time.time()
        outputs = model.generate(
            **inputs, 

            max_new_tokens=200 
        )
        print("the second stage inference time is", time.time() - start_time)
        #  Clean up the output to remove system tokens
        ## check where  <|start_header_id|>assistant<|end_header_id|> first occur in the string output
        start_header_id = "<|start_header_id|>assistant<|end_header_id|>"
        end_header_id = "<|eot_id|>"
        input_start_header_id = "<|start_header_id|>user<|end_header_id|>"
        ## check if the start_header_id is in the output
        
        ## save prediction as a list
        predictions = []
        for i, output in enumerate(outputs):
            ## raw input 
            raw_input = processor.decode(inputs['input_ids'][i])
            raw_output = processor.decode(output)
            if input_start_header_id in raw_input:
                start_index = raw_input.find(input_start_header_id)
                assistant_input = raw_input[start_index + len(input_start_header_id):].strip()
                end_index = assistant_input.find(end_header_id)
                assistant_input = assistant_input[:end_index].strip().lower()
            
                print('Assistant Input:    ', assistant_input)
            if start_header_id in raw_output:
                start_index = raw_output.find(start_header_id)
                assistant_output = raw_output[start_index + len(start_header_id):].strip()
                end_index = assistant_output.find(end_header_id)
                cleaned_output = assistant_output[:end_index].strip().lower()
               
                if use_sentence:
                    print('Output:   ', cleaned_output)
                    print('--------------------------------')
          
                    
            predictions.append(cleaned_output)
        return assistant_input, predictions

        
    def generate_dialogs(self, samples):
        dialog = []
        images = []
   
        keys = samples[0].keys()
        if 'states' in keys:
            states = []
            actions = []
            is_first = []
            is_terminal = []
            lengths = [] 
        for example in samples:        
            if self.answer_type == 'category':
                text_split = example['question'].split('<|image|>')
                current_dialog = [ {"role":"user","content":[]},
                        ]
        
                num_image_token = 0
                for i in range(len(text_split)-1):
                    current_dialog[0]["content"].append({"type":"text", "text": text_split[i].strip()})
                    for j in range(self.num_images):
                        num_image_token += 1
                        current_dialog[0]["content"].append({"type": "image"})
                current_dialog[0]["content"].append({"type":"text", "text": text_split[-1].strip()})
                print('num of image token', num_image_token)
            else:   
                current_dialog = [ {"role":"user","content":[{"type": "image"}]},]
                ## for each image, add an image token
                for i in range(self.num_images-1):
                    current_dialog[0]["content"].append({"type": "image"})
                current_dialog[0]["content"].append({"type": "text", "text": example['question'].strip()})
            dialog.append(current_dialog)
            images.append(example["images"])
            # answers.append(example['answer'])
            if 'states' in keys:
                states.append(example['states'])
                actions.append(example['actions'])
                is_first.append(example["is_first"])
                is_terminal.append(example["is_terminal"])
                lengths.append(example["length"])
        
                batch = self.tokenize_dialogs(dialog, images, states, actions, is_first, is_terminal, lengths)
            else: 
                batch = self.tokenize_dialogs(dialog, images)
        return batch
    
   
    
    def tokenize_dialogs(self, dialogs, images=None, states=None, actions=None,  is_first = None, is_terminal=None,lengths=None, labels = None):
        text_prompt = self.processor.apply_chat_template(dialogs)
        if states is not None:
            batch = self.processor(images=images, states = states, actions = actions, is_first = is_first, is_terminal  = is_terminal, lengths = lengths,text=text_prompt,padding = True, return_tensors="pt")
        else: 
            batch = self.processor(images=images, text=text_prompt,padding = True, return_tensors="pt")

        return batch
     
    def swap_key(self, pred_frames):
        keys = list(pred_frames.keys())
        batch_size  = pred_frames[keys[0]].shape[0]
        for key in keys:
            if 'wrist' in key:
                if batch_size == 1:
                    pred_frames['pred_cam_rs'] = pred_frames[key][0] ## remove the batch dimension
                else: 
                    pred_frames['pred_cam_rs'] = pred_frames[key]
            elif 'front' in key:
                if batch_size == 1:
                    pred_frames['pred_cam_zed_right'] = pred_frames[key][0] ## remove the batch_dimension
                else: 
                    pred_frames['pred_cam_zed_right'] = pred_frames[key]
            else:
                raise ValueError('Invalid key in pred_frames')
            ## delete the original key
            del pred_frames[key]
        ## this is passed to the log_obs to log the pred_cam_zed_right, and pred_cam_rs
        ## in logger also create a function to plot those pred and truth videos together
        return pred_frames

    def infer(self, obs, action_seq, normalize=False, question_key = None):
        self.processor.num_images = 16
        samples = self.process_data(obs, action_seq, normalize, question_key = question_key)
        batch = self.generate_dialogs(samples)
        
        predictions = self.generate_text_from_image(self.model, self.processor, batch, 0.0, 0.9)
        if hasattr(self.model, 'wm_model'):

            pred_frames = self.model.wm_model._wm.frames
            pred_frames = self.swap_key(pred_frames)
        else:
            pred_frames = None
        # for key in pred_frames:
        #     pred_frames[key] = pred_frames[key][0]
        return predictions, pred_frames


    def infer_action_vlm(self, obs, action_seq, normalize=False):
        samples = self.process_data(obs, action_seq, normalize)
        batch = self.generate_dialogs(samples)
        predictions = self.generate_text_from_image(self.model, self.processor, batch, 0.0, 0.9)
        return predictions 
    
    def infer_two_stage(self, obs, action_seq, normalize=False, question_key = 'grasping'):
        predictions, pred_frames = self.infer(obs, action_seq, normalize, question_key = question_key)
        ## generate the second stage dialogue
        text_input, predictions_second_stage = self.generate_second_stage_response(self.model, self.processor, predictions, 0.0, 0.9)

        return predictions, text_input, predictions_second_stage, pred_frames
    def process_pred(self, predictions):
        class_labels = []
        for text in predictions:
            keywords = {
                # 2: ["fail", "unable", "struggle", "did not", "could not", "does not", "unable", "not able", "unsuccessful", "trouble", "not succeed", "not manage"],
                # 0: ["handle", "grasp handle", "by the handle"],
                # 1: ["inside", "interior", "inner", "rim", "by the rim", "within"],
                
                2: ["fail", "unable", "struggle", "did not", "could not", "does not", "cannot", "incomplete", "unsuccessful", "trouble", "not succeed", "not manage", "ineffective"],
                0: ["corner", "edge", "side", "outer", "border"],
                1: ["center", "midsection", "middle", "central", "core", "midpoint"],

            }
            class_labels.append(3)
            for label, terms in keywords.items():
                if any(term in text.lower() for term in terms):
                    class_labels[-1] = label
                    break
            
        return class_labels
