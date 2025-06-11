# latent-safety

# making conda env
```
conda create --name latent --file requirements.txt
```
# preprocess the data
save all the rollouts in hdf5 file with similar format as robomimic data

There is an additional field of attributes in data['demo_x'].attrs['label'] that stores the mode of the episode

To relabel the data, take a look at scripts/relabel_hdf5.py.

World Model needs one single hdf5 files with all rollouts and VLM finetuning needs two hdf5 that splits into train.hdf5 and test.hdf5. To split the hdf5, you can use scripts/split_hdf5.py

To compute the norm dict of the actions and states for normalization, use scripts/compute_norm_dict.py that saves a dictionary json file in the data folder that can be loaded for world model data processing.


# training WM 
## create the customized data loading function
In model_based_irl_torch/dreamer/tools.py, create a copy of fill_expert_dataset_real_data_example function and add your data processing to load the data into a buffer. Also modify the get_real_dataset_path_and_env_meta function to specify your own data path for the hdf5 files. After modification, change the corresponding function call in train_wm_real_data.py to load the data 

## loading the dataset
 Write a new fill_expert_dataset function in dreamer/tools.py to load your own offline data to train the model.

 You can take a look at existing fill_expert_dataet and fill_expert_dataset_robomimic for reference

## change the config 
take a look at the configs/wm_example_config.yaml to check the parameters. Change the experiment name and other parameters in the config.
## parameters in config
```obs_keys```: a list of keys for camera view name to access through observation

```state_keys```: a list of state keys including robot pos, quat, gripper 

```priv_dim```: the dimension of the priviledged states (currently it is robot pos + quat + gripper + object pos)

```num_exp_trajs```: the number of demos used for training

```validation_mse_traj```: the number of demos used for evaluation

```root_dir```: directory of your dataset, the actual dataset path is (root_dir/task). You can add a function in common/utils.py to customize the dataset path depending on the env (e.g. get_robocasa_dataset_path_and_env_meta)

```task```: name of your env

```batch_length```: the horizon of the trajectory snippets, for longer temporal context, increase this


After training, change the parameter ```from_ckpt``` in config file to load the checkpoint

```
python scripts/train_wm_real_data.py --config_path wm_example_config.yaml
```

# VLM finetuning
## code structure for VLM 
vlm/llama-recipes is the VLM source code

vlm/llama-recipse/recipes/quickstart/finetuning/datasets: folder to save the data for finetuning script.

## cutomize the data for a task
1. create a folder under datasets, {task_name}_data
2. put train.hdf5 and test.hdf5 in the folder and create questions.json and answers.json
3. questions.json contains the key-value pair for different kinds of questions you want VLM to answer. For generating one sentence behaivor description, use "open-word" key to access the prompt template
4. answers.json stores the supervised answer to finetuning the model to match the output. In "open-word", it contains a variation of 15 sentences for each category of behavior. During training, one answer is randomly sampled from those 15 variations to prevserve the LLM's linguistic diversity.
5. Create a {task_name}_dataset_latent.py and customize the _generate_examples function that specifies how each prompt template in questions.json is converted to the model's input. It also has similar processing of data to world model because it also needs to preprocess data for world model's prediction too.

## finetuning wm
1. take a look at the llama-recipes/src/llama-recipes/finetuning_wm.py (don't need to change this file)
2. there are some example bash scripts to launch the training, llama-recipes/run_exp_bag.sh, change the dataset related argument to your customized task
3. In llama-recipes/recipes/quickstart/inference/local_inference/llama_wm_infer.py, it runs the evaluation of the model on test.hdf5. 
4. To run the inference, you can run the following command, change the peft_model_name to the checkpoint of your model and change customized dataset accordingly
```
 CUDA_VISIBLE_DEVICES=1 python recipes/quickstart/inference/local_inference/llama_wm_infer.py --temperature 0.7 --top_p 0.9 --dataset "custom_dataset" --custom_dataset.file "recipes/quickstart/finetuning/datasets/realfork_dataset_latent.py" --custom_dataset.data_path "realfork_data" --custom_dataset.answer_type "open-word" --custom_dataset.num_images 16 --custom_dataset.sample_size 16 --custom_dataset.num_history_images 1 --custom_dataset.imagined_steps 63 --custom_dataset.latent_mode "all"  --model_name "mllama/Llama-3.2-11B-Vision-Instruct/custom" --batch_size_training 10 --custom_dataset.test_split "test" --custom_dataset.start_index 60 --peft_model_name /data/finetuned_models/run_02_21_custom_wm_150k_vlm_finetuning_0.2%_imagined_step63_1_history_16sample_size_fork_task_open-word-fork-all_18epoch_print_eval_metrics_3class_aug_failure_by2_shuffle_key_correct_prompt_hist_no_start_from_75/peft_checkpoint_18 --use_sentence True
```
5. In src/llama-recipes/models/mllama_model.py, it has the modified the VLM that replaces the original vision encoder with the world model's latent states. There are two functions to call world model to generate latent states

The first one is to generate imagined latents (which is only having current observations and future action sequences to infer future states)
``` 
batch_embeds = self.wm_model._wm.get_latent(dict, mode=self.latent_mode, imagined_steps=self.imagined_steps, actual_lengths= actual_lengths, sample_size = self.sample_size, total_steps=self.num_images)
```

The second one is to generate groundtruth latents (which is encoding the ground truth observations for the future steps into latents). This one is defaultly not in use but if you want to compare the performance according to ground truth observations, you can uncomment this line. 
```
# batch_embeds = self.wm_model._wm.get_latent_gt(dict, mode=self.latent_mode, imagined_steps=self.imagined_steps, actual_lengths= actual_lengths, sample_size = self.sample_size, total_steps=self.num_images)
```








