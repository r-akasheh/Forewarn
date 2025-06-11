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

## change the config 
take a look at the configs/wm_example_config.yaml to check the parameters. Change the experiment name and other parameters in the config.

After training, change the parameter ```from_ckpt``` in config file to load the checkpoint

```
python scripts/train_wm_real_data.py --config_path wm_example_config.yaml
```

# Things to modify

## loading the dataset
 Write a new fill_expert_dataset function in dreamer/tools.py to load your own offline data to train the model.

 You can take a look at existing fill_expert_dataet and fill_expert_dataset_robomimic for reference

Currently I am also mixing the batch from two sources of the dataset. You can remove the mixed_success_failure_sample in the train_wm.py and just call next(dataset) to sample the next sample

## parameters in config
```obs_keys```: a list of keys for camera view name to access through observation

```state_keys```: a list of state keys including robot pos, quat, gripper 

```priv_dim```: the dimension of the priviledged states (currently it is robot pos + quat + gripper + object pos)

```num_exp_trajs```: the number of demos used for training

```validation_mse_traj```: the number of demos used for evaluation

```root_dir```: directory of your dataset, the actual dataset path is (root_dir/task). You can add a function in common/utils.py to customize the dataset path depending on the env (e.g. get_robocasa_dataset_path_and_env_meta)

```task```: name of your env

```batch_length```: the horizon of the trajectory snippets, for longer temporal context, increase this


