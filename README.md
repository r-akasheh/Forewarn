# latent-safety

# making conda env
```
conda create --name latent --file requirements.txt
```
# training WM 
```
python scripts/train_wm.py
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


