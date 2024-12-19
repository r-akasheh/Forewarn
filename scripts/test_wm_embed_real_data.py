
import argparse
import collections
import copy
import warnings
import functools
import time
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_based_irl_torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../safety_rl'))
sys.path.append(saferl_dir)
print(sys.path)
import numpy as np
import ruamel.yaml as yaml
import torch
from termcolor import cprint
from torch import distributions as torchd
from tqdm import trange
import torch.multiprocessing as mp
import dreamer.tools as tools
# import envs.wrappers as wrappers
from common.constants import HORIZONS, IMAGE_OBS_KEYS
from common.utils import (
    create_shape_meta,
    get_robomimic_dataset_path_and_env_meta,
    to_np,
    combine_dictionaries,
)
from common.ood_utils import train_pca_kmeans, load_pca_kmeans, plot_cluster_embedding
from dreamer.dreamer import Dreamer
# from dreamer.parallel import Damy, Parallel
#from environments.env_make import make_env_robomimic
from termcolor import cprint
from functools import partial
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from dreamer.tools import ModelEvaluator
# os.environ["MUJOCO_GL"] = "osmesa"
from gym import spaces
from gym.spaces import Discrete 
import gym

# def train_lx(ckpt_name, log_dir):
#     recon_steps = 2501
#     best_pretrain_success_classifier = float("inf")
#     lx_mlp, lx_opt = agent._wm._init_lx_mlp(config, 1)
#     train_loss = []
#     eval_loss = []
#     for i in range(recon_steps):
#         if i % 250 == 0:
#             print('eval')
#             new_loss, eval_plot = agent.train_lx(
#                 next(obs_eval_dataset), lx_mlp, lx_opt, eval=True
#             )
#             eval_loss.append(new_loss)
#             logger.image("classifier", np.transpose(eval_plot, (2, 0, 1)))
#             logger.write(step=i+40000)
#             best_pretrain_success_classifier = tools.save_checkpoint(
#                 ckpt_name, i, new_loss, best_pretrain_success_classifier, lx_mlp, logdir
#             )

#         else:
#             new_loss, _ = agent.train_lx(
#                 next(obs_train_dataset), lx_mlp, lx_opt
#             )
#             train_loss.append(new_loss)
#     log_plot("train_lx_loss", train_loss)
#     log_plot("eval_lx_loss", eval_loss)
#     logger.scalar("pretrain/train_lx_loss_min", np.min(train_loss))
#     logger.scalar("pretrain/eval_lx_loss_min", np.min(eval_loss))
#     logger.write(step=i)
#     print(eval_loss)
#     print('logged')
#     return lx_mlp, lx_opt
def main(config):
    mp.set_start_method('spawn', force=True) 
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # ==================== Logging ====================
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with open(f"{logdir}/config.yaml", "w") as f:
        yaml.dump(vars(config), f)
    # step in logger is environmental step
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    # step = count_steps(config.traindir)
    step = 0
    if config.from_ckpt and Path(config.from_ckpt).exists():
        print(f"Loading ckpt from {config.from_ckpt}")
        checkpoint = torch.load(config.from_ckpt)
        step = checkpoint['global_step']
    elif config.from_ckpt and not Path(config.from_ckpt).exists():
        print(f"folder {config.from_ckpt} not exists")
    else: 
        step = count_steps(config.traindir)
    if config.debug:
        logger = tools.DebugLogger(logdir, config.action_repeat * step)
    else:
        logger = tools.Logger(logdir, config.action_repeat * step,name = config.wandb_name)
    logger.config(vars(config))
    logger.write() 
    # success_val_dataset, failure_val_dataset, action_space, observation_space = create_val_dataset(config)
    success_eps = collections.OrderedDict()
    failure_eps = collections.OrderedDict()

    # print(success_eps)
    # tools.fill_expert_dataset_dubins(config, expert_eps)
    observation_space, action_space, _, _, _ = tools.fill_expert_dataset_real_data(config, success_eps,failure_eps,)
    # expert_dataset = make_dataset(expert_eps, config)
    #
    # success_dataset = make_dataset(success_eps, config)
   
   
    # tools.fill_expert_dataset_robocasa(config, failure_eps, dataset_type='failure')
    # failure_dataset = make_dataset(failure_eps, config)
    
    
    # validation replay buffer
    success_val_eps = collections.OrderedDict()
    failure_val_eps = collections.OrderedDict() 

    tools.fill_expert_dataset_real_data(config, success_val_eps, failure_val_eps,  is_val_set=True)
    success_val_dataset = make_dataset(success_val_eps, config)
    # tools.fill_expert_dataset_robocasa(config, failure_val_eps, is_val_set=True, dataset_type='failure')
    failure_val_dataset = make_dataset(failure_val_eps, config)
    print(f"Action Space: {action_space}.")# Low: {acts.low}. High: {acts.high}")
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    # ==================== Create Agent ====================
    agent = Dreamer(
        # train_envs[0].observation_space,
        # train_envs[0].action_space,
        observation_space,
        action_space,
        config,
        logger,
        None, #success_val_dataset,
        expert_dataset= None #failure_val_dataset if config.hybrid_training else None,
    ).to(config.device)
    # breakpoint()
    agent.requires_grad_(requires_grad=False)
    if config.from_ckpt and Path(config.from_ckpt).exists():
        print(f"Loading ckpt from {config.from_ckpt}")
        checkpoint = torch.load(config.from_ckpt)
        # step_so_far = checkpoint['total_step']
        if config.critic_ensemble_size > 1 or config.reward_ensemble_size > 1:
            # only false if loading with different ensemble size
            mk, uk = agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
            for k in mk:
                assert (
                    "Value" in k
                    or "_slow_value" in k
                    or "value" in k
                    or "Reward" in k
                    or "reward" in k
                )
            for k in uk:
                assert (
                    "Value" in k
                    or "_slow_value" in k
                    or "value" in k
                    or "Reward" in k
                    or "reward" in k
                )
        else:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        # try:
        #     tools.recursively_load_optim_state_dict(
        #         agent, checkpoint["optims_state_dict"]
        #     )
        # except Exception as e:
        #     # likely due to mismatch in pretrain optimizers
        #     print("Failed to load optim state dict", e)

        pretrained_ckpt_config = Path(config.from_ckpt).parent / "config.yaml"
        if pretrained_ckpt_config.exists():
            pretrained_config = yaml.load(pretrained_ckpt_config.read_text())
            if pretrained_config["num_exp_trajs"] == config.num_exp_trajs:
                warnings.warn(
                    f"Mismatch in number of expert trajectories in pretrained config {pretrained_config['num_exp_trajs']} and actual {config.num_exp_trajs}"
                )

        agent._should_pretrain._once = False
        if config.from_ckpt and config.pretrain_ema:
            print("Using EMA weights from pretraining")
            agent.ema.load_state_dict(checkpoint["ema"])
            agent.ema.copy_to(agent._task_behavior.actor.parameters())
    print('Evaluating Embeddings')
    agent.eval()
    if config.test_video_recon:
        ## do video prediction on the 
        print('Evaluating Video Reconstruction')
        eval_video_recon(agent, logger, success_val_dataset, failure_val_dataset, imagine=config.imagine)
    else: 
        eval_plot_embed(agent, success_val_dataset, failure_val_dataset)
    

    # return
def create_val_dataset(config):
    # validation replay buffer
    success_val_eps = collections.OrderedDict()
    observation_space, action_space, _, _, _  = tools.fill_expert_dataset_robocasa(config, success_val_eps, is_val_set=True, dataset_type='success')
    success_val_dataset = make_eval_dataset(success_val_eps, config)
    failure_val_eps = collections.OrderedDict() 
    tools.fill_expert_dataset_robocasa(config, failure_val_eps, is_val_set=True, dataset_type='failure')
    if config.plot_embed:
       
        failure_val_dataset = make_eval_failure_dataset(failure_val_eps, config)
    else:
        failure_val_dataset = make_eval_dataset(failure_val_eps, config)
        
    return success_val_dataset, failure_val_dataset, action_space, observation_space

def eval_video_recon(agent, logger, success_val_dataset, failure_val_dataset, imagine=False):
    # Evaluate the policy
    for i in range(27):
        if imagine:
            print('Generating videos with imagined latents')
            video_pred_success, success_loss = agent._wm.video_pred(next(success_val_dataset), )#batch_size = 32)
            video_pred_failure, failure_loss = agent._wm.video_pred(next(failure_val_dataset),) #batch_size = 32)
            #video_pred = agent._wm.video_pred(next(expert_dataset))
            
            #logger.video("eval_recon/openl_hand", to_np(video_pred2))
        else:
            video_pred_success, success_loss = agent._wm.video_recon(next(success_val_dataset))
            video_pred_failure, failure_loss = agent._wm.video_recon(next(failure_val_dataset))
        B, T, H, W, C  = video_pred_success.size()
        video_pred_success = video_pred_success.view(B*T, H, W, C).unsqueeze(0)
        B, T, H, W, C  = video_pred_failure.size()
        video_pred_failure = video_pred_failure.view(B*T, H, W, C).unsqueeze(0)
            #video_pred = agent._wm.video_pred(next(expert_dataset))
        logger.video(f"eval_recon_success/openl_agent_{i}", to_np(video_pred_success))
        logger.video(f"eval_recon_failure/openl_agent_{i}", to_np(video_pred_failure))
        for key in success_loss.keys():
            logger.scalar(f'eval_recon_success/{key}', float(torch.mean(success_loss[key]).cpu().numpy()))
            logger.scalar(f'eval_recon_failure/{key}', float(torch.mean(failure_loss[key]).cpu().numpy()))
    logger.write()
        # if other_dataset:
        #     for i in range(len(other_dataset)):
        #         video_pred, loss = agent._wm.video_pred(next(other_dataset[i]))
        #         if i == 0:
                
        #             logger.video("train_recon_success/openl_agent", to_np(video_pred))
        #             for key in loss:
        #                 logger.scalar(f'train_recon_success/{key}', float(torch.mean(loss[key]).cpu().numpy()))
        #         else:
        #             logger.video("train_recon_failure/openl_agent", to_np(video_pred))
        #             for key in loss:
        #                 logger.scalar(f'train_recon_failure/{key}', float(torch.mean(loss[key]).cpu().numpy()))
            #logger.video("train_recon/openl_hand", to_np(video_pred2))

    # Get stats
    '''eval_success, eval_score, eval_ep_len = ModelEvaluator(
        config=config,
        agent=eval_policy,
        env=eval_envs[0]._env,
        default_seed=config.seed,
        NUM_SEEDS=config.eval_num_seeds,
        NUM_EVALS_PER_SEED=config.eval_per_seed,
    ).evaluate_agent()'''

    # Update the metrics in the logger
    #logger.scalar(f"{eval_prefix}/eval_return", eval_score)
    #logger.scalar(f"{eval_prefix}/eval_length", eval_ep_len)
    #logger.scalar(f"{eval_prefix}/eval_success", eval_success)
    # logger.scalar(
        # f"{eval_prefix}/eval_episodes", config.eval_num_seeds * config.eval_per_seed
    # )
    logger.write(step=logger.step)
    return 

 

def eval_plot_embed(agent, success_val_dataset, failure_val_dataset):
    su_embeds = []
    fa_embeds = []
    index = 0
    exhaust_success = False
    exhaust_failure = False
    # for i in range(10):
    #     # success_data = next(success_val_dataset)
    #     # failure_data = next(failure_val_dataset)
    #     video_pred_success, success_loss = agent._wm.video_pred(next(success_val_dataset))
    #     video_pred_failure, failure_loss = agent._wm.video_pred(next(failure_val_dataset))
    #     #video_pred = agent._wm.video_pred(next(expert_dataset))
    #     logger.video(f"eval_recon_success/openl_agent_{i}", to_np(video_pred_success))
    #     logger.video(f"eval_recon_failure/openl_agent_{i}", to_np(video_pred_failure))
    # logger.write()
    # while True:
    #     try:
    #         success_data = next(success_val_dataset)
    #     except StopIteration:
    #         print('the success dataset has been exhausted! Index is ', index)
    #         exhaust_success = True
    #         success_data = None
    #     try:
    #         failure_data = next(failure_val_dataset)
    #     except StopIteration:
    #         print('the failure dataset has been exhausted! Index is ', index)
    #         exhaust_failure = True
    #         failure_data = None
    #     if exhaust_success and exhaust_failure:
    #         break
    #     # success_emb, failure_emb = agent.evaluate_embed({"success": success_data, "failure": failure_data})
    #     success_emb, failure_emb = agent.evaluate_imagined_embed({"success": success_data, "failure": failure_data})
    #     if success_emb is not None:
    #         su_embeds.append(success_emb.cpu().detach().numpy())
    #     if failure_emb is not None:
    #         fa_embeds.append(failure_emb.cpu().detach().numpy())
    #     index += 1
    #     print('index', index)
    #     # except StopIteration:
    #         # print('the dataset has been exhausted! Index is ', index)
    #         # break
    # plot_cluster_embedding(np.concatenate(su_embeds), np.concatenate(fa_embeds))
    return 


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def make_eval_dataset(episodes, config):
    sampler = tools.EpisodeSampler(episodes, length=config.batch_length)
    generator = sampler.sample_episodes()
    dataset = tools.from_eval_generator(generator, config.batch_size)
    return dataset
def make_eval_failure_dataset(episodes, config):
    sampler = tools.EpisodeSpecialSampler(episodes, length=config.batch_length)
    generator = sampler.sample_episodes()
    dataset = tools.from_eval_generator(generator, config.batch_size)
    return dataset

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))
def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    parser.add_argument('--config_path', type = str, default='config.yaml')
   
    config, remaining = parser.parse_known_args()
    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yaml = yaml.YAML(typ="safe", pure=True)
    # configs = yaml.load(
        # (pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
    # )
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs" / config.config_path).read_text()
    )


    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_video_recon', action='store_true')
    parser.add_argument('--plot_embed', action='store_true')
    parser.add_argument('--imagine', action='store_true')
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    final_config.time_limit = HORIZONS[final_config.task.split("_")[0]]
    

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Hybrid Training: {final_config.hybrid_training}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir}", "cyan", attrs=["bold"])
    main(final_config)
