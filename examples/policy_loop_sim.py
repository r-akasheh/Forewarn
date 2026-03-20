import argparse
import time
import cv2
import json
import numpy as np
import os
import torch
import imageio
import matplotlib.pyplot as plt

# Robomimic imports for simulation
import robomimic.utils.env_utils as EnvUtils
import robomimic.envs.env_base as EB

# Local imports
from wm_pred_fork import WMPredictor, VLMInference

class DummyPolicyCallback:
    """
    A template for the policy callback. You need to implement the actual logic.
    """
    def __init__(self):
        self.logger = type('Logger', (object,), {'storage_path': './logs', 'experiment_name': 'sim_test'})()
        self.pred_obs = None

    def on_begin_traj(self, traj_idx):
        print(f"Callback: Beginning trajectory {traj_idx}")

    def get_candidate_plans_w_current_pose(self, obs, agent_choice=2, n_clusters=6):
        # TODO: Implement your policy to generate candidate plans here.
        # Returning dummy data for demonstration
        print("Callback: Generating dummy candidate plans")
        
        # specific to your policy's output format
        # trajs: [N_candidates, Horizon, Action_Dim]
        trajs = np.zeros((6, 64, 7)) 
        pred_trajs = np.zeros((6, 64, 7))
        aggregated_trajs = np.zeros((6, 64, 7))
        mode_probs = np.ones(6) / 6
        labels = np.zeros(6)
        current_pose = np.zeros(7)
        
        return trajs, pred_trajs, aggregated_trajs, mode_probs, labels, current_pose

    def visualize_plans_w_agg(self, trajs, aggregated_trajs, mode_probs, labels, current_pose):
        # TODO: Implement visualization if needed
        return plt.figure()

    def set_traj_in_the_middle(self, selected_action, pred_action, step_idx):
        print(f"Callback: Selected trajectory for step {step_idx}")

    def get_action(self, obs, pred_action=None):
        # TODO: Return the next action from your selected policy/trajectory
        # Returning a dummy action
        return np.zeros(7) # Adjust dimension to your env

    def log_obs(self, obs, action_dict):
        pass

    def on_step(self, traj_idx, step_idx):
        return None # Return 'success' or 'failure' if detected

    def on_end_traj(self, traj_idx, traj_status):
        print(f"Callback: End trajectory {traj_idx}, status: {traj_status}")
        return False

class PolicyLoopSim:
    def __init__(self, wm_config, env_meta=None, callbacks=[], T=500, mode='eval', logdir='./logs', answer_type='snippet', steering_mode='vlm'):
        self.callbacks = callbacks
        self.wm_config = wm_config
        
        # Initialize WM Predictor (if needed separately)
        # self.wm_predictor = WMPredictor(wm_config)

        # TODO: Update these paths to your local setup
        peft_model = '/data/peft_models/your_peft_model_path' 
        model_name = '/data/mllama/Llama-3.2-11B-Vision-Instruct/custom' 
        
        if steering_mode == 'vlm':
            # Ensure paths are valid before initializing
            if not os.path.exists(model_name):
                print(f"Warning: Model path {model_name} does not exist. Please update in policy_loop_sim.py")

        self.vlm_inference = VLMInference(
            wm_configs=wm_config, 
            model_name=model_name, 
            peft_model=peft_model,
            answer_type=answer_type
        )

        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
            
        self.steering_mode = steering_mode
        self.T = T
        self.agent_num = 1 if mode == 'eval' else 2
        
        # Initialize Simulation Environment
        if env_meta is None:
            # Default to a simple environment if not provided
            print("No env_meta provided. Using default Lift environment.")
            self.env_meta = {
                "env_name": "Lift",
                "env_kwargs": {
                    "has_renderer": False, # Set to True to see simulation
                    "has_offscreen_renderer": True,
                    "use_camera_obs": True,
                    "camera_names": ["agentview", "robot0_eye_in_hand"],
                    "camera_height": 64,
                    "camera_width": 64,
                }
            }
        else:
            self.env_meta = env_meta

        self.env = EnvUtils.create_env_from_metadata(
            env_meta=self.env_meta,
            render=self.env_meta['env_kwargs'].get('has_renderer', False),
            render_offscreen=self.env_meta['env_kwargs'].get('has_offscreen_renderer', True),
            use_image_obs=self.env_meta['env_kwargs'].get('use_camera_obs', True),
        )

    def process_pred(self, predictions):
        class_labels = []
        for text in predictions:
            keywords = {
                2: ["fail", "unable", "struggle", "did not", "could not", "does not", "cannot", "incomplete", "unsuccessful", "trouble", "not succeed", "not manage", "ineffective"],
                0: ["corner", "edge", "side", "outer", "border", "in"],
                1: ["center", "midsection", "middle", "central", "core", "midpoint"],

            }
            class_labels.append(3)
            for label, terms in keywords.items():
                if any(term in text.lower() for term in terms):
                    class_labels[-1] = label
                    break
        return class_labels

    def generate_plans(self, obs, step_idx):
        trajs_candidates = []
        pred_trajs_candidates = []
        
        # In sim, we might loop differently, but keeping logic close to original
        for callback in self.callbacks:
            print("generating candidate plans...")
            
            # Using agent_choice=2 as in original
            trajs2, pred_trajs2, aggregated_trajs2, mode_probs_2, labels2, current_pose = callback.get_candidate_plans_w_current_pose(obs, agent_choice=2, n_clusters=6)

            trajs_candidates = aggregated_trajs2 
            pred_trajs_candidates = pred_trajs2

            # Visualization logic
            fig = callback.visualize_plans_w_agg(trajs2, aggregated_trajs2, mode_probs_2, labels2, current_pose=current_pose)
            
            save_path = os.path.join(self.logdir, 'plans')
            os.makedirs(save_path, exist_ok=True)
            # fig.write_html(os.path.join(save_path, f'plan_step_{step_idx}.html'))
            break 
            
        return trajs_candidates, pred_trajs_candidates

    def run(self):
        traj_idx = 0
        
        # Robomimic reset returns dictionary obs directly
        obs = self.env.reset()

        try:
            while True:
                key = input(f'--> Press Enter to start new trajectory {traj_idx} (q to quit): ')
                if key == 'q':
                    break
                
                # Reset environment for new trajectory
                if traj_idx > 0:
                    obs = self.env.reset()

                start_index = 0
                pred_length = 64
                
                # Get EE pose from observation dict
                # Adjust key based on your environment (e.g., 'robot0_eef_pos')
                ee_pos_key = 'robot0_eef_pos' 
                if ee_pos_key in obs:
                    ee_pos = obs[ee_pos_key]
                else:
                    print(f"Warning: {ee_pos_key} not found in obs. Using zeros.")
                    ee_pos = np.zeros(3)

                for callback in self.callbacks:
                    callback.pred_obs = obs.copy()
                    callback.on_begin_traj(traj_idx)
                
                start_time = time.time()
                steps = 0
                break_loop = False
                success = True
                traj_status = 'unknown'
                question_key = 'grasping'

                for step_idx in range(self.T):
                    steps += 1
                    action = None
                    
                    for callback in self.callbacks:
                        if step_idx == start_index:
                            actions_candidates, pred_actions_candidates = self.generate_plans(obs, step_idx)
                          
                            infer_time = time.time()
      
                            if self.steering_mode == 'vlm':
                                ## get the vlm verification results
                                # Note: You might need to adapt obs keys for VLM inference if it expects specific camera names
                                self.pred, self.text_input, self.pred_2, self.pred_frames = self.vlm_inference.infer_two_stage(obs, actions_candidates, normalize=True, question_key=question_key)
                                labels = self.process_pred(self.pred)
                                print('vlm prediction', self.pred, labels)
                                print('vlm prediction 2', self.pred_2)
                                print('vlm inference time', time.time() - infer_time)
                            
                            # Save prediction gifs
                            if self.pred_frames is not None:
                                for choice in range(len(self.pred_frames.get('pred_cam_rs', []))):
                                    label = labels[choice] if choice < len(labels) else 'unknown'
                                    imageio.mimsave(os.path.join(self.logdir, f'pred_{start_index}_{choice}_{label}.gif'), self.pred_frames['pred_cam_rs'][choice]) 
                            
                            question_key = 'placing'
                            success = False
                            selected_ind = None
                            
                            # Selection logic
                            for choice in range(len(self.pred)):
                                if self.steering_mode == 'vlm':
                                    # Simple heuristic: if choice number is in text
                                    if str(choice+1) in self.pred_2[0]:
                                        selected_action = actions_candidates[choice]
                                        callback.set_traj_in_the_middle(selected_action, pred_actions_candidates[choice], step_idx)
                                        success = True
                                        selected_ind = choice
                                        print('Selected trajectory number', choice)
                                        break
                                    
                            if not success:
                                for choice in range(len(self.pred)):
                                    if labels[choice] == 1: # Center/Good label
                                        selected_action = actions_candidates[choice]
                                        callback.set_traj_in_the_middle(selected_action, pred_actions_candidates[choice], step_idx)
                                        success = True
                                        selected_ind = choice
                                        print('Automatically select trajectory via gt label', choice)
                                        break
                                    
                            start_index = 50 # Next planning step
                            
                            if not success:
                                print('No success in pred, stop running policy')
                                break
                           
                        new_action = callback.get_action(obs, pred_action=ee_pos)
                        
                        if new_action is not None:
                            action = new_action

                    if not success:
                        break_loop = True
                        traj_status = 'failure'
                        break

                    if action is None:
                        # No action from policy, just step environment (maybe zero action?)
                        # In simulation, we typically need an action.
                        print("Warning: No action generated.")
                        obs, _, _, _ = self.env.step(np.zeros_like(self.env.action_space.sample())) 
                        continue
                    
                    # Step Environment
                    new_obs, reward, done, info = self.env.step(action)
                    
                    # Logging
                    # Create action_dict for callbacks
                    action_dict = {}
                    action_dict["action"] = action
                    # Robomimic might not return these keys in new_obs, check your env wrapper
                    if "delta" in new_obs: action_dict["delta"] = new_obs["delta"]
                    if "joint_action" in new_obs: action_dict["joint_action"] = new_obs["joint_action"]
                    
                    for callback in self.callbacks:
                        # Log logic similar to original
                        if step_idx < pred_length + start_index and step_idx >= start_index and self.steering_mode == 'vlm':
                             obs['labels'] = self.pred + self.pred_2
                             obs['text_input'] = self.text_input
                             if selected_ind is not None:
                                obs['selected_ind'] = selected_ind
                        
                        callback.log_obs(obs, action_dict)
                    
                    obs = new_obs

                    finish = done
                    if steps > self.T:
                        finish = True
                        traj_status = 'timeout'
                    
                    # Check custom success condition
                    if 'success' in info and info['success']:
                         finish = True
                         traj_status = 'success'

                    for callback in self.callbacks:
                        status = callback.on_step(traj_idx, step_idx)
                        if status is not None:
                            finish = True
                            traj_status = status
                    
                    if finish:
                        break
                
                print(f"Trajectory {traj_idx} finished. Status: {traj_status}. FPS: {steps / (time.time() - start_time)}")
                
                for callback in self.callbacks:
                    break_loop = callback.on_end_traj(traj_idx, traj_status=traj_status)
                
                if break_loop:
                    print("Breaking loop requested by callback")
                    break

                traj_idx += 1
                if traj_idx > 0: # Stop after 1 traj for testing
                    print("Exiting Loop")
                    break
                    
        except KeyboardInterrupt:
            print("Exiting Policy Loop Sim")
        finally:
            self.env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/wm_cup_config.yaml', help='Path to WM config')
    args = parser.parse_args()

    # Load WM Config
    import yaml
    with open(args.config, 'r') as f:
        wm_config = yaml.safe_load(f)
        
    # Mocking hydra 'defaults' which might be expected by WMPredictor
    if 'defaults' not in wm_config:
        wm_config['defaults'] = wm_config

    # TODO: Initialize your specific Policy Callback
    # my_policy = MyDiffusionPolicy(config_path=...)
    my_policy = DummyPolicyCallback()

    loop = PolicyLoopSim(
        wm_config=wm_config,
        callbacks=[my_policy],
        env_meta=None # Will use default Lift env
    )
    loop.run()

if __name__ == "__main__":
    main()
