import argparse
import hydra
import time
from manimo.environments.single_arm_env import SingleArmEnv
import cv2
import json
import numpy as np
import os
from wm_pred_fork import WMPredictor, VLMInference
import imageio
import matplotlib.pyplot as plt
class ManimoLoop:
    def __init__(self, configs=None, callbacks=[], T=2000, mode = 'eval', wm_config= None, logdir=None, answer_type = 'snippet', steering_mode = 'vlm'):
        self.callbacks = callbacks
        self.wm_predictor = WMPredictor(wm_config)
        if steering_mode == 'vlm':
            peft_model = '/data/peft_models/run_02_21_custom_wm_150k_vlm_finetuning_0.2%_imagined_step63_1_history_16sample_size_fork_task_open-word-fork-all_18epoch_print_eval_metrics_3class_aug_failure_by2_shuffle_key_correct_prompt_hist_no_start_from_75/peft_checkpoint_18'

            model_name =  '/data/mllama/Llama-3.2-11B-Vision-Instruct/custom'
       
        self.vlm_inference = VLMInference(wm_configs=wm_config, model_name=model_name, 
                                      
                                         peft_model = peft_model,
                                         
                                         answer_type=answer_type)
        if not configs:
            configs = ["sensors", "actuators", "env"]
        self.logdir = logdir
        self.steering_mode = steering_mode
        self.T = T
        if mode == 'eval':
            self.agent_num = 1
        elif mode == 'finetune':
            self.agent_num = 2
        hydra.initialize(config_path="../conf", job_name="manimo_loop")

        env_configs = [
            hydra.compose(config_name=config_name) for config_name in configs
        ]

        self.env = SingleArmEnv(*env_configs)

    def process_pred(self, predictions):
        class_labels = []
        for text in predictions:
            # keywords = {
            #     2: ["fail", "unable", "struggle", "did not", "could not", "does not", "unsuccessful", "trouble", "not succeed", "not manage"],
            #     0: ["handle", "grasp handle", "by the handle"],
            #     1: ["inside", "interior", "inner", "rim", "by the rim", "within"],
            # }
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
        key_input = "n"
        while key_input == "n":
            for callback in self.callbacks:
                print("generating candidate plans in the middle of the plan!")
 
                trajs2, pred_trajs2, aggregated_trajs2, mode_probs_2, labels2, current_pose = callback.get_candidate_plans_w_current_pose(obs, agent_choice = 2, n_clusters = 6)

                trajs_candidates = aggregated_trajs2 # traj2
                pred_trajs_candidates = pred_trajs2

                for i in range(1): #
                    if self.agent_num == 2:
                        # for i in range(7):
                        callback.__init_agent__(self.agent_names[i])
                        trajs, pred_trajs, aggregated_trajs, mode_probs, labels = callback.get_candidate_plans(obs, agent_choice = 1)
                    else:
                        trajs, pred_trajs, aggregated_trajs, mode_probs = None, None, None, None
        
                    if self.agent_num ==2:
                        fig = callback.visualize_two_plans(trajs, trajs2, aggregated_trajs, aggregated_trajs2, mode_probs, mode_probs_2)
                    else: 
                        fig = callback.visualize_plans_w_agg(trajs2, aggregated_trajs2, mode_probs_2, labels2, current_pose=current_pose)

                    os.makedirs(os.path.join(callback.logger.storage_path, callback.logger.experiment_name), exist_ok=True)
                    ## check if plan_middle.html is created
                    if os.path.exists(os.path.join(callback.logger.storage_path, callback.logger.experiment_name, f'plan_middle.html')):
                        fig.write_html(os.path.join(callback.logger.storage_path, callback.logger.experiment_name, f'plan_middle_2.html'))
                    else:
                        fig.write_html(os.path.join(callback.logger.storage_path, callback.logger.experiment_name, f'plan_middle.html'))
                break

            break  
   
        return trajs_candidates, pred_trajs_candidates
    def plot_plans(self, trajs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj_idx in range(len(trajs)):
            real_trajectory = trajs[traj_idx]
            ## get color for each trajectory
            colormap = plt.cm.get_cmap('tab10')
            color = colormap(traj_idx/len(trajs))
            for j in range(len(real_trajectory)):
                pos= real_trajectory[j][:3]
                ax.scatter(pos[0], pos[1], pos[2], c=color, marker='o', label = f'traj_{traj_idx}' if f'traj_{traj_idx}' \
            not in plt.gca().get_legend_handles_labels()[1] else '')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.view_init(elev =0 , azim=0)
        plt.legend()
        plt.show()
        cv2.waitKey(0)
        plt.close()
    def run(self):
        traj_idx = 0
        trajs_candidates = []
        pred_trajs_candidates = []
        obs, _ = self.env.reset()

        try:
            while True:
                
                key = input(f'--> Press Enter to start new trajectory {traj_idx}')
                if key == 'q':
                    break
                
                start_index = 0
                pred_length = 64
                ee_pos = self.env.actuators[0].robot.get_ee_pose()
                for callback in self.callbacks:
                    callback.pred_obs = obs.copy()

                    callback.on_begin_traj(traj_idx)
                start_time = time.time()
                steps = 0
                break_loop = False
                success = True
                question_key = 'grasping'
                for step_idx in range(self.T):
                    steps += 1
                    action = None
                    # if step_idx == stargrasp-handlet_index:
                    for callback in self.callbacks:
                        if step_idx == start_index:
                            actions_candidates, pred_actions_candidates = self.generate_plans(obs, step_idx)
                          
                            infer_time = time.time()
      
                            if self.steering_mode == 'vlm':
                                ## get the vlm verification results
                                self.pred, self.text_input, self.pred_2, self.pred_frames = self.vlm_inference.infer_two_stage(obs, actions_candidates, normalize=True, question_key = question_key)
                                labels = self.process_pred(self.pred)
                                print('vlm prediction', self.pred, labels)
                                print('vlm prediction 2', self.pred_2)
                                print('vlm inference time', time.time() - infer_time)
                            
                            if self.pred_frames is not None:
                                for choice in range(len(self.pred_frames['pred_cam_rs'])):
                                    label = labels[choice]
                                    
                                    imageio.mimsave(os.path.join(self.logdir, f'pred_{start_index}_{choice}_{label}.gif'), self.pred_frames['pred_cam_rs'][choice]) 
                            # ## if there is 'success in one of the pred, use that actions
                            question_key = 'placing'
                            success = False
                            selected_ind = None
                            for choice in range(len(self.pred)):
                                if self.steering_mode == 'vlm':
                                    if str(choice+1) in self.pred_2[0]:
                                        selected_action = actions_candidates[choice]
                                        callback.set_traj_in_the_middle(selected_action, pred_actions_candidates[choice], step_idx)
                                        success = True
                                        selected_ind = choice
                                        print('Selected trajectory number', choice)
                                        break
                                    
                            if success == False:
                                for choice in range(len(self.pred)):
                                    if labels[choice] == 1:
                                        selected_action = actions_candidates[choice]
                                        callback.set_traj_in_the_middle(selected_action, pred_actions_candidates[choice], step_idx)
                                        success = True
                                        selected_ind = choice
                                        print('automatically select trajectory number via gt label', choice)
                                        break
                                    
                            
                            start_index = 50 
                            
                            ## if there is no success in the labels, return the failure
                            if not success:
                                print('no success in the pred, stop running out the policy')
                                break
                           
                        new_action = callback.get_action(obs, pred_action=ee_pos)
                        


                        
                        print('new_action', new_action)
                        if new_action is not None:
                            action = new_action
                    if not success:
                        break_loop = True
                        traj_status = 'failure'
                        break
                    if action is None:
                        obs = self.env.get_obs()
                        ee_pos = self.env.actuators[0].robot.get_ee_pose()
                        time.sleep(1/self.env.hz)
                        continue
                    
                    new_obs, _, _, _ = self.env.step(action)
                    # Logging
                    if action is not None:
                        # Create action_dict with keys "action", and "joint_action", "ee_pos_action", "ee_quat_action" "eef_gripper_action" from new_obs
                        action_dict = {}
                        action_dict["action"] = action
                        action_dict["delta"] = new_obs["delta"]
                        action_dict["joint_action"] = new_obs["joint_action"]
                        action_dict["ee_pos_action"] = new_obs["ee_pos_action"]
                        action_dict["ee_quat_action"] = new_obs["ee_quat_action"]
                        action_dict["eef_gripper_action"] = new_obs["eef_gripper_action"]
                        # log obs and action
                        for callback in self.callbacks:
                            ## append the pred frames keys to obs keys
                            if step_idx < pred_length+start_index and step_idx >= start_index:
                                if self.pred_frames is not None:
                                    for key in self.pred_frames.keys():
                                        for choice in range(6):
                                            obs[f'{key}_{choice}'] = self.pred_frames[key][choice][step_idx-start_index]
                                            assert obs[f'{key}_{choice}'].shape == (64, 64, 3), obs[f'{key}_{choice}'].shape
                                        obs[key] = self.pred_frames[key][selected_ind][step_idx-start_index]
                                        assert obs[key].shape == (64, 64, 3), obs[key].shape
                                if self.steering_mode == 'vlm':
                                    obs['labels'] = self.pred + self.pred_2
                                    obs['text_input'] = self.text_input
                                if self.steering_mode == 'classifier':
                                    obs['labels'] = self.pred


                              
                                obs['selected_ind'] = selected_ind
                            if step_idx >= start_index and step_idx < 80 +start_index:
                                obs['pred_actions'] = np.array(actions_candidates)[:, step_idx-start_index]
                                    # assert obs[key].shape == obs['cam_rs'][0].shape, (obs[key].shape, obs['cam_rs'][0].shape)   
                            callback.log_obs(obs, action_dict)
                    
                    # Update obs
                    obs = new_obs

                    finish = False
                    if steps > 140:
                        finish = True
                        traj_status = 'success'
                        break
                    for callback in self.callbacks:
                        traj_status = callback.on_step(traj_idx, step_idx)
                        if traj_status == None:
                            continue
                        else:
                            finish = True
                            print('finishing')
                            
                        if finish:
                            break

                    if finish:
                        traj_status = 'success'
                        break
                        

                print(f"fps: {steps / (time.time() - start_time)}")
                
                for callback in self.callbacks:
                    break_loop = callback.on_end_traj(traj_idx, traj_status=traj_status)
                
                if break_loop:
                    print("Breaking loop")
                    break

                traj_idx += 1

                if traj_idx > 0:
                    print("Exiting Trajectory Loop")
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("Exiting Manimo Loop")

        self.env.close()

def main():
    manimo_loop = ManimoLoop()


if __name__ == "__main__":
    main()
