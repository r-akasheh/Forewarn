import argparse

import h5py
import hydra
import numpy as np
import pickle as pkl
import os 
# from manimo.environments.single_arm_env import SingleArmEnv
import time
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument('--vis_imgs', action='store_true')
    parser.add_argument('--execute_action', action='store_true')
    args = parser.parse_args()
    demo_path = Path(args.file)
    vis_img = args.vis_imgs
    # files = os.listdir(demo_path)
    # files = [os.path.join(demo_path, f) for f in files if f.endswith('.pkl')]
    # files.sort()
    # for num_file, file in enumerate(files):
    #     with open(file, 'rb') as f:
    #         traj = pkl.load(f)[0]
    #     rs_img = traj[0][0]['cam_rs'][0]
    #     ## convert color for cv2
    #     rs_img_cv2 = cv2.cvtColor(rs_img, cv2.COLOR_RGB2BGR)
    #     zed_img = traj[0][0]['cam_zed_right'][0]
    #     zed_img_cv2 = cv2.cvtColor(zed_img, cv2.COLOR_RGB2BGR)
    #     cv2.imshow('rs',rs_img_cv2)
    #     cv2.imshow('zed',zed_img_cv2)
    #     cv2.waitKey(0)
    #     cv2.imwrite(f'camera_rs_{num_file}.png',rs_img_cv2)
    #     cv2.imwrite(f'camera_zed_{num_file}.png',zed_img_cv2)
    files = os.listdir(demo_path)
    files.sort()
    for file in files:
        if file.endswith('.pkl'):
            # demo_path = demo_path / file
            with open(demo_path/file, "rb") as f:
                traj = pkl.load(f)[0]
        else: 
            continue
          
        

        # get position action
        obs, action, reward = traj[1]
        start_state = obs["state"]

        ee_pos_desired = start_state[7:10]
        ee_quat_desired = start_state[10:13]


        # initialize agent
        # hydra.initialize(config_path="../conf", job_name="collect_demos_test")
        # actuators_cfg = hydra.compose(config_name="actuators_playback")
        # sensors_cfg = hydra.compose(config_name="sensors")
        # env_cfg = hydra.compose(config_name="env")
        # if args.execute_action:
        #     env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg=env_cfg)
        # hydra.core.global_hydra.GlobalHydra.instance().clear()

        # print(f"Setting home to demo start pos: {demo_start_pos}")
        # env.set_home(demo_start_pos)

        num_actions = len(traj)
        print("TRAJECTORY LOADED WITH {} ACTIONS".format(num_actions))
        print("Resetting environment")
        # if args.execute_action:
            # env.reset()
        print("Starting replay")
        action_count = 0
        new_traj_list = []
        print("Number of actions: ", num_actions)
        for i in range(0, num_actions-15):
            # print("\nAction: ", i)
            start_time = time.time()
            # breakpoint()
            # if i% 2 == 0 and (i< 40 or i> 52):
                # continue
            new_traj_list.append(traj[i])
            # if i % 2 == 0:
                # continue
            # total_action = [traj[i][1][:7], traj[i][1][7]]
            action_count += 1
            if vis_img and i in np.linspace(0, 64, 4, dtype=int):
                rs_img = traj[i][0]['cam_rs'][0]
                ## convert color for cv2
                rs_img_cv2 = cv2.cvtColor(rs_img, cv2.COLOR_RGB2BGR)
                zed_img = traj[i][0]['cam_zed_right'][0]
                zed_img_cv2 = cv2.cvtColor(zed_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'rs_{file}',rs_img_cv2)
                cv2.imshow(f'zed_{file}',zed_img_cv2)
                cv2.waitKey(0)
                # print(traj[i][1]['action'])
                # cv2.imwrite(f'camera_rs_{i}.png',rs_img_cv2)
                # cv2.imwrite(f'camera_zed_{i}.png',zed_img_cv2)
                # break
            # print("Total action: ", total_action)
            # if args.execute_action:
                # env.step(total_action)
            
            # env.step(total_action)
            # print("Step time: ", time.time() - start_time)
            # print('action_count',action_count)
            
            # if action_count > 34:
                # break
            # print("ROBOT STATUS: ", env.check_status())
        print("Replay finished")
        cv2.destroyAllWindows()
    # save new trajectory
    # the original one is real_data/exp_name/type/traj_xxxx.pkl and save the trajector to the new one is real_data/exp_name_cut/type/traj_xxxx.pkl
    # new_traj_path = demo_path.parent.parent.parent/(demo_path.parent.parent.stem +"_cut") / demo_path.parent.stem / demo_path.name
    # new_traj_path_short = demo_path.parent.parent.parent/(demo_path.parent.parent.stem +"_short") / demo_path.parent.stem / demo_path.name
    # os.makedirs(new_traj_path.parent, exist_ok=True)
    # os.makedirs(new_traj_path_short.parent, exist_ok=True)
    # with open(new_traj_path, 'wb') as f:   
        # pkl.dump([new_traj_list[:61]], f) 
    # # save every 2 of the new_traj_list to the new_traj_path_short
    # new_traj_list_short = new_traj_list[::2]
    # with open(new_traj_path_short, 'wb') as f:
        # pkl.dump([new_traj_list], f)
    # if args.execute_action:
        # env.close()

if __name__=="__main__":
    main()