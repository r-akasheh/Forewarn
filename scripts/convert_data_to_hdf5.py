import os
import pickle
import h5py
import numpy as np
import scipy 
from scipy.spatial.transform import Rotation as R
def quaternion_subtract(q1, q2):
    """
    Subtracts two quaternions by calculating the difference in orientation.
    Args:
        q1: Quaternion 1 as a numpy array (4,).
        q2: Quaternion 2 as a numpy array (4,).
    Returns:
        The quaternion difference as a numpy array (4,).
    """
    # Convert quaternions to Rotation objects
    r1 = R.from_quat([q1[0], q1[1], q1[2], q1[3]])
    r2 = R.from_quat([q2[0], q2[1], q2[2], q2[3]])
    # Calculate the relative rotation from r2 to r1
    r_diff = r1 * r2.inv()
    # Return the quaternion representation of the relative rotation
    return r_diff.as_quat()
def convert_gripper_to_eef_state(gripper_state):
    """
    Input: (eef_x, eef_y, eef_z, quat_x, quat_y, quat_z, quat_w, gripper)
    Used for clipping actions
    Recovers EEF Position by shifting the gripper position by 0.145 in the negative z direction
    Returns the computed EEF State
    """
    rotation= R.from_quat(gripper_state[3:7])
    pos_offset = rotation.apply(np.array([0, 0, 0.145]))
    eef_pos = gripper_state[:3] - pos_offset
    eef_rot = gripper_state[3:7]
    return np.concatenate([eef_pos, eef_rot, [gripper_state[7]]])

def convert_pickle_to_hdf5(input_dir, output_file):
    # Create or open HDF5 file for writing
    with h5py.File(output_file, 'a') as hdf5_file:
        # Create a group named 'data'
        if "data" not in hdf5_file.keys():
            data_group = hdf5_file.create_group('data')
            index = 1
        else :
            data_group = hdf5_file['data']
            index = len(data_group.keys()) + 1
        
        # Iterate over all pickle files in the input directory
        input_dir_folders = os.listdir(input_dir)
        input_dir_folders.sort()
        for folder in input_dir_folders:
            ## check if it is a folder
            ## if not continue
            print('folder', folder)
            if not os.path.isdir(os.path.join(input_dir, folder)):
                continue
            # if folder != 'optimal':
                # continue
            folder_path = os.path.join(input_dir, folder)
            if os.path.isdir(folder_path):
                mode_dict = {}
            mode_file_path = os.path.join(folder_path, 'mode.txt')
            if os.path.exists(mode_file_path):
                with open(mode_file_path, 'r') as mode_file:
                    for line in mode_file:
                        file_id, mode_id = line.strip().split(', ')
                        mode_dict[file_id] = mode_id
            # breakpoint()
            all_files = os.listdir(folder_path)
            all_files.sort()
            for i, filename in enumerate(all_files):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(folder_path, filename)
                else: 
                    continue
                
                # Load pickle file
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Create a group for each demonstration
                demo_group = data_group.create_group(f'demo_{index}')
                index += 1

                # Add attributes to the demo group
                file_id = str(int(filename.split('_')[1].split('.')[0]))
                demo_group.attrs['mode'] = mode_dict.get(file_id, 'unknown')
                mode = mode_dict.get(file_id, 'unknown')
                demo_group.attrs['label'] = 0 if mode == '0' else 1
                if 'grasp' in input_dir:
                    demo_group.attrs['type'] = 'demonstration'
                elif 'preference' in input_dir:
                    demo_group.attrs['type'] = 'finetune'
                elif 'baselines' in input_dir:
                    demo_group.attrs['type'] = 'baseline'
                elif 'paper_exps' in input_dir:
                    demo_group.attrs['type'] = 'headroom'
                else:
                    demo_group.attrs['type'] = 'unknown'
                demo_group.attrs['obs_id'] = folder

                # Prepare lists for storing actions and observations
                actions_abs_list = []
                actions_list = []
                states_list = []
                cam_wrist_view_list = []
                cam_front_view_list = []

                # Iterate through the trajectories in the pickle file
                # for trajectory in data[0]:
                    # for step in trajectory:
                print('file processed', index)
                for step in data[0][1:]:
                    # Extract observations and actions from the step
                    obs_data = step[0]
                    if isinstance(step[1], dict):
                        action_data = step[1]['action']
                    else:
                        action_data = step[1]

                    # Extracting state, cam_rs, cam_zed_right from observations
                    states_list.append(obs_data['state'])
                    cam_wrist_view_list.append(obs_data['cam_rs'][0])
                    cam_front_view_list.append(obs_data['cam_zed_right'][0])

                    # Extracting actions from action_data
                    actions_abs_list.append(action_data)
                    new_eef_state = convert_gripper_to_eef_state(obs_data['state'])
                    delta_action = np.zeros_like(action_data)
                    # First three elements: direct subtraction
                    delta_action[:3] = action_data[:3] - new_eef_state[:3]
                    # Next four elements: quaternion reduction
                    delta_action[3:7] = quaternion_subtract(action_data[3:7], new_eef_state[3:7])
                    # Last element: keep as in the original action
                    delta_action[7] = action_data[7]
                    actions_list.append(delta_action)
                print('file length', len(actions_list))
                
                # Convert collected data into numpy arrays for HDF5 storage
                actions_abs_np = np.array(actions_abs_list)
                actions_np = np.array(actions_list)
                states_np = np.array(states_list)
                cam_wrist_view_np = np.array(cam_wrist_view_list)
                cam_front_view_np = np.array(cam_front_view_list)
                
                # Store data in HDF5
                demo_group.create_dataset('actions_abs', data=actions_abs_np)
                demo_group.create_dataset('actions', data=actions_np)
                obs_group = demo_group.create_group('obs')
                obs_group.create_dataset('state', data=states_np)
                obs_group.create_dataset('cam_wrist_view_image', data=cam_wrist_view_np)
                obs_group.create_dataset('cam_front_view_image', data=cam_front_view_np)

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert pickle files to HDF5 format.')
    parser.add_argument('--input_dir', type=str, default= '/data/wm_data/',help='Path to the directory containing pickle files.')
    parser.add_argument('--output_file', type=str, default = '/data/wm_data/', help='Path to the output HDF5 file.')
    args = parser.parse_args()

    # Convert pickle files to HDF5 format
    convert_pickle_to_hdf5(args.input_dir, args.output_file)
    print(f'Conversion completed. HDF5 file saved as {args.output_file}')