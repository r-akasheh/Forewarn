import h5py
import numpy as np
import json
import os 
import tqdm

def get_min_max_actions(file_path):
    """
    Computes the min and max values across all 'actions', 'actions_abs', and 'state'
    stored in an HDF5 file under file['data']['demo_x'] and saves to a JSON file.

    Args:
        file_path (str): Path to the HDF5 file.
        output_file (str): Path to the JSON file where results will be saved.

    Returns:
        dict: Contains min and max values for 'actions', 'actions_abs', and 'state'.
    """
    with h5py.File(file_path, 'r') as file:
        if 'data' in file:
            data = file['data']
            demos = list(data.keys())
        else:
            data = file
            demos = [k for k in file.keys() if 'demo' in k]

        all_actions = []
        all_actions_abs = []
        all_states = []

        for demo in tqdm.tqdm(demos):
            demo_data = data[demo]
            
            actions = demo_data['actions'][:]
            actions_abs = demo_data['actions_abs'][:]
            state = demo_data['obs']['state'][:]

            all_actions.append(actions)
            all_actions_abs.append(actions_abs)
            all_states.append(state)

        # Concatenate all collected actions into a single array
        all_actions = np.vstack(all_actions)
        all_actions_abs = np.vstack(all_actions_abs)
        all_states = np.vstack(all_states)
        
        # Compute min and max for each dimension (axis=0)
        actions_min = np.min(all_actions, axis=0).tolist()
        actions_max = np.max(all_actions, axis=0).tolist()
        actions_abs_min = np.min(all_actions_abs, axis=0).tolist()
        actions_abs_max = np.max(all_actions_abs, axis=0).tolist()
        state_min = np.min(all_states, axis=0).tolist()
        state_max = np.max(all_states, axis=0).tolist()

        # Create the normalization dictionary
        norm_dict_delta = {
            'ob_min': state_min,
            'ob_max': state_max,
            'ac_min': actions_min,
            'ac_max': actions_max
        }
        norm_dict_abs = {
            'ob_min': state_min,
            'ob_max': state_max,
            'ac_min': actions_abs_min,
            'ac_max': actions_abs_max
        }
        
        output_file_abs = os.path.join(os.path.dirname(file_path), 'norm_dict_abs.json')
        # Save to a JSON file
        with open(output_file_abs, 'w') as json_file:
            json.dump(norm_dict_abs, json_file)
            
        output_file_delta = os.path.join(os.path.dirname(file_path), 'norm_dict_delta.json')
        with open(output_file_delta, 'w') as json_file:
            json.dump(norm_dict_delta, json_file)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the HDF5 file.")
    # parser.add_argument("--output_file", type=str, default='norm_dict.json', help="Path to the JSON file where results will be saved.")
    args = parser.parse_args()
    get_min_max_actions(args.file_path)