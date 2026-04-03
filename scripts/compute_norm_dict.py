import h5py
import numpy as np
import json
import os 
import tqdm

def get_min_max_actions(file_path):
    """
    Computes the min and max values across all 'obs' (state) and 'actions' (or 'actions_abs')
    stored in an HDF5 file under file['data']['demo_x'] and saves to a JSON file.

    Supports both old format (nested obs/state) and ManiSkill format (flat obs array).

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: Contains min and max values for 'actions'/'actions_abs' and 'obs'.
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
            
            # Resolve action key: prefer 'actions', fallback to 'actions_abs'
            if 'actions' in demo_data:
                actions = demo_data['actions'][:]
                all_actions.append(actions)
            elif 'actions_abs' in demo_data:
                actions_abs = demo_data['actions_abs'][:]
                all_actions_abs.append(actions_abs)

            # Resolve state/obs: support both flat obs and nested obs['state']
            if 'obs' in demo_data:
                obs = demo_data['obs']
                if isinstance(obs, h5py.Dataset):
                    # Flat obs array (ManiSkill format)
                    state = obs[:]
                elif isinstance(obs, h5py.Group) and 'state' in obs:
                    # Nested obs/state format
                    state = obs['state'][:]
                else:
                    continue
            else:
                continue

            all_states.append(state)

        # Concatenate all collected data into single arrays
        if all_actions:
            all_actions = np.vstack(all_actions)
            actions_min = np.min(all_actions, axis=0).tolist()
            actions_max = np.max(all_actions, axis=0).tolist()
        else:
            actions_min = None
            actions_max = None

        if all_actions_abs:
            all_actions_abs = np.vstack(all_actions_abs)
            actions_abs_min = np.min(all_actions_abs, axis=0).tolist()
            actions_abs_max = np.max(all_actions_abs, axis=0).tolist()
        else:
            actions_abs_min = None
            actions_abs_max = None

        all_states = np.vstack(all_states)
        state_min = np.min(all_states, axis=0).tolist()
        state_max = np.max(all_states, axis=0).tolist()

        # Create the normalization dictionary for delta actions
        if actions_min is not None:
            norm_dict_delta = {
                'ob_min': state_min,
                'ob_max': state_max,
                'ac_min': actions_min,
                'ac_max': actions_max
            }
            output_file_delta = os.path.join(os.path.dirname(file_path), 'norm_dict_delta.json')
            with open(output_file_delta, 'w') as json_file:
                json.dump(norm_dict_delta, json_file)
            print(f"Saved norm_dict_delta.json to {output_file_delta}")

        # Create the normalization dictionary for absolute actions
        if actions_abs_min is not None:
            norm_dict_abs = {
                'ob_min': state_min,
                'ob_max': state_max,
                'ac_min': actions_abs_min,
                'ac_max': actions_abs_max
            }
            output_file_abs = os.path.join(os.path.dirname(file_path), 'norm_dict_abs.json')
            with open(output_file_abs, 'w') as json_file:
                json.dump(norm_dict_abs, json_file)
            print(f"Saved norm_dict_abs.json to {output_file_abs}")

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the HDF5 file.")
    # parser.add_argument("--output_file", type=str, default='norm_dict.json', help="Path to the JSON file where results will be saved.")
    args = parser.parse_args()
    get_min_max_actions(args.file_path)