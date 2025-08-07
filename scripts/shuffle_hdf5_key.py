import h5py
import random

# Open the HDF5 file
file_path = '/home/jzyuan/uncertainty_aware_steering/robomimic/datasets/square/combined_wm/70k_rollouts_and_demos_extra_fails.hdf5'
with h5py.File(file_path, 'r+') as file:
    data = file['data']
    
    # Extract keys and labels
    keys = list(data.keys())
    labels = {key: data[key].attrs['label'] for key in keys}
    
    # Separate keys by label
    label_0_keys = [key for key in keys if labels[key] == 0]
    label_1_keys = [key for key in keys if labels[key] == 1]
    label_2_keys = [key for key in keys if labels[key] == 2]
    
    # Shuffle for randomness
    random.shuffle(label_0_keys)
    random.shuffle(label_1_keys)
    random.shuffle(label_2_keys)
    
    # Select the last 36 keys with the desired label distribution
    last_36_keys = (
        label_0_keys[:48] + 
        label_1_keys[:24] + 
        label_2_keys[:24]
    )
    
    # The rest of the keys
    remaining_keys = [
        key for key in keys if key not in last_36_keys
    ]
    
    # Merge remaining keys and last 36 keys
    new_order = remaining_keys + last_36_keys
    
    # Create a new HDF5 file to avoid overwriting issues
    with h5py.File('/home/jzyuan/uncertainty_aware_steering/robomimic/datasets/square/combined_wm/shuffled_extra_fails.hdf5', 'w') as new_file:
        new_data = new_file.create_group('data')
        ## the key in new_data starts from demo_1 to demo_336
        for i,key in enumerate(new_order):
            # Copy datasets and attributes
            data.copy(key, new_data, name=f'demo_{i+1}')
            new_data[f'demo_{i+1}'].attrs['label'] = data[key].attrs['label']

print("Reordering complete! Saved as 'reordered_file.hdf5'")
