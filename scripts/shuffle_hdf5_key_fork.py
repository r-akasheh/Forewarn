import h5py
import random

# Open the HDF5 file
file_path = '/data/wm_data/GraspFork_data/GraspFork_demo_copy.hdf5'
with h5py.File(file_path, 'r+') as file:
    data = file['data']
    
    # Extract keys and labels
    keys = list(data.keys())
    labels = {key: data[key].attrs['label'] for key in keys}
    
    # Separate keys by label
    label_0_keys = [key for key in keys if labels[key] == 0]
    label_1_keys = [key for key in keys if labels[key] == 1]
    label_2_keys = [key for key in keys if labels[key] == 2]
    label_3_keys = [key for key in keys if labels[key] == 3]
    label_4_keys = [key for key in keys if labels[key] == 4]
    label_5_keys = [key for key in keys if labels[key] == 5]
    label_6_keys = [key for key in keys if labels[key] == 6]
    label_7_keys = [key for key in keys if labels[key] == 7]
    
    # Shuffle for randomness
    random.shuffle(label_0_keys)
    random.shuffle(label_1_keys)
    random.shuffle(label_2_keys)
    random.shuffle(label_3_keys)
    random.shuffle(label_4_keys)
    random.shuffle(label_5_keys)
    random.shuffle(label_6_keys)
    random.shuffle(label_7_keys)
    
    # Select the last 36 keys with the desired label distribution
    last_36_keys = (
        label_0_keys[:5] + 
        label_1_keys[:5] + 
        label_2_keys[:5] + 
        label_3_keys[:5] + 
        label_4_keys[:4] + 
        label_5_keys[:4] + 
        label_6_keys[:4] + 
        label_7_keys[:4]
    )
    
    # The rest of the keys
    remaining_keys = [
        key for key in keys if key not in last_36_keys
    ]
    
    # Merge remaining keys and last 36 keys
    new_order = remaining_keys + last_36_keys
    
    # Create a new HDF5 file to avoid overwriting issues
    with h5py.File('/data/wm_data/GraspFork_data/train.hdf5', 'w') as new_file:
        new_data = new_file.create_group('data')
        ## the key in new_data starts from demo_1 to demo_336
        for i,key in enumerate(remaining_keys):
            # Copy datasets and attributes
            data.copy(key, new_data, name=f'demo_{i+1}')
            new_data[f'demo_{i+1}'].attrs['label'] = data[key].attrs['label']
    with h5py.File('/data/wm_data/GraspFork_data/test.hdf5', 'w') as new_file:
        new_data = new_file.create_group('data')
        ## the key in new_data starts from demo_1 to demo_336
        for i,key in enumerate(last_36_keys):
            # Copy datasets and attributes
            data.copy(key, new_data, name=f'demo_{i+1}')
            new_data[f'demo_{i+1}'].attrs['label'] = data[key].attrs['label']

print("Reordering complete! Saved as 'train.hdf5' and 'test.hdf5'")
