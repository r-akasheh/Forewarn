import h5py
import random

# Open the HDF5 file
file_path = '/data/wm_data/GraspBag_data/bag_demo.hdf5'
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
        label_0_keys[:12] + 
        label_1_keys[:12] + 
        label_2_keys[:12]
    )
    
    # The rest of the keys
    remaining_keys = [
        key for key in keys if key not in last_36_keys
    ]
    
    # Merge remaining keys and last 36 keys
    new_order = remaining_keys + last_36_keys
    
    # Create a new HDF5 file to avoid overwriting issues
    with h5py.File('/data/wm_data/GraspBag_data/GraspBag_demo.hdf5', 'w') as new_file:
        new_data = new_file.create_group('data')
        ## the key in new_data starts from demo_1 to demo_336
        for i,key in enumerate(new_order):
            # Copy datasets and attributes
            data.copy(key, new_data, name=f'demo_{i+1}')
            new_data[f'demo_{i+1}'].attrs['label'] = data[key].attrs['label']

print("Reordering complete! Saved as 'reordered_file.hdf5'")
