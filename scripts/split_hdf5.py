import h5py
import argparse
import numpy as np

def split_hdf5_file(args):
    # File paths for input and output files
    file = args.file_path[0]  # Single input file
    train_file = args.train_file  # Output file for the first 300 demos
    remaining_file = args.remaining_file  # Output file for the remaining demos
    
    # Open the original HDF5 file in read mode
    with h5py.File(file, 'r') as f:
        # Get the 'data' group
        data_group = f['data']
        
        # Sort the keys in the data group (assuming keys are demo_1, demo_2, ..., demo_n)
        keys = sorted(data_group.keys(), key=lambda x: int(x.split('_')[1]))
        
        # Determine the split point (demo_300)
        split_point = 300
        ## filter out all the demo key if f['data'][key]['actions_abs'] length is not 160
        keys = [key for key in keys if len(f['data'][key]['actions_abs']) == 160]
        ## get 10 keys for each label
        label_keys = {0: [], 1: [], 2: []}
        for key in keys:
            label = f['data'][key].attrs['label']
            label_keys[label].append(key)
        ## randomly select 10 key for each label
        selected_keys = {label: np.random.choice(keys, 10, replace=False) for label, keys in label_keys.items()}
        remaining_keys = selected_keys[0].tolist() + selected_keys[1].tolist() + selected_keys[2].tolist()
        ## the left keys not in remaining keys are in train_keys 
        train_keys = [key for key in keys if key not in remaining_keys]
        # train_keys = keys[:split_point]
        # remaining_keys = keys[split_point:]
        
        # Create the 'train.hdf5' file and copy the first 300 demos
        label_count = {0: 0, 1: 0, 2: 0}
        with h5py.File(train_file, 'w') as train_f:
            train_data_group = train_f.create_group('data')
            for attr_name in f['data'].attrs:
                train_data_group.attrs[attr_name] = f['data'].attrs[attr_name]
            
            for i, key in enumerate(train_keys):
                f.copy(f'data/{key}', train_data_group, name=f'demo_{i+1}')
                label_count[f['data'][key].attrs['label']] += 1
                print(f"Copied {key} to {train_file}")
        print(f"Label distribution in the training demos: {label_count}")
        print('total train keys', len(train_keys))
        
        # Create the 'remaining.hdf5' file and copy the remaining demos
        label_count = {0: 0, 1: 0, 2: 0}
        with h5py.File(remaining_file, 'w') as remaining_f:
            remaining_data_group = remaining_f.create_group('data')
            for attr_name in f['data'].attrs:
                remaining_data_group.attrs[attr_name] = f['data'].attrs[attr_name]
            for i, key in enumerate(remaining_keys):
                f.copy(f'data/{key}', remaining_data_group, name=f'demo_{i+1}')
                label_count[f['data'][key].attrs['label']] += 1
                
                print(f"Copied {key} to {remaining_file}")
        print(f"Label distribution in the remaining demos: {label_count}")
        print('total remaining keys', len(remaining_keys))
    
    print("Splitting complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, nargs=1, help="File path to the original HDF5 file")
    parser.add_argument("--train_file", type=str, help="Output file path for the first 300 demos")
    parser.add_argument("--remaining_file", type=str, help="Output file path for the remaining demos")
    
    args = parser.parse_args()
    split_hdf5_file(args)
