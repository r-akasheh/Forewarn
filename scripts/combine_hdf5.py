import h5py
import argparse

import numpy as np
def combine_hdf5_files(args):
    # breakpoint()
    files = args.file_path
    file1 = files[0]
    file2 = files[1]    
    output_file = args.output_file
    # # exclude_idxs = [2,5, 7,9, 11, 18, 25, 28,30,32,34,36, 37, 40,52,55,61, 62, 63, 65, 66, 72, 80, 85, 87]
    # exclude_idxs = [7,15,18,21,23,26]
    # exclude_demos = [f'demo_{idx}' for idx in exclude_idxs]
    # Open the first HDF5 file in read mode and create an output file
    
    with h5py.File(file1, 'r') as f1:
        with h5py.File(output_file, 'a') as f_out:
            data_group = f_out.create_group('data')
            data_group = f_out['data']
            for attr_name in f1['data'].attrs:
                data_group.attrs[attr_name] = f1['data'].attrs[attr_name]
            # breakpoint()
            # # Copy datasets from the first file to the output file, excluding specified demos
            # inds = np.argsort([int(elem[5:]) for elem in f1['data'].keys()])
            # keys = [list(f1['data'].keys())[i] for i in inds]
            keys = list(f1['data'].keys())
            old_num = 0
            for i, key in enumerate(keys):
                # if key in exclude_demos:            
                # if i not in [2, 14, 17, 40, 44]:
                #     print(f"Excluding {key} from {file1}")
                #     continue
                # if i != 2:
                #    assert f1['data'][key].attrs['label'] == 2
                old_num += 1
                # if key in ['demo_35', 'demo_36']:
                    # continue
                f1.copy(f'data/{key}', data_group, name = f'demo_{old_num}')
                print(f"Copied {key} from {file1}")
    # breakpoint()
    # # Open the output file in append mode and the second HDF5 file in read mode
    for file2 in files[1:]:
        with h5py.File(output_file, 'a') as f_out:
            with h5py.File(file2, 'r') as f2:
                data_group = f_out['data']

                # Find the maximum existing demo number in the output file
                existing_keys = [key for key in data_group.keys() if key.startswith('demo_')]
                if existing_keys:
                    max_num = max(int(key.split('_')[1]) for key in existing_keys)
                else:
                    max_num = 0

                # Iterate over the keys in the second file and copy them to the output file
                # inds = np.argsort([int(elem[5:]) for elem in f2['data'].keys()])
                # keys = [list(f2['data'].keys())[i] for i in inds]
                keys = list(f2['data'].keys())
                for i, key in enumerate(keys):
                    if key.startswith('demo_'):
                        # Automatically increment the key number
                        
                        # if i in [0,13,16,17]:
                        #     if i != 0:
                        #         assert f2['data'][key].attrs['label'] == 2
                        #     continue
                        
                        max_num += 1
                        new_key = f'demo_{max_num}'
                        f2.copy(f'data/{key}', data_group, name=new_key)
                        print(f"Copied {key} as {new_key}")
                    else:
                        f2.copy(f'data/{key}', data_group)
                        print(f"Copied {key}")
                print('the number of all the demos:', len(data_group.keys()))
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type = str, nargs = '+', help = "List of file paths to combine")
    parser.add_argument("--output_file", type = str, help = "Output file path")
    
    args = parser.parse_args()
    combine_hdf5_files(args)