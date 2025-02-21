import h5py
import random

# Path to the input and output HDF5 files
input_file_path = '/home/yilin/Projects/failure_detection/vlm/llama-recipes/recipes/quickstart/finetuning/datasets/realcup_data/test_relabel.hdf5'
output_file_path = '/home/yilin/Projects/failure_detection/vlm/llama-recipes/recipes/quickstart/finetuning/datasets/realcup_data/test_relabel_30.hdf5'

# Open the input HDF5 file
with h5py.File(input_file_path, 'r') as infile:
    demos_by_label = {0: [], 1: [], 2: []}

    # Collect demos based on their labels
    for demo in infile['data']:
        label = infile['data'][demo].attrs['label']
        if label in demos_by_label:
            demos_by_label[label].append(demo)
    print(demos_by_label)
    # Randomly sample 10 demos for each label
    selected_demos = {label: random.sample(demos, 10) for label, demos in demos_by_label.items()}
    
    # Write selected demos to the new HDF5 file
    with h5py.File(output_file_path, 'w') as outfile:
        outfile.create_group('data')

        for label, demos in selected_demos.items():
            for demo in demos:
                # Copy the demo data and attributes
                infile.copy(f'data/{demo}', outfile['data'])
                outfile['data'][demo].attrs['label'] = label

print(f"Filtered trajectories successfully saved to {output_file_path}")
