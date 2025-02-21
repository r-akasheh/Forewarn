import h5py
import cv2
import numpy as np

def update_demo_label(hdf5_file_path, demo_index, output_file):
    """
    Update the label of a specific demo in an HDF5 file based on user input and save to a new file.

    Args:
        hdf5_file_path (str): Path to the original HDF5 file.
        demo_index (int): Index of the demo to process (e.g., 'demo_0').
        output_file (h5py.File): HDF5 file object to save the updated data.

    Returns:
        None
    """
    with h5py.File(hdf5_file_path, 'r') as file:
        demo_key = f'demo_{demo_index}'
        labels = [2,0,5,0,2,1,1,6,1,1,2,2,5,2,0,6,3,1,3]
        if demo_key not in file['data']:
            print(f"Demo {demo_key} not found in the HDF5 file.")
            return

        # Access the 120th image in the wrist camera view
        try:
            length = len(file['data'][demo_key]['obs']['cam_wrist_view_image'])
            # if len(file['data'][demo_key]['obs']['cam_wrist_view_image'])< 130:
                # image_data = file['data'][demo_key]['obs']['cam_wrist_view_image'][-1]
            # else:
            image_data = file['data'][demo_key]['obs']['cam_wrist_view_image']
            front_view_data = file['data'][demo_key]['obs']['cam_front_view_image']
            if 'label' in file['data'][demo_key].attrs:
                label = file['data'][demo_key].attrs['label']
            else: 
                label = None 
        except KeyError as e:
            print(f"KeyError: {e}")
            return
        except IndexError as e:
            print(f"IndexError: {e}")
            return

        # Convert the image data to a format suitable for OpenCV rendering
        # if demo_index <= len(labels):
            # new_label = labels[demo_index-1]
        # if demo_index not  in [178, 179]:
            # return 
        # else: 
        output_label = output_file['data'][demo_key].attrs['label']
        if  output_label== 4:
            for i in range(length):
                print('step', i)
                if isinstance(image_data[i], np.ndarray):
                    image = cv2.cvtColor(image_data[i], cv2.COLOR_RGB2BGR) if len(image_data[i].shape) == 3 else image_data
                    front_image = cv2.cvtColor(front_view_data[i], cv2.COLOR_RGB2BGR) if len(front_view_data[i].shape) == 3 else front_view_data
                else:
                    print("Invalid image format.")
            # return

        # Render the image and display it
                cv2.imshow(f"Demo {demo_key} - Label: {output_label}", image)
                cv2.imshow(f"Demo {demo_key} - Label: {output_label} frontview", front_image)
        # print("Press a key to update the label (0, 1, or 2):")
                cv2.waitKey(0)
            # key = input("Press Enter to continue...")
            key = input("input number...")
            # key = cv2.waitKey(0)
            # Map key inputs to labels
            # if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
            
            if key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                new_label = int(key)
                output_file['data'][demo_key].attrs['label'] = new_label
            # new_label = labels[demo_index-1]
                # print(f"Updated label for {demo_key} to {new_label}.")
            # else:
                # print("Invalid input", key)
                # print("Invalid input. Label not updated.")
                # print(f"Current label for {demo_key} is {label}.")
                # new_label = label

            # Close the OpenCV window
            cv2.destroyAllWindows()
        # new_label = demos_labels[demo_index]
        # # Copy the demo data to the new file and update the label
        # if demo_key not in output_file['data']:
        #     output_file['data'].create_group(demo_key)
        # for key, value in file['data'][demo_key].items():
        #     # file.copy(value, output_file['data'][demo_key], name=key) 
        #     # del output_file['data'][demo_key][key]
        #     if key in output_file['data'][demo_key].keys():
        #         del output_file['data'][demo_key][key]
        #         print(key)
        #     if key != 'obs':
        #         output_file['data'][demo_key][key] = value[:new_label]
        #     else:
        #         output_file['data'][demo_key].create_group('obs')
        #         for subkey in value.keys():
        #             output_file['data'][demo_key][key].create_dataset(subkey, data=value[subkey][:new_label])
        # ## also copy the attributes
        # for key, value in file['data'][demo_key].attrs.items():
        #     output_file['data'][demo_key].attrs[key] = value
        # # output_file['data'][demo_key].attrs['label'] = new_label

def update_demo_length (hdf5_file_path, demo_index, output_file):
  
    """
    Update the label of a specific demo in an HDF5 file based on user input and save to a new file.

    Args:
        hdf5_file_path (str): Path to the original HDF5 file.
        demo_index (int): Index of the demo to process (e.g., 'demo_0').
        output_file (h5py.File): HDF5 file object to save the updated data.

    Returns:
        None
    """
    with h5py.File(hdf5_file_path, 'r') as file:
        demo_key = f'demo_{demo_index}'
        if demo_key not in file['data']:
            print(f"Demo {demo_key} not found in the HDF5 file.")
            return

        # Copy the demo data to the new file and only keep the first 160 elements
        
        attrs_dict = {key: value for key, value in output_file['data'][demo_key].attrs.items()}
        del output_file['data'][demo_key]
        if demo_key not in output_file['data']:
            output_file['data'].create_group(demo_key)
        for key, value in file['data'][demo_key].items():
            # if key in output_file['data'][demo_key].keys():
                # del output_file['data'][demo_key][key]
            
            if demo_index <= 100:
                file.copy(value, output_file['data'][demo_key], name=key)
            else:
                
                cut_length = 160
                start_index = 0
                if demo_index == 134:
                    cut_length = 200
                    start_index = 99
                if demo_index == 142:
                    start_index = 46
                    cut_length = 160
                if demo_index in [178, 179]:
                    start_index = 90
                    cut_length = 170
                if key != 'obs':
                    # Copy the first 160 elements for non-'obs' keys
                    # output_file['data'][demo_key][key] = value[start_index:cut_length]
                    output_file['data'][demo_key].create_dataset(
                        key, data=value[start_index:cut_length]
                    )
                else:
                    # For 'obs', iterate over subkeys and copy the first 160 elements
                    output_file['data'][demo_key].create_group('obs')
                    for subkey in value.keys():
                        output_file['data'][demo_key][key].create_dataset(
                            subkey, data=value[subkey][start_index:cut_length]
                        )
        # Also copy the attributes
        for key, value in attrs_dict.items():
            output_file['data'][demo_key].attrs[key] = value
        # for key, value in file['data'][demo_key].attrs.items():
        #     output_file['data'][demo_key].attrs[key] = value

def main(hdf5_file_path, output_file_path):
    """
    Enumerate over all demo keys in the HDF5 file, update their labels, and save to a new file.

    Args:
        hdf5_file_path (str): Path to the original HDF5 file.
        output_file_path (str): Path to the new HDF5 file to save updated data.

    Returns:
        None
    """
    # demo_indexes = [109, 120, 129, 133, 138, 156, 171, 195, 196, 217, 218, 221, 223, 237, 247, 249, 260, 261, 263, 273, 282, 283, 289, 290, 299, 300]
    # import re

    #File path to the log file
    # file_path = "log.txt"

    # # Dictionary to store the mapping of demo IDs to their labels
    # demo_labels = {}

    # # Regular expression to match the desired pattern
    # pattern = r"Processing (\d+)\.\.\.\nUpdated label for demo_(\d+) to (\d+)\."

    # # Read the file and process its content#
    # # with open(file_path, "r") as file:
    #     content = file.read()
    #     matches = re.findall(pattern, content)

# Populate the dictionary with extracted demo IDs and labels
    # for processing_id, demo_id, label in matches:
        # demo_labels[int(demo_id)] = int(label)
    with h5py.File(hdf5_file_path, 'r') as file, h5py.File(output_file_path, 'a') as output_file:
        if 'data' not in output_file:
            
            output_file.create_group('data')
        demo_keys = list(file['data'].keys())
        ## sort by the last index
        demo_keys.sort(key=lambda x: int(x.split('_')[-1]))
        for demo_key in demo_keys:
            demo_index = int(demo_key.split('_')[-1])
            # if demo_index != 206:
            #     continue
        # for demo_index in demo_indexes:
            print(f"Processing {demo_index}...")
            update_demo_label(hdf5_file_path, demo_index, output_file)
            # update_demo_length(hdf5_file_path, demo_index, output_file)

# Example usage
# main('path_to_file.hdf5', 'path_to_new_file.hdf5')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the original HDF5 file')
    parser.add_argument('--output', type=str, required=True, help='Path to the new HDF5 file')
    args = parser.parse_args()
    main(args.input, args.output)