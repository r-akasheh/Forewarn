import h5py
import cv2
import numpy as np

def update_labels(input_file, output_file_filtered, output_file_unfiltered):
    """
    Update the label of a specific demo in an HDF5 file based on user input and save to a new file.

    Args:
        hdf5_file_path (str): Path to the original HDF5 file.
        demo_index (int): Index of the demo to process (e.g., 'demo_0').
        output_file (h5py.File): HDF5 file object to save the updated data.

    Returns:
        None
    """
    f_in = h5py.File(input_file, 'r')
    f_filtered = h5py.File(output_file_filtered, 'w')
    f_unfiltered = h5py.File(output_file_unfiltered, 'w')
    data_grp_filtered = f_filtered.create_group("data")
    data_grp_unfiltered = f_unfiltered.create_group("data")

    demos = sorted(list(f_in["data"].keys()))
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    total_samples = 0
    new_index = 0

    NUM_START_STATES = 20  # Number of start states to process
    NUM_ROLLOUTS_PER_STATE = 10  # Number of rollouts per start state

    for i in range(NUM_START_STATES):
        one_state_demo_keys = demos[i*NUM_ROLLOUTS_PER_STATE:(i+1)*NUM_ROLLOUTS_PER_STATE]
        print(f"Processing demos: {one_state_demo_keys}")

        labels = []

        for demo_key in one_state_demo_keys:

            length = len(f_in['data'][demo_key]['obs']['robot0_eye_in_hand_image'])
            if 'label' in f_in['data'][demo_key].attrs:
                label = f_in['data'][demo_key].attrs['label']
            else: 
                label = None

            frame_delay = int(10)
            # for j in range(length):
                # print('step', i)
            agentview_image = f_in['data'][demo_key]['obs']['agentview_image'][-1]
            agentview_image = cv2.cvtColor(agentview_image, cv2.COLOR_RGB2BGR)
            robot0_eye_in_hand_image = f_in['data'][demo_key]['obs']['robot0_eye_in_hand_image'][-1]
            robot0_eye_in_hand_image = cv2.cvtColor(robot0_eye_in_hand_image, cv2.COLOR_RGB2BGR)
            #concatenate the two images
            image = np.concatenate((agentview_image, robot0_eye_in_hand_image), axis=1)
            # save to png
            cv2.imwrite(f"image.png", image)
        # return

    # Render the image and display it
    #         cv2.imshow(f"Demo {demo_key} - Label: {label}", agentview_image)
    # # print("Press a key to update the label (0, 1, or 2):")
    #         cv2.waitKey(frame_delay)  # Wait for the specified frame delay
            # key = input("Press Enter to continue...")
            key = input("input number...")
            if key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                labels.append(int(key))
                # 1 = left, 2 = right, 0 = failure

            # cv2.destroyAllWindows()

            ep_data_grp = data_grp_unfiltered.create_group(demo_key) 
    
            ep_data_grp.create_dataset("actions", data=np.array(f_in["data"][demo_key]["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(f_in["data"][demo_key]["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(f_in["data"][demo_key]["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(f_in["data"][demo_key]["dones"]))
            for k in f_in["data"][demo_key]["obs"]:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(f_in["data"][demo_key]["obs"][k]))
            for k in f_in["data"][demo_key]["next_obs"]:
                ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(f_in["data"][demo_key]["next_obs"][k]))
            ep_data_grp.attrs["label"] = labels[-1]  # Save the label for this demo

            print(f_in["data"][demo_key].keys())

            # episode metadata
            if "model_file" in f_in["data"][demo_key].attrs:
                ep_data_grp.attrs["model_file"] = f_in["data"][demo_key].attrs["model_file"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = f_in["data"][demo_key].attrs["num_samples"] # number of transitions in this episode

            if "camera_info" in f_in["data"][demo_key].attrs:
                ep_data_grp.attrs["camera_info"] = f_in["data"][demo_key].attrs["camera_info"]

        if 1 in labels and 2 in labels and 0 in labels:

            for j, original_demo_key in enumerate(one_state_demo_keys):
                ep_data_grp = data_grp_filtered.create_group("demo_{}".format(new_index + j)) 
        
                ep_data_grp.create_dataset("actions", data=np.array(f_in["data"][original_demo_key]["actions"]))
                ep_data_grp.create_dataset("states", data=np.array(f_in["data"][original_demo_key]["states"]))
                ep_data_grp.create_dataset("rewards", data=np.array(f_in["data"][original_demo_key]["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(f_in["data"][original_demo_key]["dones"]))
                for k in f_in["data"][original_demo_key]["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(f_in["data"][original_demo_key]["obs"][k]))
                for k in f_in["data"][original_demo_key]["next_obs"]:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(f_in["data"][original_demo_key]["next_obs"][k]))
                ep_data_grp.attrs["label"] = labels[j]  # Save the label for this demo

                print(f_in["data"][original_demo_key].keys())

                # episode metadata
                if "model_file" in f_in["data"][original_demo_key].attrs:
                    ep_data_grp.attrs["model_file"] = f_in["data"][original_demo_key].attrs["model_file"] # model xml for this episode
                ep_data_grp.attrs["num_samples"] = f_in["data"][original_demo_key].attrs["num_samples"] # number of transitions in this episode

                if "camera_info" in f_in["data"][original_demo_key].attrs:
                    ep_data_grp.attrs["camera_info"] = f_in["data"][original_demo_key].attrs["camera_info"]

                total_samples += f_in["data"][original_demo_key].attrs["num_samples"]

            new_index += len(one_state_demo_keys)

            print(f"Updated {len(one_state_demo_keys)} demos with labels: {labels}")
        else:
            print(f"Skipping demos {one_state_demo_keys} due to insufficient labels: {labels}")

    if "mask" in f_in:
        f_in.copy("mask", f_filtered)
        f_in.copy("mask", f_unfiltered)


    # global metadata
    data_grp_filtered.attrs["total"] = total_samples  # total number of demos
    data_grp_filtered.attrs["env_args"] = f_in["data"].attrs["env_args"]

    data_grp_unfiltered.attrs["total"] = f_in["data"].attrs["total"]  # total number of demos
    data_grp_unfiltered.attrs["env_args"] = f_in["data"].attrs["env_args"]

    f_in.close()
    f_unfiltered.close()
    f_filtered.close()

# Example usage
# main('path_to_file.hdf5', 'path_to_new_file.hdf5')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the original HDF5 file')
    parser.add_argument('--output_filtered', type=str, required=True, help='Path to the output HDF5 file')
    parser.add_argument('--output_unfiltered', type=str, required=True, help='Path to the output HDF5 file for unfiltered data')
    args = parser.parse_args()
    update_labels(args.input, args.output_filtered, args.output_unfiltered)