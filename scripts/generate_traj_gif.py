import h5py
import numpy as np
import imageio
def save_image_sequence_as_gif(images, output_file, fps=30):
    "save image sequence as gif"
    imageio.mimsave(output_file, images, fps=fps)
    
def main():
    ## load the trajectory data from the hdf5 file
    hdf5_file = h5py.File('/home/yilin/data/robocasa/PnPCounterToSink/demo_im128_failure_front_wrist_view_127_1500.hdf5', 'r')
    ## load the hdf5 file with open
    demos = [f'demo_{i}' for i in range(101, 128)]
    for demo in demos:
        traj = hdf5_file['data'][demo]
        wrist_view = traj['obs']['robot0_eye_in_hand_image']
        front_view = traj['obs']['robot0_agentview_front_image']
        image_all = np.concatenate([wrist_view, front_view], axis=1)
        save_image_sequence_as_gif(image_all, f'/home/yilin/data/robocasa/PnPCounterToSink/failure_{demo}.gif', fps=30)
    
    
if __name__ == '__main__':
    main()