import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from costar_dataset.block_stacking_reader_torch import CostarBlockStackingDataset

from tqdm import tqdm

if __name__ == '__main__':
    """ To test block_stacking_reader for feature_modes time_difference_images and cross_modal_embeddings
    After reading the images and vectors using the dataloader, this program asserts that images and poses are in range. 
    Args: 
        videos_path - Path to the dataset. The path should point to the costar_block_stacking_dataset_v0.4 folder level 
                      which contains the blocks_only and blocks_with_plush_toy folders.  
        feature_mode - Options include cross_modal_embeddings and time_difference_images. Default is time_difference_images.
                       time_difference_images - a feature mode where we try to classify time intervals between two frames.
                       cross_modal_embeddings - is a mode where we try to classify time intervals between a frame and a contiguous M(default 10) vectors 
                                                that comprises of encoded pose data, 6 joint angles of the UR5, change in joint angle and state of the gripper.
        batch_size - Defaults to 32
        visualize - Set true to view frames being read by the reader. Default is True.  
    Usage: python test_time_difference_reader.py --videos_path ~/.keras/datasets/costar_block_stacking_dataset_v0.4 --feature_mode cross_modal_embeddings
    """
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--videos_path', help='Path to dataset', default='~/.keras/datasets/costar_block_stacking_dataset_v0.4')
    args_parser.add_argument('--feature_mode', help = 'cross_modal_embeddings or time_difference_images',default='time_difference_images')
    args_parser.add_argument('--batch_size', help='Batch size for training', type=int, default=32)
    args_parser.add_argument('--visualize', help='To view frames, set true', action='store_true')
    args = args_parser.parse_args()

    visualize = args.visualize

    costar_dataset = CostarBlockStackingDataset.from_standard_txt(
                      root=args.videos_path,
                      version='v0.4', set_name='blocks_only', subset_name='success_only',
                      split='test', feature_mode=args.feature_mode, output_shape=(4, 128, 128),
                      num_images_per_example=200, is_training=False)
    generator = DataLoader(costar_dataset, args.batch_size, shuffle=False, num_workers=4)
    print("Length of the dataset: {}. Length of the loader: {}.".format(len(costar_dataset), len(generator)))

    generator_output = iter(generator)
    print("-------------------op")
    #x1, x2, y = next(generator_output)
    im1, im2, vec1, vec2, y = next(generator_output)
    #print("Image 1 shape: ", x1.shape, "  Image 2/Joint shape: ",x2.shape, "  Labels shape: ", y.shape)
    print("Image 1 shape: ", im1.shape, "  Image 2 shape: ", im2.shape, "Vec 1 shape: ", vec1.shape, "  Vec 2 shape: ", vec2.shape, "  Labels shape: ", y.shape)
    
    pb = tqdm(range(len(generator)-1))
    for i, (x1, x2, y) in enumerate(generator):
        pb.set_description('batch: {}'.format(i))

        y = y.numpy()
        x1 = x1.numpy()
        x2 = x2.numpy()
        distances = ['0', '1', '2','3 or 4', 'btw 5 and 20', 'btw 21 and 150']
        if visualize:
            import matplotlib
            import matplotlib.pyplot as plt
            fig = plt.figure()
            if (args.feature_mode == 'time_difference_images'):
                title = "Interval between frames is " + str(distances[y[0]])
            else:
                title = "Interval between frame and joint_vector is " + str(distances[y[0]])
            plt.title(title)            
            img1 = np.moveaxis(x1[0], 0, 2)
            img2 = np.moveaxis(x2[0], 0, 2)
            
            # image 1
            fig1 = fig.add_subplot(1,2,1)
            fig1.set_title("Frame 1")
            plt.imshow(img1[:, :, :3])
            plt.draw()
            plt.pause(0.25)

            if (args.feature_mode == 'time_difference_images'):
                # image 2
                fig2 = fig.add_subplot(1,2,2)
                fig2.set_title("Frame 2")
                plt.imshow(img2[:, :, :3])
                plt.draw()
                plt.pause(0.25)
            # uncomment the following line to wait for one window to be closed before showing the next    
            plt.show()

        assert np.all(x1 <= 1) and np.all(x1 >= -1), "Image1 is not within range!"
        if args.feature_mode == 'time_difference_images':
            assert np.all(x2 <= 1) and np.all(x2 >= -1), "Image2 is not within range!"
        else:
            assert np.all(x2[:, :,:,:7] <= 1) and np.all(x2[:,:,:,:7] >= 0), "Joint_vec is not within range!"
        assert np.all(y <= 5) and np.all(y >= 0), "y is not within range!"
        pb.update(1)
