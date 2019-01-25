'''Dataset loader for PyTorch.
Originally written for Tensorflow by Andrew Hundt (ahundt) in jhu-lcsr/costar_plan (https://github.com/jhu-lcsr/costar_plan)
Ported to PyTorch by Chia-Hung Lin (rexxarchl)
'''
import h5py
import os
import io
import sys
import glob
import traceback
from PIL import Image
from skimage.transform import resize

import numpy as np
from numpy.random import RandomState
# import json
import costar_dataset.hypertree_pose_metrics_torch as hypertree_pose_metrics
from torch.utils.data import Dataset, DataLoader
import scipy


def random_eraser(input_img, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=True):
    """ Cutout and random erasing algorithms for data augmentation

    source:
    https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
    """
    img_h, img_w, img_c = input_img.shape
    p_1 = np.random.rand()

    if p_1 > p:
        return input_img

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))
    else:
        c = np.random.uniform(v_l, v_h)

    input_img[top:top + h, left:left + w, :] = c

    return input_img


def tile_vector_as_image_channels_np(vector_op, image_shape):
    """
    Takes a vector of length n and an image shape BHWC,
    and repeat the vector as channels at each pixel.

    # Params

      vector_op: A tensor vector to tile.
      image_shape: A list of integers [width, height] with the desired dimensions.
    """
    # input vector shape
    ivs = np.shape(vector_op)
    # reshape the vector into a single pixel
    vector_pixel_shape = [ivs[0], 1, 1, ivs[1]]
    vector_op = np.reshape(vector_op, vector_pixel_shape)
    # tile the pixel into a full image
    tile_dimensions = [1, image_shape[1], image_shape[2], 1]
    vector_op = np.tile(vector_op, tile_dimensions)

    return vector_op


def concat_images_with_tiled_vector_np(images, vector):
    """Combine a set of images with a vector, tiling the vector at each pixel in the images and concatenating on the channel axis.

    # Params

        images: list of images with the same dimensions
        vector: vector to tile on each image. If you have
            more than one vector, simply concatenate them
            all before calling this function.

    # Returns

    """
    if not isinstance(images, list):
        images = [images]
    image_shape = np.shape(images[0])
    tiled_vector = tile_vector_as_image_channels_np(vector, image_shape)
    images.append(tiled_vector)
    combined = np.concatenate(images, axis=-1)

    return combined


def concat_unit_meshgrid_np(tensor):
    """ Concat unit meshgrid onto the tensor.

    This is roughly equivalent to the input in uber's coordconv.
    TODO(ahundt) concat_unit_meshgrid_np is untested.
    """
    assert len(tensor.shape) == 4
    # print('tensor shape: ' + str(tensor.shape))
    y_size = tensor.shape[1]
    x_size = tensor.shape[2]
    max_value = max(x_size, y_size)
    y, x = np.meshgrid(np.arange(y_size),
                       np.arange(x_size),
                       indexing='ij')
    assert y.size == x.size and y.size == tensor.shape[1] * tensor.shape[2]
    # print('x shape: ' + str(x.shape) + ' y shape: ' + str(y.shape))
    # rescale data and reshape to have the same dimension as the tensor
    y = np.reshape(y / max_value, [1, y.shape[0], y.shape[1], 1])
    x = np.reshape(x / max_value, [1, x.shape[0], x.shape[1], 1])

    # need to have a meshgrid for each example in the batch,
    # so tile along batch axis
    tile_dimensions = [tensor.shape[0], 1, 1, 1]
    y = np.tile(y, tile_dimensions)
    x = np.tile(x, tile_dimensions)
    combined = np.concatenate([tensor, y, x], axis=-1)
    return combined


def blend_images_np(image, image2, alpha=0.5):
    """Draws image2 on an image.
    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      image2: a uint8 numpy array of shape (img_height, img_height) with
        values between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)
    Raises:
      ValueError: On incorrect data type for image or image2s.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if image2.dtype != np.uint8:
        raise ValueError('`image2` not of type np.uint8')
    if image.shape[:2] != image2.shape[:2]:
        raise ValueError('The image has spatial dimensions %s but the image2 has '
                         'dimensions %s' % (image.shape[:2], image2.shape[:2]))
    pil_image = Image.fromarray(image)
    pil_image2 = Image.fromarray(image2)

    pil_image = Image.blend(pil_image, pil_image2, alpha)
    np.copyto(image, np.array(pil_image.convert('RGB')))
    return image


def blend_image_sequence(images, alpha=0.5, verbose=0):
    """ Blend past goal images
    """
    blended_image = images[0]
    if len(images) > 1:
        for image in images[1:]:
            if verbose > 1:
                print('image type: ' + str(type(image)) + ' dtype: ' + str(image.dtype))
            blended_image = blend_images_np(blended_image, image)
    return blended_image


def get_past_goal_indices(current_robot_time_index, goal_indices, filename='', verbose=0):
    """ get past goal image indices, including the initial image

    # Arguments

    current_robot_time_index: the index of the current "robot time" being simulated
    goal_indices: a list of goal time indices for every robot time

    # Returns

    A list of indices representing all the goal time steps
    """
    image_indices = [0]
    total_goal_indices = len(goal_indices)
    if verbose:
        print('total images: ' + str(total_goal_indices))
    image_index = 1
    while image_index < current_robot_time_index and image_index < total_goal_indices:
        if verbose > 0:
            print('image_index: ' + str(image_index))
        goal_image_index = goal_indices[image_index]
        if goal_image_index < current_robot_time_index and goal_image_index < total_goal_indices:
            if verbose > 0:
                print('goal_indices[goal_image_index]: ' + str(goal_indices[goal_image_index]))
            image_indices += [goal_image_index]
            if goal_image_index <= goal_indices[goal_image_index]:
                image_index += 1
        # TODO(ahundt) understand the cause of the warning below, modify the preprocessing script to correct it
        elif goal_image_index >= total_goal_indices and verbose > 0:
            print('block_stacking_reader.py::get_past_goal_indices(): warning, goal index equals '
                  'or exceeds total_goal_indices. filename: ' + str(filename) +
                  ' goal_image_index: ' + str(goal_image_index) +
                  ' total_goal_indices: ' + str(total_goal_indices))
        image_index = goal_image_index
    return image_indices


def encode_label(label_features_to_extract, y, action_successes=None, random_augmentation=None, current_stacking_reward=None):
    """ Encode a label based on the features that need to be extracted from the pose y.

    y: list of poses in [[x, y, z, qx, qy, qz, qw]] format
    action_successes: list of labels with successful actions
    """
    # determine the label
    if label_features_to_extract is None or 'grasp_goal_xyz_3' in label_features_to_extract:
        # regression to translation case, see semantic_translation_regression in hypertree_train.py
        y = hypertree_pose_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(y, random_augmentation=random_augmentation)
        y = y[:, :3]
    elif label_features_to_extract is None or 'grasp_goal_aaxyz_nsc_5' in label_features_to_extract:
        # regression to rotation case, see semantic_rotation_regression in hypertree_train.py
        y = hypertree_pose_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(y, random_augmentation=random_augmentation)
        y = y[:, 3:]
    elif label_features_to_extract is None or 'grasp_goal_xyz_aaxyz_nsc_8' in label_features_to_extract:
        # default, regression label case
        y = hypertree_pose_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(y, random_augmentation=random_augmentation)
    elif 'grasp_success' in label_features_to_extract or 'action_success' in label_features_to_extract:
        if action_successes is None:
            raise ValueError(
                    'encode_label() was not provided with action_successes, '
                    'which should contain data about the future outcome of the action.')
        # classification label case
        y = action_successes
    elif 'stacking_reward' in label_features_to_extract:
        y = current_stacking_reward
    else:
        raise ValueError('Unsupported label_features_to_extract: ' + str(label_features_to_extract))
    return y


def is_string_an_int(s):
    s = s.strip()
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def encode_action(
        action, possible_actions=None, data_features_to_extract=None,
        total_actions_available=41, one_hot_encoding=True):
    """ Encode an action for feeding to the neural network.
    """
    if isinstance(action, str) and is_string_an_int(action):
        action_index = int(action)
    elif isinstance(action, int):
        action_index = action
    elif possible_actions is not None:
        action_index, = np.where(action == possible_actions)
    else:
        raise ValueError(
            'encode_action() called with action ' + str(action) + ' which is not supported. Try an int, '
            'a string containing an int, or a string matching a list of actions')
    if (data_features_to_extract is not None and
            ('image_0_image_n_vec_xyz_aaxyz_nsc_15' in data_features_to_extract or
             'image_0_image_n_vec_xyz_nxygrid_12' in data_features_to_extract or
             'image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17' in data_features_to_extract or
             'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25' in data_features_to_extract) and not one_hot_encoding):
        # normalized floating point encoding of action vector
        # from 0 to 1 in a single float which still becomes
        # a 2d array of dimension batch_size x 1
        # np.expand_dims(data['gripper_action_label'][indices[1:]], axis=-1) / self.total_actions_available
        action = [float(action_index / total_actions_available)]
    else:
        # generate the action label one-hot encoding
        action = np.zeros(total_actions_available)
        action[action_index] = 1
    return action


def encode_action_and_images(
        data_features_to_extract,
        poses,
        action_labels,
        init_images,
        current_images,
        y=None,
        random_augmentation=None,
        encoded_goal_pose=None,
        epsilon=1e-3):
    """ Given an action and images, return the combined input object performing prediction with keras.

    data_features_to_extract: A string identifier for the encoding to use for the actions and images.
        Options include: 'image_0_image_n_vec_xyz_aaxyz_nsc_15', 'image_0_image_n_vec_xyz_10',
            'current_xyz_aaxyz_nsc_8', 'current_xyz_3', 'proposed_goal_xyz_aaxyz_nsc_8'.
    action_labels: batch of already one-hot or floating point encoded action label
    init_images: batch of clear view images, the initial in the time series.
        These should already be the appropriate size and rgb values in the range [0, 255].
    current_images: batch of current image in the time series.
        These should already be the appropriate size and rgb values in the range [0, 255].
    y: labels, particularly useful when classifying the quality of a regressed action.
    random_augmentation: None has no effect, if given a float from 0 to 1
        it will modify the poses with a small amount of translation and rotation
        with the probablity specified by the provided floating point number.
    encoded_goal_pose: A pre-encoded goal pose for use in actor/critic classification of proposals.
    """

    action_labels = np.array(action_labels)
    init_images = preprocess_numpy_input(np.array(init_images, dtype=np.float32))
    current_images = preprocess_numpy_input(np.array(current_images, dtype=np.float32))
    poses = np.array(poses)

    # print('poses shape: ' + str(poses.shape))
    encoded_poses = hypertree_pose_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(
        poses, random_augmentation=random_augmentation)
    if data_features_to_extract is None or 'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25':
        # TODO(ahundt) This should actually encode two poses like the commented encoded_poses line below because it is for grasp proposal success/failure
        # classification. First need to double check all code that uses it in enas and costar_plan
        encoded_goal_pose = hypertree_pose_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(
            poses, random_augmentation=random_augmentation)
        # encoded_poses = np.array([encoded_poses, encoded_goal_pose])

    if np.any(encoded_poses < 0 - epsilon) or np.any(encoded_poses > 1 + epsilon):
        raise ValueError('An encoded pose was outside the [0,1] range! Update your encoding. poses: ' +
                         str(poses) + ' encoded poses: ' + str(encoded_poses))

    if (data_features_to_extract is None or
            'current_xyz_3' in data_features_to_extract or
            'image_0_image_n_vec_xyz_10' in data_features_to_extract or
            'image_0_image_n_vec_xyz_nxygrid_12' in data_features_to_extract):
        # regression input case for translation only
        action_poses_vec = np.concatenate([encoded_poses[:, :3], action_labels], axis=-1)
        X = [init_images, current_images, action_poses_vec]
    elif (data_features_to_extract is None or
            'current_xyz_aaxyz_nsc_8' in data_features_to_extract or
            'image_0_image_n_vec_xyz_aaxyz_nsc_15' in data_features_to_extract or
            'image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17' in data_features_to_extract):
        # default, regression input case for translation and rotation
        action_poses_vec = np.concatenate([encoded_poses, action_labels], axis=-1)
        X = [init_images, current_images, action_poses_vec]
    elif(data_features_to_extract is None or 'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25' in data_features_to_extract):
        # this is for classification of actions
        action_poses_vec = np.concatenate([encoded_poses, encoded_goal_pose, action_labels], axis=-1)
        X = [init_images, current_images, action_poses_vec]
    elif 'proposed_goal_xyz_aaxyz_nsc_8' in data_features_to_extract:
        # classification input case
        proposed_and_current_action_vec = np.concatenate([encoded_poses, action_labels, y], axis=-1)
        X = [init_images, current_images, proposed_and_current_action_vec]

    else:
        raise ValueError('Unsupported data input: ' + str(data_features_to_extract))

    if (data_features_to_extract is not None and
            ('image_0_image_n_vec_xyz_10' in data_features_to_extract or
             'image_0_image_n_vec_xyz_aaxyz_nsc_15' in data_features_to_extract or
             'image_0_image_n_vec_xyz_nxygrid_12' in data_features_to_extract or
             'image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17' in data_features_to_extract or
             'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25' in data_features_to_extract)):
        # make the giant data cube if it is requested
        vec = np.squeeze(X[2:])
        assert len(vec.shape) == 2, 'we only support a 2D input vector for now but found shape:' + str(vec.shape)
        X = concat_images_with_tiled_vector_np(X[:2], vec)

    # check if any of the data features expect nxygrid normalized x, y coordinate grid values
    grid_labels = [s for s in data_features_to_extract if 'nxygrid' in s]
    # print('grid labels: ' + str(grid_labels))
    if (data_features_to_extract is not None and grid_labels):
        X = concat_unit_meshgrid_np(X)
    return X


def inference_mode_gen(file_names):
    """ Generate data for all time steps in a single example.
    """
    file_list_updated = []
    # print(len(file_names))
    for f_name in file_names:
        with h5py.File(f_name, 'r') as data:
            file_len = len(data['gripper_action_goal_idx']) - 1
            # print(file_len)
            list_id = [f_name] * file_len
        file_list_updated = file_list_updated + list_id
    return file_list_updated


def preprocess_numpy_input(x):
    """From keras_applications.imagenet_utils
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py

    Preprocesses a Numpy array encoding a batch of images.
    Will scale pixels between -1 and 1, sample-wise. ('tf' mode in original code)
    # Arguments
        x: Input array, 3D or 4D.
    # Returns
        Preprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(np.float32, copy=False)

    x /= 127.5
    x -= 1.
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0., interpolation_order=1):
    """Performs a random spatial shift of a Numpy image tensor.
    From keras_preprocessing.images.random_shift
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order int: order of spline interpolation.
            see `ndimage.interpolation.affine_transform`
    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval,
                               order=interpolation_order)
    return x


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.
    From keras_preprocessing.images.apply_affine_transform
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py
    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order int: order of interpolation
    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    """From keras_preprocessing.images.apply_affine_transform
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


class CostarBlockStackingDataset(Dataset):
    def __init__(self, list_example_filenames,
                 label_features_to_extract=None, data_features_to_extract=None,
                 total_actions_available=41,
                 seed=0, random_state=None,
                 is_training=True, random_augmentation=None,
                 random_shift=False,
                 output_shape=None,
                 blend_previous_goal_images=False,
                 estimated_time_steps_per_example=250, verbose=0, inference_mode=False, one_hot_encoding=True,
                 pose_name='pose_gripper_center',
                 force_random_training_pose_augmentation=None):
        '''Initialization

        # Arguments

        list_Ids: a list of file paths to be read
        seed: a random seed to use. If seed is None it will be in order!
        random_state: A numpy RandomState object, if not provided one will be generated from the seed.
            Used exclusively for example data ordering and the indices to visit within an example.
        # TODO(ahundt) better notes about the two parameters below. See choose_features_and_metrics() in cornell_grasp_trin.py.
        label_features_to_extract: defaults to regression options, classification options are also available
        data_features_to_extract: defaults to regression options, classification options are also available
            Options include 'image_0_image_n_vec_xyz_aaxyz_nsc_15' which is a giant NHWC cube of image and pose data,
            'current_xyz_aaxyz_nsc_8' a vector with the current pose,
            'proposed_goal_xyz_aaxyz_nsc_8' a pose at the end of the current action (for classification cases),
            'image_0_image_n_vec_xyz_nxygrid_12' another giant cube without rotation and with explicit normalized xy coordinates,
            'image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17' another giant cube with rotation and explicit normalized xy coordinates.
        random_augmentation: None or a float value between 0 and 1 indiciating how frequently random augmentation should be applied.
        estimated_time_steps_per_example: The number of images in each example varies,
            so we simply sample in proportion to an estimated number of images per example.
            Due to random sampling, there is no guarantee that every image will be visited once!
            However, the images can be visited in a fixed order, particularly when is_training=False.
        one_hot_encoding flag triggers one hot encoding and thus numbers at the end of labels might not correspond to the actual size.
        force_random_training_pose_augmentation: override random_augmenation when training for pose data only.
        pose_name: Which pose to use as the robot 3D position in space. Options include:
            'pose' is the end effector ee_link pose at the tip of the connector
                of the robot, which is the base of the gripper wrist.
            'pose_gripper_center' is a point in between the robotiq C type gripping plates when the gripper is open
                with the same orientation as pose.

        # Explanation of abbreviations:

        aaxyz_nsc: is an axis and angle in xyz order, where the angle is defined by a normalized sin(theta) cos(theta).
        nxygrid: at each pixel, concatenate two additional channels containing the pixel coordinate x and y as values between 0 and 1.
            This is similar to uber's "coordconv" paper.
        '''
        if random_state is None:
            random_state = RandomState(seed)
        # self.batch_size = batch_size
        self.list_example_filenames = list_example_filenames
        # self.shuffle = shuffle
        self.seed = seed
        self.random_state = random_state
        self.output_shape = output_shape
        self.is_training = is_training
        self.verbose = verbose
        # self.on_epoch_end()
        if isinstance(label_features_to_extract, str):
            label_features_to_extract = [label_features_to_extract]
        self.label_features_to_extract = label_features_to_extract
        # TODO(ahundt) total_actions_available can probably be extracted from the example hdf5 files and doesn't need to be a param
        if isinstance(data_features_to_extract, str):
            data_features_to_extract = [data_features_to_extract]
        self.data_features_to_extract = data_features_to_extract
        self.total_actions_available = total_actions_available
        self.random_augmentation = random_augmentation
        self.random_shift = random_shift
        self.inference_mode = inference_mode
        self.infer_index = 0
        self.one_hot_encoding = one_hot_encoding
        self.pose_name = pose_name

        if self.seed is not None and not self.is_training:
            # repeat the same order if we're validating or testing
            # continue the large random sequence for training
            self.random_state.seed(self.seed)
            # torch.manual_seed(self.seed)

        # the pose encoding augmentation can be specially added separately from all other augmentation
        self.random_encoding_augmentation = None
        if self.is_training:
            if self.random_augmentation:
                self.random_encoding_augmentation = self.random_augmentation
            elif force_random_training_pose_augmentation is not None:
                self.random_encoding_augmentation = force_random_training_pose_augmentation

        self.blend = blend_previous_goal_images
        self.estimated_time_steps_per_example = estimated_time_steps_per_example
        if self.inference_mode is True:
            self.list_example_filenames = inference_mode_gen(self.list_example_filenames)

    def __len__(self):
        """Return the lenth of file names
        """
        return len(self.list_example_filenames)

    def __getitem__(self, index):
        '''Generate one example of data
        '''
        # Generate indexes of the batch
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # if self.verbose > 0:
        #     print("batch getitem indices:" + str(indexes))
        # # Find list of example_filenames
        # list_example_filenames_temp = [self.list_example_filenames[k] for k in indexes]
        # Generate data
        self.infer_index = self.infer_index + 1
        X, y = self.__data_generation(self.list_example_filenames[index], self.infer_index)

        return X, y

    def get_estimated_time_steps_per_example(self):
        """ Get the estimated images per example,

        Run extra steps in proportion to this if you want to get close to visiting every image.
        """
        return self.estimated_time_steps_per_example

    def __data_generation(self, data_path, images_index):
        """ Generates data containing batch_size samples

        # Arguments

        data_path: the file path to be read
        """

        def JpegToNumpy(jpeg):
            stream = io.BytesIO(jpeg)
            im = np.asarray(Image.open(stream))
            try:
                return im.astype(np.uint8)
            except(TypeError) as exception:
                print("Failed to convert PIL image type", exception)
                print("type ", type(im), "len ", len(im))

        def ConvertImageListToNumpy(data, format='numpy', data_format='NHWC'):
            """ Convert a list of binary jpeg or png files to numpy format.

            # Arguments

            data: a list of binary jpeg images to convert
            format: default 'numpy' returns a 4d numpy array,
                'list' returns a list of 3d numpy arrays
            """
            # length = len(data)
            imgs = []
            for raw in data:
                img = JpegToNumpy(raw)
                if data_format == 'NCHW':
                    img = np.transpose(img, [2, 0, 1])
                imgs.append(img)
            if format == 'numpy':
                imgs = np.array(imgs)
            return imgs
        try:
            # Initialization
            if self.verbose > 0:
                print("generating data: " + str(data_path))
            X = []
            init_images = []
            current_images = []
            poses = []
            goal_pose = []
            y = []
            action_labels = []
            action_successes = []
            example_filename = ''

            # Generate data
            example_filename = os.path.expanduser(data_path)
            if self.verbose > 0:
                print('reading from path: ' + str(example_filename))
            # Store sample
            # X[i,] = np.load('data/' + example_filename + '.npy')
            x = ()
            try:
                if not os.path.isfile(example_filename):
                    raise ValueError('CostarBlockStackingDataset: Trying to open something which is not a file: ' + str(example_filename))
                with h5py.File(example_filename, 'r') as data:
                    if 'gripper_action_goal_idx' not in data or 'gripper_action_label' not in data:
                        raise ValueError('block_stacking_reader.py: You need to run preprocessing before this will work! \n' +
                                         '    python2 ctp_integration/scripts/view_convert_dataset.py --path ~/.keras/datasets/costar_block_stacking_dataset_v0.4 --preprocess_inplace gripper_action --write'
                                         '\n File with error: ' + str(example_filename))
                    # indices = [0]
                    # len of goal indexes is the same as the number of images, so this saves loading all the images
                    all_goal_ids = np.array(data['gripper_action_goal_idx'])
                    if('stacking_reward' in self.label_features_to_extract):
                        # TODO(ahundt) move this check out of the stacking reward case after files have been updated
                        if all_goal_ids[-1] > len(all_goal_ids):
                            raise ValueError(' File contains goal id greater than total number of frames ' + str(example_filename))
                    if len(all_goal_ids) < 2:
                        print('block_stacking_reader.py: ' + str(len(all_goal_ids)) + ' goal indices in this file, skipping: ' + example_filename)
                    if 'success' in example_filename:
                        label_constant = 1
                    else:
                        label_constant = 0
                    stacking_reward = np.arange(len(all_goal_ids))
                    stacking_reward = 0.999 * stacking_reward * label_constant
                    # print("reward estimates", stacking_reward)

                    if self.seed is not None:
                        rand_max = len(all_goal_ids) - 1
                        if rand_max <= 1:
                            print('CostarBlockStackingDataset: not enough goal ids: ' + str(all_goal_ids) + ' file: ' + str(rand_max))
                        image_indices = self.random_state.randint(1, rand_max, 1)
                    else:
                        raise NotImplementedError
                    indices = [0] + list(image_indices)

                    if self.blend:
                        img_indices = get_past_goal_indices(image_indices, all_goal_ids, filename=example_filename)
                    else:
                        img_indices = indices
                    if self.inference_mode is True:
                        if images_index >= len(data['gripper_action_goal_idx']):
                            self.infer_index = 1
                            image_idx = 1
                            # image_idx = (images_index % (len(data['gripper_action_goal_idx']) - 1)) + 1
                        else:
                            image_idx = images_index

                        img_indices = [0, image_idx]
                        # print("image_index", image_idx)
                        # print("image_true", images_index, len(data['gripper_action_goal_idx']))
                        # print("new_indices-----", image_idx)
                    if self.verbose > 0:
                        print("Indices --", indices)
                        print('img_indices: ' + str(img_indices))
                    rgb_images = list(data['image'][img_indices])
                    rgb_images = ConvertImageListToNumpy(rgb_images, format='numpy')

                    if self.blend:
                        # TODO(ahundt) move this to after the resize loop for a speedup
                        blended_image = blend_image_sequence(rgb_images)
                        rgb_images = [rgb_images[0], blended_image]
                    # resize using skimage
                    rgb_images_resized = []
                    for k, images in enumerate(rgb_images):
                        if (self.is_training and self.random_augmentation is not None and
                                self.random_shift and np.random.random() > self.random_augmentation):
                            # apply random shift to the images before resizing
                            images = random_shift(images,
                                                  # height, width
                                                  1./(48. * 2.), 1./(64. * 2.),
                                                  row_axis=0, col_axis=1, channel_axis=2)
                        # TODO(ahundt) improve crop/resize to match cornell_grasp_dataset_reader
                        if self.output_shape is not None:
                            resized_image = resize(images, self.output_shape, mode='constant', preserve_range=True, order=1)
                        else:
                            resized_image = images
                        if self.is_training and self.random_augmentation:
                            # do some image augmentation with random erasing & cutout
                            resized_image = random_eraser(resized_image)
                        rgb_images_resized.append(resized_image)

                    init_images.append(rgb_images_resized[0])
                    current_images.append(rgb_images_resized[1])
                    poses.append(np.array(data[self.pose_name][indices[1:]])[0])
                    if(self.data_features_to_extract is not None and 'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25' in self.data_features_to_extract):
                        next_goal_idx = all_goal_ids[indices[1:][0]]
                        goal_pose.append(np.array(data[self.pose_name][next_goal_idx]))
                        print("final pose added", goal_pose)
                        current_stacking_reward = stacking_reward[indices[1]]
                        print("reward estimate", current_stacking_reward)
                    # x = x + tuple([rgb_images[indices]])
                    # x = x + tuple([np.array(data[self.pose_name])[indices]])

                    # WARNING: IF YOU CHANGE THIS ACTION ENCODING CODE BELOW, ALSO CHANGE encode_action() function ABOVE
                    if (self.data_features_to_extract is not None and
                            ('image_0_image_n_vec_xyz_aaxyz_nsc_15' in self.data_features_to_extract or
                             'image_0_image_n_vec_xyz_nxygrid_12' in self.data_features_to_extract or
                             'image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17' in self.data_features_to_extract or
                             'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25' in self.data_features_to_extract) and not self.one_hot_encoding):
                        # normalized floating point encoding of action vector
                        # from 0 to 1 in a single float which still becomes
                        # a 2d array of dimension batch_size x 1
                        # np.expand_dims(data['gripper_action_label'][indices[1:]], axis=-1) / self.total_actions_available
                        for j in indices[1:]:
                            action = [float(data['gripper_action_label'][j] / self.total_actions_available)]
                            action_labels.append(action)
                    else:
                        # one hot encoding
                        for j in indices[1:]:
                            # generate the action label one-hot encoding
                            action = np.zeros(self.total_actions_available)
                            action[data['gripper_action_label'][j]] = 1
                            action_labels.append(action)
                    # action_labels = np.array(action_labels)

                    # print(action_labels)
                    # x = x + tuple([action_labels])
                    # X.append(x)
                    # action_labels = np.unique(data['gripper_action_label'])
                    # print(np.array(data['labels_to_name']).shape)
                    # X.append(np.array(data['pose'])[indices])

                    # Store class
                    label = ()
                    # change to goals computed
                    index1 = indices[1]
                    goal_ids = all_goal_ids[index1]
                    # print(index1)
                    label = np.array(data[self.pose_name])[goal_ids]
                    # print(type(label))
                    # for items in list(data['all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json'][indices]):
                    #     json_data = json.loads(items.decode('UTF-8'))
                    #     label = label + tuple([json_data['gripper_center']])
                    #     print(np.array(json_data['gripper_center']))
                    #     print(json_data.keys())
                    #     y.append(np.array(json_data['camera_rgb_frame']))
                    if('stacking_reward' in self.label_features_to_extract):
                        # print(y)
                        y.append(current_stacking_reward)
                    else:
                        y.append(label)
                    if 'success' in example_filename:
                        action_successes = action_successes + [1]
                    else:
                        action_successes = action_successes + [0]
                    # print("y = ", y)
            except IOError as ex:
                print('Error: Skipping file due to IO error when opening ' + example_filename + ': ' + str(ex))

            action_labels = np.array(action_labels)
            init_images = preprocess_numpy_input(np.array(init_images, dtype=np.float32))
            current_images = preprocess_numpy_input(np.array(current_images, dtype=np.float32))
            poses = np.array(poses)

            # encoded_goal_pose = None
            # print('encoded poses shape: ' + str(encoded_poses.shape))
            # print('action labels shape: ' + str(action_labels.shape))
            # print('encoded poses vec shape: ' + str(action_poses_vec.shape))
            # print("---",init_images.shape)
            # init_images = tf.image.resize_images(init_images,[224,224])
            # current_images = tf.image.resize_images(current_images,[224,224])
            # print("---",init_images.shape)
            # X = init_images

            X = encode_action_and_images(
                data_features_to_extract=self.data_features_to_extract,
                poses=poses, action_labels=action_labels,
                init_images=init_images, current_images=current_images,
                y=y, random_augmentation=self.random_encoding_augmentation)

            # print("type=======",type(X))
            # print("shape=====",X.shape)

            # determine the label
            if('stacking_reward' in self.label_features_to_extract):
                y = encode_label(self.label_features_to_extract, y, action_successes, self.random_augmentation, current_stacking_reward)
            else:
                y = encode_label(self.label_features_to_extract, y, action_successes, self.random_augmentation, None)

            # Debugging checks
            if X is None:
                raise ValueError('Unsupported input data for X: ' + str(x))
            if y is None:
                raise ValueError('Unsupported input data for y: ' + str(x))

            # Assemble the data batch
            batch = (X, y)

            if self.verbose > 0:
                # diff should be nonzero for most timesteps except just before the gripper closes!
                print('encoded current poses: ' + str(poses) + ' labels: ' + str(y))
                # commented next line due to dimension issue
                # + ' diff: ' + str(poses - y))
                print("generated data: " + str(data_path))
        except Exception:
            print('CostarBlockStackingDataset: Keras will often swallow exceptions without a stack trace, '
                  'so we are printing the stack trace here before re-raising the error.')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
            raise

        return batch


if __name__ == "__main__":
    visualize = False
    output_shape = (224, 224, 3)
    # output_shape = None
    # tf.enable_eager_execution()
    filenames = glob.glob(os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.4/blocks_only/*success.h5f'))
    # print(filenames)
    # filenames_new = inference_mode_gen(filenames)
    costar_dataset = CostarBlockStackingDataset(
        filenames, verbose=1,
        output_shape=output_shape,
        label_features_to_extract='grasp_goal_xyz_aaxyz_nsc_8',
        data_features_to_extract=['current_xyz_aaxyz_nsc_8'],
        blend_previous_goal_images=False, inference_mode=False)
    num_batches = len(costar_dataset)
    print(num_batches)

    generator = DataLoader(costar_dataset, batch_size=1, shuffle=True, num_workers=1)

    for generator_output in generator:
        print("-------------------op")
        x, y = generator_output

        for i, data in enumerate(x):
            print("x[{}]: ".format(i) + str(data.shape))

        for i, data in enumerate(y):
            print("y[{}]: ".format(i) + str(data.shape))

        print("-------------------")
