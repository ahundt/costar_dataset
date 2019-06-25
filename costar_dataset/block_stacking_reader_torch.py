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
import warnings

import numpy as np
from numpy.random import RandomState
# import json
import costar_dataset.hypertree_pose_metrics_torch as hypertree_pose_metrics
import torch
from torch.utils.data import Dataset, DataLoader
import scipy
import random

COSTAR_SET_NAMES = ['blocks_only', 'blocks_with_plush_toy']
COSTAR_SUBSET_NAMES = ['success_only', 'error_failure_only', 'task_failure_only', 'task_and_error_failure']
COSTAR_FEATURE_MODES = ['translation_only', 'rotation_only', 'stacking_reward', 'time_difference_images', 'cross_modal_embeddings']


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
    Takes a vector of length n and an image shape BCHW,
    and repeat the vector as channels at each pixel.

    # Params

      vector_op: A tensor vector to tile.
      image_shape: A list of integers [width, height] with the desired dimensions.
    """
    # input vector shape
    ivs = np.shape(vector_op)
    # reshape the vector into a single pixel
    # vector_pixel_shape = [ivs[0], 1, 1, ivs[1]]
    vector_pixel_shape = [ivs[0], ivs[1], 1, 1]
    vector_op = np.reshape(vector_op, vector_pixel_shape)
    # tile the pixel into a full image
    # tile_dimensions = [1, image_shape[1], image_shape[2], 1]
    tile_dimensions = [1, 1, image_shape[2], image_shape[3]]
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
    # combined = np.concatenate(images, axis=-1)
    combined = np.concatenate(images, axis=1)

    return combined


def concat_unit_meshgrid_np(tensor):
    """ Concat unit meshgrid onto the tensor.

    This is roughly equivalent to the input in uber's coordconv.
    TODO(ahundt) concat_unit_meshgrid_np is untested.
    """
    assert len(tensor.shape) == 4
    # print('tensor shape: ' + str(tensor.shape))
    # y_size = tensor.shape[1]
    # x_size = tensor.shape[2]
    y_size = tensor.shape[2]
    x_size = tensor.shape[3]
    max_value = max(x_size, y_size)
    y, x = np.meshgrid(np.arange(y_size),
                       np.arange(x_size),
                       indexing='ij')
    # assert y.size == x.size and y.size == tensor.shape[1] * tensor.shape[2]
    assert y.size == x.size and y.size == tensor.shape[2] * tensor.shape[3]
    # print('x shape: ' + str(x.shape) + ' y shape: ' + str(y.shape))
    # rescale data and reshape to have the same dimension as the tensor
    # y = np.reshape(y / max_value, [1, y.shape[0], y.shape[1], 1])
    # x = np.reshape(x / max_value, [1, x.shape[0], x.shape[1], 1])
    y = np.reshape(y / max_value, [1, 1, y.shape[0], y.shape[1]])
    x = np.reshape(x / max_value, [1, 1, x.shape[0], x.shape[1]])

    # need to have a meshgrid for each example in the batch,
    # so tile along batch axis
    tile_dimensions = [tensor.shape[0], 1, 1, 1]
    y = np.tile(y, tile_dimensions)
    x = np.tile(x, tile_dimensions)
    # combined = np.concatenate([tensor, y, x], axis=-1)
    combined = np.concatenate([tensor, y, x], axis=1)
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
        epsilon=1e-3,
        # TODO(ahundt) make single_batch_cube default to false, fix all code deps and bugs first
        single_batch_cube=True):
    """ Given an action and images, return the combined input object performing prediction with keras.

    data_features_to_extract: A string identifier for the encoding to use for the actions and images.
        Options include: 'image_0_image_n_vec_xyz_aaxyz_nsc_15', 'image_0_image_n_vec_xyz_10',
            'current_xyz_aaxyz_nsc_8', 'current_xyz_3', 'proposed_goal_xyz_aaxyz_nsc_8',
            'image_n_vec_xyz_aaxyz_nsc_q_dq_gripper', 'image_m_image_n' .
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
    single_batch_cube: False will keep images and vector data in a tuple.
       True will tile the vector data to be the same shape as the image data, and concatenate it all into a cube of data.
    """
    if data_features_to_extract == 'image_n_vec_xyz_aaxyz_nsc_q_dq_gripper':
        init_images = preprocess_numpy_input(np.array(init_images, dtype=np.double))
        poses = np.array(poses, dtype=np.float32)
        encoded_poses = []
        for pose in poses:
            pose = pose[np.newaxis]
            encoded_pose = hypertree_pose_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(
                        pose[:,:7], random_augmentation=random_augmentation)
            encoded_poses.append(np.hstack((encoded_pose[0],pose[0,7:])))
        encoded_poses = np.array(encoded_poses)
        encoded_poses = encoded_poses[np.newaxis]
        return [init_images,encoded_poses]
    elif data_features_to_extract == 'image_m_image_n':
        init_images = preprocess_numpy_input(np.array(init_images, dtype=np.double))
        current_images = preprocess_numpy_input(np.array(current_images, dtype=np.double))
        return [init_images, current_images]
    else:
        action_labels = np.array(action_labels)
        init_images = preprocess_numpy_input(np.array(init_images, dtype=np.float32))
        current_images = preprocess_numpy_input(np.array(current_images, dtype=np.float32))
        poses = np.array(poses, dtype=np.float32)

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

    if (single_batch_cube and data_features_to_extract is not None and
            ('image_0_image_n_vec_xyz_10' in data_features_to_extract or
             'image_0_image_n_vec_xyz_aaxyz_nsc_15' in data_features_to_extract or
             'image_0_image_n_vec_xyz_nxygrid_12' in data_features_to_extract or
             'image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17' in data_features_to_extract or
             'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25' in data_features_to_extract)):
        # make the giant data cube if it is requested
        # HACK(rexxarchl): in torch, batch size is controlled by DataLoader whereas in tf batch size is written in Sequence
        #                  therefore, reshape this to always be (1, num_channels)
        vec = np.squeeze(X[2:]).reshape((1, -1))
        assert len(vec.shape) == 2, 'we only support a 2D input vector for now but found shape:' + str(vec.shape)
        X = concat_images_with_tiled_vector_np(X[:2], vec)

    # check if any of the data features expect nxygrid normalized x, y coordinate grid values
    grid_labels = [s for s in data_features_to_extract if 'nxygrid' in s]
    # print('grid labels: ' + str(grid_labels))
    if (data_features_to_extract is not None and grid_labels and single_batch_cube):
        # TODO(ahundt) clean up this nxygrid stuff, which is like coordconv, it does not work if nxygrid is specified and nxygrid string is present 
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


def preprocess_numpy_input_tf(x):
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


def preprocess_numpy_input_torch(x):
    """From keras_applications.imagenet_utils
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py

    Preprocesses a Numpy array encoding a batch of images.
    Will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset.
    ('torch' mode in original code)

    # Arguments
        x: Input array, 3D or 4D. BCHW or CHW format
    # Returns
        Preprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(np.float32, copy=False)

    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if x.ndim == 3:
        x[0, :, :] -= mean[0]
        x[1, :, :] -= mean[1]
        x[2, :, :] -= mean[2]
        x[0, :, :] /= std[0]
        x[1, :, :] /= std[1]
        x[2, :, :] /= std[2]
    else:
        x[:, 0, :, :] -= mean[0]
        x[:, 1, :, :] -= mean[1]
        x[:, 2, :, :] -= mean[2]
        x[:, 0, :, :] /= std[0]
        x[:, 1, :, :] /= std[1]
        x[:, 2, :, :] /= std[2]

    print("min={}, max={}".format(np.min(x), np.max(x)))

    return x


preprocess_numpy_input = preprocess_numpy_input_tf


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

def generate_temporal_distance_training_data(video_frames):
        """
            Generate training data for 'time_difference_images' feature_mode
            Args: numpy array of frames of a video with shape - (No. of frames in video, No. of channels, height, width)
            Returns: Two normalized frames of shape - (No. of channels, height, width) each and the interval (of type int) between them
            Usage example -
                open video to read: 
                create numpy array of all frames of a video - video_frames
                generate_temporal_distance_training_data(video_frames)
        """
        frames = np.empty((2, *video_frames.shape[1:]))
        label = np.zeros(6)

        interval = random.randint(0, 5)
        if interval == 0:
            possible_frames_start = 0
            possible_frames_end = 0
        elif interval == 1:
            possible_frames_start = 1
            possible_frames_end = 1
        elif interval == 2:
            possible_frames_start = 2
            possible_frames_end = 2
        elif interval == 3:
            possible_frames_start = 3
            possible_frames_end = 4
        elif interval == 4:
            possible_frames_start = 5
            possible_frames_end = 20
        elif interval == 5:
            possible_frames_start = 21
            possible_frames_end = 150

        first_frame_index = random.randint(0, video_frames.shape[0] - possible_frames_end - 1)
        second_frame_index = random.randint(possible_frames_start, possible_frames_end)

        frames[0] = video_frames[first_frame_index] 
        frames[1] = video_frames[first_frame_index + second_frame_index]
        label[interval] = 1.
        normalized_frames = encode_action_and_images(
                 data_features_to_extract='image_m_image_n',
                 poses=None, action_labels=None,
                 init_images=frames[0], current_images=frames[1])
        return (normalized_frames[0],normalized_frames[1], interval)

def generate_crossmodal_training_data(video_frames, joints, length = 10):
    """
        Generate training data for 'time_difference_images' feature_mode
        Args: video_frames - numpy array of frames of a video with shape - (No. of frames in video, No. of channels, height, width)
              joints - numpy array of shape - (No. of frames in video, 20); 20 because of stacking of 'pose', 'q', 'dq' and 'gripper_state'
        Returns: A normalized frame of shape - (No. of channels, heigh, width)
                 encoded joint vector of shape - (1, length, 21)
                 interval (type int) that classifies how many frames there are between the frame and the vector of joints over 'length' frames
        Usage example -
                open video to read: 
                create numpy array of all frames of a video - video_frames
                create numpy array of the joint_vector recorded throughout the video  - joints
                set length, where length is the number of contiguous joint_vectors to be concatenated
                generate_crossmodal_training_data(video_frames, joints, length)
    """
    frame = np.empty((1, *video_frames.shape[1:]))
    label = np.zeros(6)

    interval = np.random.choice(6)

    if interval == 0:
        distance = 0
    elif interval == 1:
        distance = 1
    elif interval == 2:
        distance = 2
    elif interval == 3:
        distance = np.random.choice(np.arange(3, 4 + 1))
    elif interval == 4:
        distance = np.random.choice(np.arange(5, 20 + 1))
    elif interval == 5:
        distance = np.random.choice(np.arange(21, 150 + 1))
    if (video_frames.shape[0] - distance - length) < 0:
        raise ValueError('Length of video is less than 150 frames')
    frame_index = np.random.randint(0, video_frames.shape[0] - distance - length)
    joint_index = frame_index + distance

    frame = video_frames[frame_index]
    joint = joints[joint_index:joint_index + length] 
    label[interval] = 1.

    [normalized_frame, joint_encoded] = encode_action_and_images(
                data_features_to_extract='image_n_vec_xyz_aaxyz_nsc_q_dq_gripper',
                poses=joint, action_labels=None,
                init_images=frame, current_images=None)
    return (normalized_frame, joint_encoded, interval)

class CostarBlockStackingDataset(Dataset):
    def __init__(self, list_example_filenames,
                 label_features_to_extract=None, data_features_to_extract=None,
                 total_actions_available=41,
                 seed=0, random_state=None,
                 is_training=True, random_augmentation=None,
                 random_shift=False,
                 output_shape=None,
                 blend_previous_goal_images=False,
                 num_images_per_example=200, verbose=0, inference_mode=False, one_hot_encoding=True,
                 pose_name='pose_gripper_center',
                 force_random_training_pose_augmentation=None,
                 visual_mode=False,
                 # TODO(ahundt) make single_batch_cube default to false, fix all code deps and bugs first
                 single_batch_cube=False):
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
            'image_m_image_n' a giant NCHW cuboid of images 
            'image_n_vec_xyz_aaxyz_nsc_q_dq_gripper' another giant NCHW cuboid of images, encoded pose data, 6 joint angles of the UR5, change in joint angle and state of the gripper.
        random_augmentation: None or a float value between 0 and 1 indiciating how frequently random augmentation should be applied.
        num_images_per_example: The number of images in each example varies, so we simply sample in proportion to an estimated number
            of images per example. Set this number high if you want to visit more images in the same example.
            The data loader will visit each example `num_images_per_example` times, where the default is 200.
            Due to random sampling, there is no guarantee that every image will be visited once!
            The images can also be visited in a fixed order, particularly when is_training=False.
        one_hot_encoding flag triggers one hot encoding and thus numbers at the end of labels might not correspond to the actual size.
        force_random_training_pose_augmentation: override random_augmenation when training for pose data only.
        pose_name: Which pose to use as the robot 3D position in space. Options include:
            'pose' is the end effector ee_link pose at the tip of the connector
                of the robot, which is the base of the gripper wrist.
            'pose_gripper_center' is a point in between the robotiq C type gripping plates when the gripper is open
                with the same orientation as pose.
        visual_mode:To visualize features, set visual_mode to True. 
                    For feature_mode = 'time_difference_images', returns a sample from a numpy array of frames of shape - (No. of frames, No. of channels, height, width)
                    For feature_mode = 'cross_modal_embeddings', returns a sample from a numpy array of frames of shape - (No. of frames, No. of channels, height, width) and 
                                                                                       a numpy array of joint_vectors of shape - (No. of frames, 1, length, 20)     
        single_batch_cube: False will keep images and vector data in a tuple.
           True will tile the vector data to be the same shape as the image data, and concatenate it all into a cube of data.

        # Explanation of abbreviations:

        aaxyz_nsc: is an axis and angle in xyz order, where the angle is defined by a normalized sin(theta) cos(theta).
        nxygrid: at each pixel, concatenate two additional channels containing the pixel coordinate x and y as values between 0 and 1.
            This is similar to uber's "coordconv" paper.
        '''
        if random_state is None:
            random_state = RandomState(seed)
        # self.batch_size = batch_size

        # HACK(rexxarchl): fix bad file paths in standard txt files on the Archive
        if '/.keras/dataset/' in list_example_filenames[0]:
            for i, f in enumerate(list_example_filenames):
                path = f.split('/')
                path[:3] = ['~', '.keras', 'datasets']
                list_example_filenames[i] = os.path.join(*path)

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
        self.single_batch_cube = single_batch_cube

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
        self.num_images_per_example = num_images_per_example
        if self.inference_mode is True:
            self.list_example_filenames = inference_mode_gen(self.list_example_filenames)
        self.visual_mode = visual_mode
        if self.visual_mode:
            self.dataset = self.__data_generation(self.list_example_filenames, self.infer_index)
            
    @classmethod
    def from_standard_txt(cls, root, version, set_name, subset_name, split, feature_mode=None,
                          total_actions_available=41,
                          seed=0, random_state=None,
                          is_training=True, random_augmentation=None,
                          random_shift=False,
                          output_shape=None,
                          blend_previous_goal_images=False,
                          num_images_per_example=200, verbose=0, inference_mode=False, one_hot_encoding=True,
                          pose_name='pose_gripper_center',
                          force_random_training_pose_augmentation=None,
                          visual_mode=False,
                          # TODO(ahundt) make single_batch_cube default to false, fix all code deps and bugs first
                          single_batch_cube=False):
        '''
        Loads the filenames from specified set, subset and split from the standard txt files.
        Since CoSTAR BSD v0.4, the names for the .txt files that stores filenames for train/val/test splits are in standardized.
        For example, if you want to train the network in v0.4 blocks_only set, success_only subset, the txt file is called
        "costar_block_stacking_dataset_v0.4_blocks_only_success_only_train_files.txt".
        This function opens the standard txt files and create the Dataset object using the filenames in the txt file.
        The returned Dataset object can later be used in a DataLoader of choice to create a queue for train/val/test purposes.

        The following parameters are specific to this function. Other parameters follow that of the class constructor. See the docstring
        for __init__ for details.
        :param root: The root directory for the costar dataset.
        :param version: The CoSTAR Dataset version, as is used in the filename txt files.
        :param set_name: The set that will be loaded. Currently one of {'blocks_only', 'blocks_with_plush_toy'}.
        :param subset_name: The subset that will be loaded.
                            Currently one of {'success_only', 'error_failure_only', 'task_failure_only', 'task_and_error_failure'}.
        :param split: The split that will be loaded. One of {'train', 'test', 'val'}
        :param feature_mode: One of {'translation_only', 'rotation_only','stacking_reward', 'all_features', 'time_difference_images', 'cross_modal_embeddings'}. Correspond to different
                             feature combos that the returned data will have. If leave blank, will default to 'all_features'
                             Feature combo and their corresponding data and label features:
                             - 'all_features': data = 'image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17', label = grasp_goal_xyz_aaxyz_nsc_8'
                             - 'translation_only': data = 'image_0_image_n_vec_xyz_nxygrid_12', label = 'grasp_goal_xyz_3'
                             - 'rotation_only': data = 'image_0_image_n_vec_xyz_aaxyz_nsc_15', label = 'grasp_goal_aaxyz_nsc_5'
                             - 'stacking_reward': data = 'image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25', label = 'stacking_reward'
                             - 'time_difference_images': data = 'image_m_image_n', label = 'time_intervals'
                             - 'cross_modal_embeddings': data = 'image_n_vec_xyz_aaxyz_nsc_q_dq_gripper', label = 'cross_modal_time_intervals'
                             See the docstring for __init__ for details on data_features_to_extract and label_features_to_extract.
        :return: The class object for CostarBlockStackingDataset that can be fed into a DataLoader of choice to get train/test/val queues.
        '''
        if set_name not in COSTAR_SET_NAMES:
            raise ValueError("CostarBlockStackingDataset: Specify costar_set_name as one of {'blocks_only', 'blocks_with_plush_toy'}")
        if subset_name not in COSTAR_SUBSET_NAMES:
            raise ValueError("CostarBlockStackingDataset: Specify costar_subset_name as one of "
                             "{'success_only', 'error_failure_only', 'task_failure_only', 'task_and_error_failure'}")

        txt_filename = 'costar_block_stacking_dataset_{0}_{1}_{2}_{3}_files.txt'.format(version, set_name, subset_name, split)
        txt_filename = os.path.expanduser(os.path.join(root, set_name, txt_filename))
        if verbose > 0:
            print("Loading {0} filenames from txt files: \n\t{1}".format(split, txt_filename))

        with open(txt_filename, 'r') as f:
            data_filenames = f.read().splitlines()

        if feature_mode not in COSTAR_FEATURE_MODES:
            if verbose > 0:
                print("Using feature mode: " + feature_mode)
                if feature_mode != 'all_features':
                    print("Unknown feature mode: {}".format(feature_mode))
                    print("Using the original input block as the features")
            data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17']
            label_features = ['grasp_goal_xyz_aaxyz_nsc_8']
        else:
            if feature_mode == 'translation_only':
                data_features = ['image_0_image_n_vec_xyz_nxygrid_12']
                label_features = ['grasp_goal_xyz_3']
            elif feature_mode == 'rotation_only':
                data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_15']
                label_features = ['grasp_goal_aaxyz_nsc_5']
            elif feature_mode == 'stacking_reward':
                data_features = ['image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25']
                label_features = ['stacking_reward']
            elif feature_mode == 'time_difference_images':
                data_features = ['image_m_image_n']
                label_features = ['time_intervals']
            elif feature_mode == 'cross_modal_embeddings':
                data_features = ['image_n_vec_xyz_aaxyz_nsc_q_dq_gripper']
                label_features = ['cross_modal_time_intervals']

        if visual_mode:
            # Create separate dataset for each video listed in the file.
            data = [cls(
            data_filename,
            label_features_to_extract=label_features, data_features_to_extract=data_features,
            total_actions_available=total_actions_available,
            seed=seed, random_state=random_state,
            is_training=is_training, random_augmentation=random_augmentation,
            random_shift=random_shift,
            output_shape=output_shape,
            blend_previous_goal_images=blend_previous_goal_images,
            num_images_per_example=num_images_per_example,
            verbose=verbose, inference_mode=inference_mode, one_hot_encoding=one_hot_encoding,
            pose_name=pose_name,
            force_random_training_pose_augmentation=force_random_training_pose_augmentation,
            visual_mode=visual_mode,
            single_batch_cube=single_batch_cube) for data_filename in data_filenames]
        else:
            data = cls(
                data_filenames,
                label_features_to_extract=label_features, data_features_to_extract=data_features,
                total_actions_available=total_actions_available,
                seed=seed, random_state=random_state,
                is_training=is_training, random_augmentation=random_augmentation,
                random_shift=random_shift,
                output_shape=output_shape,
                blend_previous_goal_images=blend_previous_goal_images,
                num_images_per_example=num_images_per_example,
                verbose=verbose, inference_mode=inference_mode, one_hot_encoding=one_hot_encoding,
                pose_name=pose_name,
                force_random_training_pose_augmentation=force_random_training_pose_augmentation,
                visual_mode=visual_mode,
                single_batch_cube=single_batch_cube)

        return data

    def __len__(self):
        """Return the lenth of file names
        """
        if self.visual_mode:
            return len(self.dataset[0])
        return len(self.list_example_filenames) * self.num_images_per_example

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
        if self.data_features_to_extract == ['image_m_image_n'] or self.data_features_to_extract == ['image_n_vec_xyz_aaxyz_nsc_q_dq_gripper']:
            if self.visual_mode:
                # Return a sample from stack of frames and joints.
                frame_stack, joints_stack = self.dataset
                if joints_stack is not None:
                    [normalized_frame, joint_encoded] = encode_action_and_images(
                                                            data_features_to_extract='image_n_vec_xyz_aaxyz_nsc_q_dq_gripper',
                                                            poses=joints_stack[index].squeeze(), action_labels=None,
                                                            init_images=frame_stack[index], current_images=None)
                    return (normalized_frame,joint_encoded)
                return preprocess_numpy_input(np.array(frame_stack[index], dtype=np.double))
            else:
                X1, X2, y = self.__data_generation(self.list_example_filenames[index//self.num_images_per_example], self.infer_index)
                return X1, X2,y
        else:
            X, y = self.__data_generation(self.list_example_filenames[index//self.num_images_per_example], self.infer_index)
            
            return X, y

    def get_num_images_per_example(self):
        """ Get the estimated images per example.
        Note that the data loader already visit samples this many times.
        Run extra steps in proportion to this if you want to visit even more images per sample.
        """
        return self.num_images_per_example

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

        warnings.simplefilter('ignore')  # Ignore skimage warnings on anti-aliasing
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
                        raise ValueError('CostarBlockStackingDataset: You need to run preprocessing before this will work! \n' +
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

                    if (self.data_features_to_extract is not None and ('image_m_image_n' in self.data_features_to_extract or 
                                                                       'image_n_vec_xyz_aaxyz_nsc_q_dq_gripper' in self.data_features_to_extract)):
                        rgb_images = data['image']
                        rgb_images = ConvertImageListToNumpy(rgb_images, format='numpy', data_format='NCHW')
                    else: 
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

                    # create vector of pose, q and dq to form the joint embedding
                    if(self.data_features_to_extract is not None and 'image_n_vec_xyz_aaxyz_nsc_q_dq_gripper' in self.data_features_to_extract):
                        gripper_states = np.array(data['gripper'])
                        gripper_states = np.expand_dims(gripper_states,1)
                        joints = np.hstack((np.array(data[self.pose_name]), np.array(data['q']),np.array(data['dq']),gripper_states)) 

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
            # If features to extract is image_m_image_n then return batch (two images and label which is the interval)           
            if (self.data_features_to_extract is not None and 'image_m_image_n' in self.data_features_to_extract):
                if(self.visual_mode):
                    return (np.array(rgb_images_resized), None)
                batch = generate_temporal_distance_training_data(np.array(rgb_images_resized))
                return batch

            if (self.data_features_to_extract is not None and 'image_n_vec_xyz_aaxyz_nsc_q_dq_gripper' in self.data_features_to_extract): 
                if(self.visual_mode):
                    num_joints_concatenated = 10        #TODO Do we want to include this as one of the private variables and then set it during the creation of the class object? 
                    rgb_images_resized_trunc = np.array(rgb_images_resized)[:-num_joints_concatenated]
                    joints_concatenated = []           
                    for i in range(rgb_images_resized_trunc.shape[0]):
                        joints_concatenated.append(np.array(joints[i:i + num_joints_concatenated]))
                    joints_concatenated = np.stack(joints_concatenated)
                    joints_concatenated = joints_concatenated[:,np.newaxis] 
                    return (rgb_images_resized_trunc,joints_concatenated)
                batch = generate_crossmodal_training_data(np.array(rgb_images_resized),joints)                
                return batch

            action_labels = np.array(action_labels)
            # note: we disabled keras format of numpy arrays
            # TODO(ahundt) make sure this doen't happen wrong, see dataset reader collate and prefetch
            init_images = np.array(init_images, dtype=np.float32)
            # print(init_images.shape)  # (1, 224, 224, 3)
            init_images = np.moveaxis(init_images, 3, 1)  # Switch to channel-first format as per torch convention
            # print(init_images.shape)  # (1, 3, 224, 224)

            # note: we disabled keras format of numpy arrays
            # TODO(ahundt) make sure this doen't happen wrong, see dataset reader collate and prefetch
            current_images = np.array(current_images, dtype=np.float32)
            current_images = np.moveaxis(current_images, 3, 1)  # Switch to channel-first format as per torch convention
        
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
                y=y, random_augmentation=self.random_encoding_augmentation,
                single_batch_cube=self.single_batch_cube)

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
            # HACK(rexxarchl): tf process the data in batches, while torch process one-by-one.
            #                  Therefore, while tf outputs (batch, 224, 224, 57) tensors, torch will output (batch, 1, 224, 224, 57)
            #                  Use squeeze to eliminate the redundant dimensions with depth of 1.
            if isinstance(X, list):
                X = [np.squeeze(X[i]) for i in range(len(X))]
            else:
                X = np.squeeze(X)

            # TODO(ahundt) make sure this doen't happen wrong, see dataset reader collate and prefetch
            if isinstance(y, list):
                y = [np.squeeze(y[i]) for i in range(len(y))]
            else:
                y = np.squeeze(y)

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


# def collate_cube(batch):
#     '''
#     Collate function for when the data output of the Dataset object is not a mega cube.

#     When single_batch_cube=True, the output shape of the Dataset object will be, for example, (57, 224, 224)
#     Then, the user can use the default collate function of pyTorch DataLoader to form a batch of batch_size (B) with
#      shape (B, 57, 224, 224)

#     However, it's more efficient to do the mega cube formation in batch using numpy.
#     This is also what the loader does in the Tensorflow version of this code.
#     When single_batch_cube=False, the output of the Dataset object will be a list of length B, with each object in
#      this list being a list of (img_0, img_n, vector).
#     This function would convert the images and vector into np stacks, and process the stacks into a mega cube.

#     Note that, since it is impossible to determine what the data_features_to_extract argument was for the Dataset object,
#      unit mesh grid will always be added to the megacube. 
#      The channel for the output data may increase 2 compared to when single_batch_cube=True as a result.
#     '''
#     data, targets = zip(*batch)

#     # data is a list of length batch_size, and each element in this list is a list of (img_0, img_n, vector)
#     image_0 = np.array([img[0] for img in data])
#     image_n = np.array([img[1] for img in data])
#     vector = np.array([img[2] for img in data])

#     data = concat_images_with_tiled_vector_np([image_0, image_n], vector)
#     data = concat_unit_meshgrid_np(data)

#     return torch.tensor(data), torch.tensor(targets)


if __name__ == "__main__":
    from tqdm import tqdm
    visualize = True
    single_batch_cube = False
    output_shape = (224, 224, 3)

    # filenames = glob.glob(os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.4/blocks_only/*success.h5f'))
    # costar_dataset = CostarBlockStackingDataset(
    #     filenames, verbose=0,
    #     output_shape=output_shape,
    #     label_features_to_extract='grasp_goal_xyz_aaxyz_nsc_8',
    #     # data_features_to_extract=['current_xyz_aaxyz_nsc_8'],
    #     data_features_to_extract=['image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17'],
    #     blend_previous_goal_images=False, inference_mode=False, num_images_per_example=1, single_batch_cube=single_batch_cube)

    costar_dataset = CostarBlockStackingDataset.from_standard_txt(
                      root='~/.keras/datasets/costar_block_stacking_dataset_v0.4/',
                      version='v0.4', set_name='blocks_only', subset_name='success_only',
                      split='train', feature_mode='all_features', output_shape=(224, 224, 3),
                      num_images_per_example=1, is_training=False, single_batch_cube=single_batch_cube)

    # generator = DataLoader(costar_dataset, batch_size=64, shuffle=False, num_workers=4)
    generator = DataLoader(costar_dataset, batch_size=32, shuffle=False, num_workers=1)

    print("Length of the dataset: {}. Length of the loader: {}.".format(len(costar_dataset), len(generator)))

    generator_output = iter(generator)
    print("-------------------op")
    # x, y = generator_output
    x, y = next(generator_output)
    # print(len(x))
    # print(x.shape)
    if single_batch_cube:
        print(x.shape)
    else:
        print(x[0].shape, x[1].shape, x[2].shape)
    print(y.shape)

    pb = tqdm(range(len(generator)-1))
    for i in pb:
        pb.set_description('batch: {}'.format(i))

        x, y = generator_output.next()
        y = y.numpy()
        x = [t.numpy() for t in x] if not single_batch_cube else x.numpy()

        if visualize:
            import matplotlib
            import matplotlib.pyplot as plt
            if single_batch_cube:
                clear_view_img = np.moveaxis(x[0, :3, :, :], 0, 2)
                current_img = np.moveaxis(x[0, 3:6, :, :], 0, 2)
            else:
                clear_view_img = np.moveaxis(x[0][0], 0, 2)
                current_img = np.moveaxis(x[1][0], 0, 2)

            # clear view image
            plt.imshow(clear_view_img / 2.0 + 0.5)
            plt.draw()
            plt.pause(0.25)
            # current timestep image
            plt.imshow(current_img / 2.0 + 0.5)
            plt.draw()
            plt.pause(0.25)
            # uncomment the following line to wait for one window to be closed before showing the next
            # plt.show()

        if single_batch_cube:
            assert np.all(x[:, :3, :, :] <= 1) and np.all(x[:, :3, :, :] >= -1), "x[0] is not within range!"
            assert np.all(x[:, 3:6, :, :] <= 1) and np.all(x[:, 3:6, :, :] >= -1), "x[1] is not within range!"
            assert np.all(x[:, 6:, :, :] <= 1) and np.all(x[:, 6:, :, :] >= 0), "x[2] is not within range!"
            assert np.all(y <= 1) and np.all(y >= 0), "y is not within range!"
        else:
            assert np.all(x[0] <= 1) and np.all(x[0] >= -1), "x[0] is not within range!"
            assert np.all(x[1] <= 1) and np.all(x[1] >= -1), "x[1] is not within range!"
            assert np.all(x[2] <= 1) and np.all(x[2] >= 0), "x[2] is not within range!"
            assert np.all(y <= 1) and np.all(y >= 0), "y is not within range!"
