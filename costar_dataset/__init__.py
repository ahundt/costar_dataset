"""costar_dataset - Code for the The CoSTAR Block Stacking Dataset includes a real robot trying to stack colored children's blocks. https://sites.google.com/site/costardataset"""

__version__ = '0.4.0.1'
__author__ = 'Andrew Hundt <ATHundt@gmail.com>'

_backend = None

try:
    import torch
    _backend = 'torch'
except ImportError:
    print("PyTorch is not installed. Skipping torch related imports")

try:
    import tensorflow as tf
    _backend = 'tf'
except ImportError:
    print("Tensorflow is not installed. Skipping tf related imports")

if not _backend:
    raise ImportError("Neither PyTorch or Tensorflow is found! "
                      "Please install one of the backend to use this package.")

__all__ = [
    '__version__',

    # API
    'cart_error',
    'angle_error',
    'grasp_acc_in_bins_batch'
]

if _backend == 'torch':
    from costar_dataset.block_stacking_reader_torch import CostarBlockStackingDataset
    from costar_dataset.hypertree_pose_metrics_torch import cart_error, angle_error
    __all__ += ['CostarBlockStackingDataset']  # export class name
else:
    from costar_dataset.block_stacking_reader_tf import CostarBlockStackingSequence
    from costar_dataset.hypertree_pose_metrics_tf import cart_error, angle_error
    __all__ += ['CostarBlockStackingSequence']  # export class name

