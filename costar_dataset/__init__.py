"""costar_dataset - Code for the The CoSTAR Block Stacking Dataset  includes a real robot trying to stack colored children's blocks. https://sites.google.com/site/costardataset"""

__version__ = '0.4.0'
__author__ = 'Andrew Hundt <ATHundt@gmail.com>'

try:
    from costar_dataset.block_stacking_reader_torch import CostarBlockStackingDataset
    from costar_dataset.hypertree_pose_metrics_torch import cart_error, angle_error, grasp_acc_in_bins_batch
except ImportError:
    print("Torch not installed.")

__all__ = [
    '__version__',

    # Classes
    'CostarBlockStackingDataset',

    # API
    'cart_error',
    'angle_error',
    'grasp_acc_in_bins_batch'
]
