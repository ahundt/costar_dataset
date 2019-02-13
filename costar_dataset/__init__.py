"""costar_dataset - Code for the The CoSTAR Block Stacking Dataset  includes a real robot trying to stack colored children's blocks. https://sites.google.com/site/costardataset"""

__version__ = '0.4.0'
__author__ = 'Andrew Hundt <ATHundt@gmail.com>'

try:
    from costar_dataset.block_stacking_reader_torch import CostarBlockStackingDataset
    from costar_dataset.hypertree_pose_metrics_torch import absolute_cart_distance_xyz_aaxyz_nsc_batch, absolute_angle_distance_xyz_aaxyz_nsc_batch
except ImportError:
    print("Torch not installed.")

__all__ = [
    '__version__',

    # Classes
    'CostarBlockStackingDataset',

    # API
    'absolute_cart_distance_xyz_aaxyz_nsc_batch',
    'absolute_angle_distance_xyz_aaxyz_nsc_batch'
]
