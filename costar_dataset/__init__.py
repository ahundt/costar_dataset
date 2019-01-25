"""costar_dataset - Code for the The CoSTAR Block Stacking Dataset  includes a real robot trying to stack colored children's blocks. https://sites.google.com/site/costardataset"""

__version__ = '0.4.0'
__author__ = 'Andrew Hundt <ATHundt@gmail.com>'

from costar_dataset.block_stacking_reader_torch import CostarBlockStackingDataset

__all__ = [
    '__version__',
    'CostarBlockStackingDataset'
]
