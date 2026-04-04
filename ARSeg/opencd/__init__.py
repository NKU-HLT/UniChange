# Copyright (c) Open-CD. All rights reserved.
import mmcv
import mmdet
import mmengine
from mmengine.utils import digit_version

import mmseg
from .version import __version__, version_info

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.6.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

mmseg_minimum_version = '1.0.0rc6'
mmseg_maximum_version = '1.1.2'
mmseg_version = digit_version(mmseg.__version__)

mmdet_minimum_version = '3.0.0rc6'
mmdet_maximum_version = '3.1.0'
mmdet_version = digit_version(mmdet.__version__)

__all__ = ['__version__', 'version_info', 'digit_version']