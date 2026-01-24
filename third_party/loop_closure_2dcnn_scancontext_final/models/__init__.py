"""
模型模块
"""
# 导入时序模型
from .temporal_2d_cnn import Temporal2DCNN, Temporal2DCNNLite, Temporal2DCNNResNet
from .temporal_3d_cnn import Temporal3DCNN, Temporal3DCNNLite, Temporal3DCNNDeep

# 导入原有模型
from .sc_ring_cnn import SCRingCNN
from .sc_standard_cnn import SCStandardCNN, SCStandardCNNLite

__all__ = [
    # 时序模型
    'Temporal2DCNN',
    'Temporal2DCNNLite',
    'Temporal2DCNNResNet',
    'Temporal3DCNN',
    'Temporal3DCNNLite',
    'Temporal3DCNNDeep',
    # 原有模型
    'SCRingCNN',
    'SCStandardCNN',
    'SCStandardCNNLite'
]
from .sc_standard_spatial_cnn import SCStandardSpatialCNN, create_sc_standard_spatial_cnn
from .circular_conv import CircularPadConv2d, CircularResidualBlock

__all__ = [
    'SCRingCNN',
    'SCStandardCNN',
    'SCStandardCNNLite',
    'SCStandardSpatialCNN',
    'create_sc_standard_spatial_cnn',
    'CircularPadConv2d',
    'CircularResidualBlock'
]
