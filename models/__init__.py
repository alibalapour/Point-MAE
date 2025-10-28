from .build import build_model_from_cfg
from .PointCloudFeatureEncoder import PointCloudFeatureEncoder

# Delay import of Point_MAE to avoid CUDA dependency issues when only using PointCloudFeatureEncoder
def _lazy_import_point_mae():
    import models.Point_MAE

try:
    _lazy_import_point_mae()
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import Point_MAE module: {e}. PointCloudFeatureEncoder is still available.")