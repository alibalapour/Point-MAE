# Point Cloud Feature Encoder

This module provides a feature encoder for point cloud data that can leverage pretrained Point-MAE weights.

## Overview

The `PointCloudFeatureEncoder` is designed to extract fixed-size feature vectors from variable-size point clouds. It takes point cloud input of shape `[B, N, 3]` and produces feature vectors of shape `[B, 256]` (or any specified output dimension).

## Key Features

- **Pretrained weights support**: Can load pretrained weights from Point-MAE models
- **Flexible architecture**: Configurable transformer depth, attention heads, and output dimensions
- **Efficient processing**: Uses FPS (Farthest Point Sampling) and KNN for point grouping
- **Standard input/output**: Takes `B×N×3` point clouds and outputs `B×256` features

## Quick Start

### Basic Usage

```python
from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder
import torch

# Create the encoder
encoder = PointCloudFeatureEncoder(output_dim=256)

# Create sample point cloud (batch_size=4, num_points=1024, xyz=3)
point_cloud = torch.randn(4, 1024, 3)

# Extract features
features = encoder(point_cloud)  # Output shape: [4, 256]
```

### Using Pretrained Weights

```python
from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder

# Create encoder
encoder = PointCloudFeatureEncoder(output_dim=256)

# Load pretrained weights from Point-MAE
encoder.load_pretrained_weights('path/to/pretrain.pth')

# Extract features
features = encoder(point_cloud)
```

### Running the Example

```bash
# Without pretrained weights
python examples/feature_encoder_example.py

# With pretrained weights
python examples/feature_encoder_example.py --ckpt path/to/pretrain.pth

# Custom configuration
python examples/feature_encoder_example.py --batch_size 8 --num_points 2048 --output_dim 512
```

## Architecture Details

The encoder consists of:

1. **Point Grouping**: Uses Farthest Point Sampling (FPS) and K-Nearest Neighbors (KNN) to divide the point cloud into local groups
2. **Local Feature Encoding**: Each group is encoded using a PointNet-style encoder
3. **Position Embedding**: 3D coordinates are embedded into high-dimensional space
4. **Transformer Encoder**: Self-attention blocks process the group features
5. **Global Pooling**: Max pooling aggregates features across all groups
6. **Feature Projection**: Final linear layers project to the desired output dimension

## Parameters

- `num_group` (int): Number of groups to divide the point cloud into. Default: 64
- `group_size` (int): Number of points per group. Default: 32
- `trans_dim` (int): Transformer dimension. Default: 384
- `depth` (int): Number of transformer blocks. Default: 12
- `num_heads` (int): Number of attention heads. Default: 6
- `encoder_dims` (int): Encoder output dimension. Default: 384
- `output_dim` (int): Final output feature dimension. Default: 256

## Downloading Pretrained Weights

Pretrained Point-MAE weights can be downloaded from the official repository:

- **Pretrained on ShapeNet**: [pretrain.pth](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth)
- **Fine-tuned on ModelNet40**: [modelnet_1k.pth](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_1k.pth)
- **Fine-tuned on ScanObjectNN**: [scan_hardest.pth](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_hardest.pth)

## Example Use Cases

### 1. Point Cloud Classification

```python
import torch.nn as nn
from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder

class PointCloudClassifier(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.encoder = PointCloudFeatureEncoder(output_dim=256)
        self.encoder.load_pretrained_weights('pretrain.pth')
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, point_cloud):
        features = self.encoder(point_cloud)
        return self.classifier(features)
```

### 2. Point Cloud Retrieval

```python
from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder
import torch.nn.functional as F

# Extract features for database and query
encoder = PointCloudFeatureEncoder(output_dim=256)
encoder.load_pretrained_weights('pretrain.pth')
encoder.eval()

with torch.no_grad():
    db_features = encoder(database_point_clouds)  # [N, 256]
    query_features = encoder(query_point_clouds)  # [M, 256]
    
    # Compute similarity
    similarity = F.cosine_similarity(
        query_features.unsqueeze(1), 
        db_features.unsqueeze(0), 
        dim=2
    )
```

### 3. Feature Extraction Pipeline

```python
from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder
import numpy as np

def extract_features_from_dataset(point_clouds, checkpoint_path=None):
    """
    Extract features from a dataset of point clouds.
    
    Args:
        point_clouds: NumPy array of shape [N, num_points, 3]
        checkpoint_path: Path to pretrained weights (optional)
    
    Returns:
        features: NumPy array of shape [N, 256]
    """
    encoder = PointCloudFeatureEncoder(output_dim=256)
    
    if checkpoint_path:
        encoder.load_pretrained_weights(checkpoint_path)
    
    encoder.eval()
    
    # Convert to tensor
    pts_tensor = torch.from_numpy(point_clouds).float()
    
    with torch.no_grad():
        features = encoder(pts_tensor)
    
    return features.numpy()
```

## Requirements

- PyTorch >= 1.7.0
- knn_cuda (for KNN operations)
- Other dependencies as listed in `requirements.txt`

## Citation

If you use this encoder with pretrained Point-MAE weights, please cite:

```bibtex
@inproceedings{pang2022masked,
  title={Masked autoencoders for point cloud self-supervised learning},
  author={Pang, Yatian and Wang, Wenxiao and Tay, Francis EH and Liu, Wei and Tian, Yonghong and Yuan, Li},
  booktitle={ECCV 2022},
  pages={604--621},
  year={2022},
  organization={Springer}
}
```
