"""
Example script demonstrating how to use the PointCloudFeatureEncoder.

This script shows:
1. How to instantiate the PointCloudFeatureEncoder
2. How to load pretrained weights (optional)
3. How to extract features from point cloud data

Usage:
    # Without pretrained weights (random initialization)
    python examples/feature_encoder_example.py
    
    # With pretrained weights
    python examples/feature_encoder_example.py --ckpt path/to/pretrain.pth
"""

import torch
import argparse
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Feature Encoder Example')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to pretrained checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for input point clouds')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points in each point cloud')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Output feature dimension')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Point Cloud Feature Encoder Example")
    print("=" * 80)
    
    # Create the feature encoder
    print("\n1. Creating PointCloudFeatureEncoder...")
    encoder = PointCloudFeatureEncoder(
        num_group=64,        # Number of point groups
        group_size=32,       # Points per group
        trans_dim=384,       # Transformer dimension
        depth=12,            # Number of transformer layers
        num_heads=6,         # Number of attention heads
        encoder_dims=384,    # Encoder output dimension
        output_dim=args.output_dim,  # Final output dimension
    )
    print(f"   ✓ Encoder created with output dimension: {args.output_dim}")
    
    # Load pretrained weights if provided
    if args.ckpt is not None:
        print(f"\n2. Loading pretrained weights from: {args.ckpt}")
        encoder.load_pretrained_weights(args.ckpt)
        print("   ✓ Pretrained weights loaded successfully")
    else:
        print("\n2. No pretrained weights provided, using random initialization")
    
    # Set to evaluation mode
    encoder.eval()
    
    # Create sample input
    print(f"\n3. Creating sample input point cloud...")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Number of points: {args.num_points}")
    print(f"   - Input shape: [{args.batch_size}, {args.num_points}, 3]")
    
    # Generate random point cloud data (B, N, 3)
    with torch.no_grad():
        input_pts = torch.randn(args.batch_size, args.num_points, 3)
        
        # Normalize points to unit sphere (optional but recommended)
        input_pts = input_pts / torch.norm(input_pts, dim=2, keepdim=True)
        
        print("   ✓ Sample point cloud created")
        
        # Extract features
        print(f"\n4. Extracting features...")
        features = encoder(input_pts)
        
        print(f"   ✓ Features extracted successfully")
        print(f"   - Output shape: {list(features.shape)}")
        print(f"   - Expected shape: [{args.batch_size}, {args.output_dim}]")
        print(f"   - Feature range: [{features.min().item():.4f}, {features.max().item():.4f}]")
        print(f"   - Feature mean: {features.mean().item():.4f}")
        print(f"   - Feature std: {features.std().item():.4f}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    
    # Additional information
    print("\nUsage in your code:")
    print("-" * 80)
    print("from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder")
    print("")
    print("# Create encoder")
    print("encoder = PointCloudFeatureEncoder(output_dim=256)")
    print("")
    print("# Optionally load pretrained weights")
    print("encoder.load_pretrained_weights('path/to/pretrain.pth')")
    print("")
    print("# Extract features from point cloud (B, N, 3)")
    print("features = encoder(point_cloud)  # Output: (B, 256)")
    print("-" * 80)


if __name__ == '__main__':
    main()
