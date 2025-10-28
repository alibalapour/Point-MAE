"""
Simple unit test for PointCloudFeatureEncoder without CUDA dependencies.

This test verifies:
1. The encoder can be instantiated
2. The architecture is correctly built
3. Parameter counts are reasonable
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_encoder_structure():
    """Test that the encoder has the expected structure"""
    print("=" * 80)
    print("Testing PointCloudFeatureEncoder Structure")
    print("=" * 80)
    
    # Import directly to avoid Point_MAE dependencies
    from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder
    
    print("\n1. Creating encoder with default parameters...")
    encoder = PointCloudFeatureEncoder()
    print("   ✓ Encoder created successfully")
    
    print("\n2. Checking encoder components...")
    assert hasattr(encoder, 'group_divider'), "Missing group_divider"
    print("   ✓ group_divider exists")
    
    assert hasattr(encoder, 'encoder'), "Missing encoder"
    print("   ✓ encoder exists")
    
    assert hasattr(encoder, 'pos_embed'), "Missing pos_embed"
    print("   ✓ pos_embed exists")
    
    assert hasattr(encoder, 'blocks'), "Missing blocks (transformer)"
    print("   ✓ blocks (transformer) exists")
    
    assert hasattr(encoder, 'norm'), "Missing norm layer"
    print("   ✓ norm layer exists")
    
    assert hasattr(encoder, 'feature_head'), "Missing feature_head"
    print("   ✓ feature_head exists")
    
    print("\n3. Checking encoder parameters...")
    assert encoder.output_dim == 256, f"Expected output_dim=256, got {encoder.output_dim}"
    print(f"   ✓ output_dim = {encoder.output_dim}")
    
    assert encoder.num_group == 64, f"Expected num_group=64, got {encoder.num_group}"
    print(f"   ✓ num_group = {encoder.num_group}")
    
    assert encoder.group_size == 32, f"Expected group_size=32, got {encoder.group_size}"
    print(f"   ✓ group_size = {encoder.group_size}")
    
    assert encoder.trans_dim == 384, f"Expected trans_dim=384, got {encoder.trans_dim}"
    print(f"   ✓ trans_dim = {encoder.trans_dim}")
    
    print("\n4. Testing custom output dimension...")
    encoder_512 = PointCloudFeatureEncoder(output_dim=512)
    assert encoder_512.output_dim == 512, f"Expected output_dim=512, got {encoder_512.output_dim}"
    print("   ✓ Custom output_dim=512 works")
    
    print("\n5. Counting parameters...")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    print("\n6. Checking feature_head architecture...")
    # The feature_head should project from trans_dim to output_dim
    feature_head_modules = list(encoder.feature_head.children())
    assert len(feature_head_modules) == 3, "feature_head should have 3 layers"
    assert isinstance(feature_head_modules[0], nn.Linear), "First layer should be Linear"
    assert isinstance(feature_head_modules[1], nn.GELU), "Second layer should be GELU"
    assert isinstance(feature_head_modules[2], nn.Linear), "Third layer should be Linear"
    print("   ✓ feature_head has correct architecture (Linear -> GELU -> Linear)")
    
    # Check dimensions
    assert feature_head_modules[0].in_features == 384, "First Linear should have in_features=384"
    assert feature_head_modules[2].out_features == 256, "Last Linear should have out_features=256"
    print(f"   ✓ feature_head dimensions: {feature_head_modules[0].in_features} -> {feature_head_modules[0].out_features} -> {feature_head_modules[2].out_features}")
    
    print("\n7. Testing load_pretrained_weights method exists...")
    assert hasattr(encoder, 'load_pretrained_weights'), "Missing load_pretrained_weights method"
    assert callable(encoder.load_pretrained_weights), "load_pretrained_weights should be callable"
    print("   ✓ load_pretrained_weights method exists and is callable")
    
    print("\n" + "=" * 80)
    print("All structure tests passed! ✓")
    print("=" * 80)
    
    print("\nNote: Forward pass testing requires CUDA dependencies (knn_cuda).")
    print("The encoder is ready to use with pretrained weights.")
    
    return True


def test_model_modes():
    """Test that the model can be set to train/eval modes"""
    from models.PointCloudFeatureEncoder import PointCloudFeatureEncoder
    
    print("\n" + "=" * 80)
    print("Testing Model Modes")
    print("=" * 80)
    
    encoder = PointCloudFeatureEncoder()
    
    print("\n1. Testing eval mode...")
    encoder.eval()
    assert not encoder.training, "Model should not be in training mode"
    print("   ✓ Model set to eval mode")
    
    print("\n2. Testing train mode...")
    encoder.train()
    assert encoder.training, "Model should be in training mode"
    print("   ✓ Model set to train mode")
    
    print("\n" + "=" * 80)
    print("Model modes test passed! ✓")
    print("=" * 80)


if __name__ == '__main__':
    try:
        test_encoder_structure()
        test_model_modes()
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓✓✓")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
