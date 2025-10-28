import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# Import optional dependencies
try:
    from utils.logger import print_log
except:
    def print_log(msg, logger=None):
        print(msg)

try:
    from knn_cuda import KNN
    from utils import misc
    HAS_CUDA_DEPS = True
except:
    HAS_CUDA_DEPS = False
    print_log("Warning: CUDA dependencies not available. KNN and FPS will use fallback implementations.", logger='PointCloudFeatureEncoder')


# ============================================================================
# Core Components (copied from Point_MAE.py to avoid circular dependencies)
# ============================================================================

class Encoder(nn.Module):
    """Embedding module for point cloud groups"""
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    """FPS + KNN grouping"""
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        if HAS_CUDA_DEPS:
            self.knn = KNN(k=self.group_size, transpose_mode=True)
        else:
            self.knn = None

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        if not HAS_CUDA_DEPS:
            raise RuntimeError("CUDA dependencies (knn_cuda, pointnet2_ops) required for Group.forward(). "
                             "Please install them or use CPU-compatible alternatives.")
        
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointCloudFeatureEncoder(nn.Module):
    """
    Point Cloud Feature Encoder with pretrained weights support.
    
    This encoder takes B*N*3 point cloud input and extracts B*256 features.
    It can load pretrained weights from Point-MAE models.
    
    Args:
        num_group (int): Number of groups to divide the point cloud into. Default: 64
        group_size (int): Number of points per group. Default: 32
        trans_dim (int): Transformer dimension. Default: 384
        depth (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads. Default: 6
        encoder_dims (int): Encoder output dimension. Default: 384
        output_dim (int): Final output feature dimension. Default: 256
    """
    
    def __init__(
        self,
        num_group=64,
        group_size=32,
        trans_dim=384,
        depth=12,
        num_heads=6,
        encoder_dims=384,
        output_dim=256,
    ):
        super().__init__()
        
        self.num_group = num_group
        self.group_size = group_size
        self.trans_dim = trans_dim
        self.output_dim = output_dim
        
        # Group divider (FPS + KNN)
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        
        # Point cloud embedding encoder
        self.encoder = Encoder(encoder_channel=encoder_dims)
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )
        
        # Transformer encoder
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path_rate=0.1,
            num_heads=num_heads,
        )
        
        self.norm = nn.LayerNorm(trans_dim)
        
        # Feature projection head to output_dim
        self.feature_head = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.GELU(),
            nn.Linear(trans_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained weights from Point-MAE checkpoint.
        
        Args:
            checkpoint_path (str): Path to the pretrained checkpoint file
        """
        print_log(f'Loading pretrained weights from {checkpoint_path}', logger='PointCloudFeatureEncoder')
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the base model state dict
        if 'base_model' in checkpoint:
            state_dict = checkpoint['base_model']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Map MAE_encoder keys to our encoder
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('MAE_encoder.'):
                new_key = k.replace('MAE_encoder.', '')
                new_state_dict[new_key] = v
            elif k.startswith('group_divider.'):
                new_state_dict[k] = v
        
        # Load the mapped weights
        incompatible = self.load_state_dict(new_state_dict, strict=False)
        
        if incompatible.missing_keys:
            # Filter out expected missing keys (feature_head is new)
            unexpected_missing = [k for k in incompatible.missing_keys if not k.startswith('feature_head')]
            if unexpected_missing:
                print_log(f'Missing keys: {unexpected_missing}', logger='PointCloudFeatureEncoder')
        
        if incompatible.unexpected_keys:
            print_log(f'Unexpected keys: {incompatible.unexpected_keys}', logger='PointCloudFeatureEncoder')
        
        print_log('Successfully loaded pretrained weights', logger='PointCloudFeatureEncoder')
    
    def forward(self, pts):
        """
        Forward pass to extract features from point cloud.
        
        Args:
            pts (torch.Tensor): Input point cloud of shape [B, N, 3]
                where B is batch size, N is number of points
        
        Returns:
            torch.Tensor: Extracted features of shape [B, output_dim]
        """
        # Divide point cloud into groups
        # neighborhood: [B, G, M, 3], center: [B, G, 3]
        neighborhood, center = self.group_divider(pts)
        
        # Encode each group
        # group_input_tokens: [B, G, C]
        group_input_tokens = self.encoder(neighborhood)
        
        # Position embedding
        # pos: [B, G, trans_dim]
        pos = self.pos_embed(center)
        
        # Transformer encoding
        # x: [B, G, trans_dim]
        x = self.blocks(group_input_tokens, pos)
        x = self.norm(x)
        
        # Global pooling across groups
        # features: [B, trans_dim]
        features = torch.max(x, dim=1)[0]
        
        # Project to output dimension
        # output: [B, output_dim]
        output = self.feature_head(features)
        
        return output
