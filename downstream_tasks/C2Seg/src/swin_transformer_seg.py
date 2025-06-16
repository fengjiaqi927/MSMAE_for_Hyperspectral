import torch
import torch.nn as nn
import math
from models_vit_tensor_CD_2 import FPNHEAD
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, to_2tuple


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


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


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, posmlp_dim=32, qkv_bias=True, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.posmlp = nn.Sequential(
            nn.Conv2d(2, posmlp_dim, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(posmlp_dim, num_heads, 1)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pos_hw, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B*nWindow, nHead, Wh*Ww, Wh*Ww)

        # pos_hw (B, H, W, 2); during finetuning, B == 1 to save computation and storage
        assert self.window_size[0] == self.window_size[1]
        pos_windows = window_partition(pos_hw, self.window_size[0])  # B*nWindow, window_size, window_size, 2
        pos_windows = pos_windows.permute(0, 3, 1, 2).flatten(2)  # B*nW, 2, Wh*Ww
        pos_input = pos_windows.unsqueeze(2) - pos_windows.unsqueeze(3) # B*nW, 2, Wh*Ww, Wh*Ww
        # log-spaced coords
        pos_input = torch.sign(pos_input) * torch.log(1. + pos_input.abs())
        relative_position_bias = self.posmlp(pos_input) # B*nW, nH, WW, WW

        if pos_hw.size(0) == 1: # for finetuning B == 1
            nW, nH, WW, WW = relative_position_bias.size()
            B = B_ // nW
            relative_position_bias = relative_position_bias.unsqueeze(0).expand(B, -1, -1, -1, -1)
            relative_position_bias = relative_position_bias.reshape(-1, nH, WW, WW)

        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, posmlp_dim=32,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, posmlp_dim=posmlp_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # pay attention
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.attn_mask = attn_mask
        self.downsample = downsample

    def forward(self, x, pos_hw): # pos_hw (B, H, W, 2)
        def _inner_forward(x, pos_hw):
            #H, W = self.input_resolution
            H, W = pos_hw.size(1), pos_hw.size(2)
            if self.downsample:
                x, (H, W), pos_hw = self.downsample(x, pos_hw)
            B, L, C = x.shape
            assert L == H * W, "input dimension wrong"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                pos_hw = torch.roll(pos_hw, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            if self.attn_mask is not None:
                self.attn_mask = self.attn_mask.to(x.device)
            attn_windows = self.attn(x_windows, pos_hw, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                pos_hw = torch.roll(pos_hw, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            x = x.view(B, H * W, C)

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x, pos_hw

        if self.with_cp and x.requires_grad:
            x, pos_hw = cp.checkpoint(_inner_forward, x, pos_hw)
        else:
            x, pos_hw = _inner_forward(x, pos_hw)

        return x, pos_hw


class PatchMerge(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.norm = norm_layer(in_chans)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, pos_hw): # pos_hw are absolute positions of h and w
        N, L, C = x.shape
        _, H, W, _ = pos_hw.shape
        assert L == H * W
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(N, C, H, W)
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        assert self.patch_size[0] == 2
        if self.patch_size[0] == 2:
            # shrink position values and scales
            pos_hw = pos_hw[:, 0::2, 0::2, :] / 2.
            
        return x, (H, W), pos_hw


class SwinTransformerSegmentation(nn.Module):
    """Swin Transformer for Segmentation"""
    
    def __init__(self, 
                 img_size=256, 
                 patch_size=4, 
                 in_chans=3,
                 num_classes=10,
                 embed_dim=96, 
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4, 
                 window_size=8,
                 posmlp_dim=32,
                 norm_layer=nn.LayerNorm,
                 dropout=0.5,
                 **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = len(depths)
        
        # --------------------------------------------------------------------------
        # Swin Encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer)
        patches_resolution = self.patch_embed.patches_resolution
        
        self.embed_h = self.embed_w = int(self.patch_embed.num_patches ** 0.5)
        self.patches_resolution = self.patch_embed.patches_resolution
        
        # Position encoding for Swin
        pos_h = torch.arange(0, self.embed_h)[None, None, :, None].repeat(1, 1, 1, self.embed_w).float()
        pos_w = torch.arange(0, self.embed_w)[None, None, None, :].repeat(1, 1, self.embed_h, 1).float()
        self.pos_hw = torch.cat((pos_h, pos_w), dim=1) #(1, 2, H, W)
        
        # Swin Transformer blocks
        self.blocks = nn.ModuleList()
        for i_layer in range(self.num_layers):
            for dep in range(depths[i_layer]):
                downsample_flag = (i_layer > 0) and (dep == 0)
                add = 1
                layer = SwinBlock(dim=embed_dim*(2**i_layer), 
                                 input_resolution=(
                                     patches_resolution[0] // (2**(i_layer+add)),
                                     patches_resolution[1] // (2**(i_layer+add))
                                 ),
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 shift_size=0 if (dep % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, 
                                 qkv_bias=True, 
                                 qk_scale=None, 
                                 posmlp_dim=posmlp_dim,
                                 drop_path=0.,
                                 downsample=PatchMerge(
                                     patch_size=2,
                                     in_chans=embed_dim*(2**(i_layer - 1)), 
                                     embed_dim=embed_dim*(2**i_layer),
                                     norm_layer=norm_layer
                                 ) if downsample_flag else None
                )
                self.blocks.append(layer)
        
        # Normalization for different stages
        self.norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            norm = norm_layer(embed_dim * (2 ** i_layer))
            self.norms.append(norm)
        
        # --------------------------------------------------------------------------
        # Segmentation Head specifics (similar to ViT version)
        
        # FPN-style decoder
        self.decoder = FPNHEAD()
        
        # Final segmentation classifier
        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1),
        )
        
        # Multi-scale feature processing layers
        # Stage 0: embed_dim (96) -> 256 channels, upsample 8x
        self.conv0 = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 1, 1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 256, 8, 8),  # 8x upsampling
            nn.Dropout(dropout)
        )
        
        # Stage 1: embed_dim*2 (192) -> 512 channels, upsample 4x  
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, 1, 1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 512, 4, 4),  # 4x upsampling
            nn.Dropout(dropout)
        )
        
        # Stage 2: embed_dim*4 (384) -> 1024 channels, upsample 2x
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim * 4, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 1024, 2, 2),  # 2x upsampling
            nn.Dropout(dropout)
        )
        
        # Stage 3: embed_dim*8 (768) -> 2048 channels, no upsampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim * 8, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        self.apply(self._init_weights)
        
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, x):
        """Forward through Swin Transformer backbone"""
        N = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # [N, H*W, C]
        
        # Prepare position encoding
        pos_hw = self.pos_hw.repeat(N, 1, 1, 1).to(x.device)
        pos_hw = pos_hw.permute(0, 2, 3, 1)  # (N, H, W, 2)
        
        # Multi-stage features for segmentation
        stage_features = []
        block_idx = 0
        
        for i_layer in range(self.num_layers):
            # Process blocks in current stage
            for dep in range(len(self.blocks) // self.num_layers):  # Approximate division
                if block_idx < len(self.blocks):
                    x, pos_hw = self.blocks[block_idx](x, pos_hw)
                    block_idx += 1
            
            # Apply normalization and collect stage features
            if i_layer < len(self.norms):
                stage_x = self.norms[i_layer](x)
                
                # Reshape to spatial format for convolution
                L = stage_x.shape[1]
                H = W = int(L ** 0.5)
                stage_x = stage_x.reshape(N, H, W, -1).permute(0, 3, 1, 2)
                stage_features.append(stage_x)
        
        return stage_features
    
    def forward(self, x):
        """Forward pass for segmentation"""
        # Extract multi-stage features
        stage_features = self.forward_features(x)
        
        # Process features through convolution layers
        m = {}
        
        if len(stage_features) >= 1:
            m[0] = self.conv0(stage_features[0])  # 256 channels
        if len(stage_features) >= 2:
            m[1] = self.conv1(stage_features[1])  # 512 channels  
        if len(stage_features) >= 3:
            m[2] = self.conv2(stage_features[2])  # 1024 channels
        if len(stage_features) >= 4:
            m[3] = self.conv3(stage_features[3])  # 2048 channels
        
        # FPN decoder
        m_list = list(m.values())
        if len(m_list) > 0:
            x = self.decoder(m_list)
            
            # Final segmentation prediction
            x = self.cls_seg(x)
        else:
            # Fallback if no features extracted
            x = torch.zeros(x.shape[0], self.num_classes, 
                          x.shape[2], x.shape[3], device=x.device)
        
        return {'out': x}


# 如果您需要处理高光谱数据的版本，可以添加类似ViT版本的时间维度处理
class SwinTransformerHyperspectralSegmentation(SwinTransformerSegmentation):
    """Swin Transformer for Hyperspectral Segmentation"""
    
    def __init__(self, num_frames=12, t_patch_size=4, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.t_patch_size = t_patch_size
        
        # Add temporal processing layer similar to ViT version
        self.fc = nn.Sequential(
            nn.Linear(int(num_frames/t_patch_size), 1)
        )
    
    def forward(self, x):
        """Forward pass for hyperspectral segmentation"""
        # Handle temporal dimension similar to ViT
        if len(x.shape) == 4:  # [N, C, H, W]
            x = torch.unsqueeze(x, dim=1)  # Add temporal dimension
        
        # Process through Swin backbone
        stage_features = self.forward_features(x)
        
        # Apply temporal processing if needed
        processed_features = []
        for feat in stage_features:
            if feat.dim() == 5:  # Has temporal dimension
                B, T, C, H, W = feat.shape
                feat = feat.permute(0, 3, 4, 1, 2)  # [B, H, W, T, C]
                feat = self.fc(feat.permute(0, 1, 2, 4, 3)).squeeze(-1)  # [B, H, W, C]
                feat = feat.permute(0, 3, 1, 2)  # [B, C, H, W]
            processed_features.append(feat)
        
        # Process through decoder similar to base class
        m = {}
        if len(processed_features) >= 1:
            m[0] = self.conv0(processed_features[0])
        if len(processed_features) >= 2:
            m[1] = self.conv1(processed_features[1])
        if len(processed_features) >= 3:
            m[2] = self.conv2(processed_features[2])
        if len(processed_features) >= 4:
            m[3] = self.conv3(processed_features[3])
        
        m_list = list(m.values())
        if len(m_list) > 0:
            x = self.decoder(m_list)
            x = self.cls_seg(x)
        else:
            x = torch.zeros(x.shape[0], self.num_classes, 
                          x.shape[2], x.shape[3], device=x.device)
        
        return {'out': x}
