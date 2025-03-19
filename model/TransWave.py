
import math
import logging
from functools import partial
from collections import OrderedDict
from model.attention import CBAM
from model import wavelet
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
_logger = logging.getLogger(__name__)

# Configuration function for model settings
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

# A memory-efficient implementation of the Swish activation function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

# Custom activation function: Adaptive Thresholding followed by Swish
class AdaptiveThresholdingSwish(nn.Module):
    def __init__(self, in_channels, threshold=0.1, smooth_factor=5.0):
        super(AdaptiveThresholdingSwish, self).__init__()
        # Initialize a per-channel threshold as a learnable parameter
        self.threshold = nn.Parameter(torch.full((in_channels,), threshold, dtype=torch.float32))  # Threshold per channel
        self.smooth_factor = nn.Parameter(torch.tensor(smooth_factor, dtype=torch.float32))  # Learnable smooth factor
        
    def adaptive_thresholding(self, x):
        abs_x = torch.abs(x)
        threshold = self.threshold.view(1, -1, 1, 1)
        # Apply soft thresholding with tanh to prevent overflow
        clamped_value = torch.tanh(self.smooth_factor * (abs_x - threshold))
        smoothed_x = x - torch.sign(x) * (abs_x / (1 + clamped_value))
        
        # Only modify values below the threshold, keep values above threshold unchanged
        output = torch.where(abs_x <= threshold, smoothed_x, x)
        
        return output

    def forward(self, x):
        # Apply adaptive thresholding and Swish activation
        thresholded_x = self.adaptive_thresholding(x)
        swish_x = thresholded_x * torch.sigmoid(thresholded_x)
        return swish_x

# Wavelet transform block
class WT(nn.Module):
    def __init__(self, in_channels, wt_type='db1'):
        super(WT, self).__init__()
        self.in_channels = in_channels
        self.wt_filter, _ = wavelet.create_wavelet_filter(wt_type, self.in_channels, self.in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        
        self.wt_function = partial(wavelet.wavelet_transform, filters=self.wt_filter)
        # Denosing activation layers for each wavelet component
        self.denosice_act1 = AdaptiveThresholdingSwish(in_channels)
        self.denosice_act2 = AdaptiveThresholdingSwish(in_channels)
        self.denosice_act3 = AdaptiveThresholdingSwish(in_channels)
        self.denosice_act4 = AdaptiveThresholdingSwish(in_channels)

    def forward(self, x):
        original_h, original_w = x.shape[2], x.shape[3]
        # Padding if height or width is odd
        pad_h = original_h % 2
        pad_w = original_w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        # Apply wavelet transform
        x = self.wt_function(x)

        # Extract the four components: LL, LH, HL, HH
        ll = x[:, :, 0, :, :]  # Low-frequency component (LL)
        lh = x[:, :, 1, :, :]  # Horizontal high-frequency component (LH)
        hl = x[:, :, 2, :, :]  # Vertical high-frequency component (HL)
        hh = x[:, :, 3, :, :]  # Diagonal high-frequency component (HH)
        
        # Apply denoising activation on LH,HL,HH component
        lh_1 = self.denosice_act1(lh)
        hl_1 = self.denosice_act2(hl)
        hh_1 = self.denosice_act3(hh)
        return ll, lh_1, hl_1, hh_1, original_h, original_w

# Inverse Wavelet Transform (IWT)
class IWT(nn.Module):
    def __init__(self, in_channels, wt_type='db1'):
        super(IWT, self).__init__()
        self.in_channels = in_channels
        # Initialize the reconstruction filter for the wavelet transform
        _, self.rec_filter = wavelet.create_wavelet_filter(wt_type, self.in_channels, self.in_channels, torch.float)
        self.rec_filter = nn.Parameter(self.rec_filter, requires_grad=False)
        
        # Inverse wavelet transform function
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters=self.rec_filter)
        self.act = nn.GELU()

    def forward(self, ll, lh, hl, hh, original_h, original_w):
        # Combine the four components to [B, C, 4, H/2, W/2]
        x = torch.stack([ll, lh, hl, hh], dim=2)

        # Apply inverse wavelet transform to recover the original size
        x = self.iwt_function(x)

        # Crop the padded parts if the original size was odd
        x = x[:, :, :original_h, :original_w]
        
        # Apply activation function
        x = self.act(x)
        
        return x

# Stem block for the initial feature extraction
class StemBlock(nn.Module):
    def __init__(self, in_channels, stem_channel):
        super(StemBlock, self).__init__()
        self.stem_conv1 = nn.Conv2d(in_channels, stem_channel, kernel_size=7, stride=2, padding=3, bias=True)
        self.stem_gelu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_gelu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_gelu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

    def forward(self, x):
        x = self.stem_conv1(x)
        x = self.stem_gelu1(x)
        x = self.stem_norm1(x)
        
        x = self.stem_conv2(x)
        x = self.stem_gelu2(x)
        x = self.stem_norm2(x)
        
        x = self.stem_conv3(x)
        x = self.stem_gelu3(x)
        x = self.stem_norm3(x)

        return x

# Attention mechanism with custom spatial reduction
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1,related_posembed =None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio
        self.related_posembed = related_posembed
        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        self.cbam = CBAM(dim)
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
        progress_x = x.permute(0, 2, 1).reshape(B, C, H, W)
        progress_x = self.cbam(progress_x) + progress_x
        if self.sr_ratio > 1:
            progress_x = self.sr(progress_x).reshape(B, C, -1).permute(0, 2, 1)
            k = self.k(progress_x).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(progress_x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            progress_x = progress_x.reshape(B, C, -1).permute(0, 2, 1)
            k = self.k(progress_x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(progress_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.related_posembed
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FCNGLU(nn.Module):
    def __init__(self, in_features, hidden_features, drop):
        super(FCNGLU, self).__init__()
        
        # 1x1 convolution for dimensionality reduction and expansion (pointwise convolution)
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)

        # 3x3 depthwise convolution to extract spatial features
        self.depthwise_conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)  # 
        self.bn = nn.BatchNorm2d(hidden_features)  # Batch normalization
        
        # GLU gating branch, using 1x1 convolution + batch normalization + Sigmoid
        self.gate_conv = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.gate_bn = nn.BatchNorm2d(in_features)
        self.sigmoid = nn.Sigmoid()
        
        # Dropout layer
        self.drop = nn.Dropout(p=drop)

        # Residual scale coefficient
        self.residual_scale = nn.Parameter(torch.ones(1))  # Trainable parameter to control residual scaling

        # Using SiLU (Swish) activation function instead of ReLU
        self.act = nn.SiLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # First branch: Pointwise convolution + depthwise convolution operation
        x1 = self.conv1(x)  
        x1 = self.depthwise_conv(x1)
        x1 = self.bn(x1)
        x1 = self.act(x1)
        x1 = self.conv2(x1)

        # GLU gating branch
        gate = self.gate_conv(x1)
        gate = self.gate_bn(gate)
        gate = self.sigmoid(gate)
        
        # Multiply the convolution features by the gating branch element-wise
        x_glu = x1 * gate

        # Check if the residual connection needs dimension adjustment
        if x.shape != x_glu.shape:
            x = self.conv2(x)


        x = x + self.residual_scale * x_glu
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x



class Block(nn.Module):
    def __init__(self, stage_idx, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1, related_posembed=None,name=''):
        super().__init__()
        self.stage_idx = stage_idx
        self.norms = nn.ModuleList([norm_layer(dim) for _ in range(2)])
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio, related_posembed=related_posembed)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FCNGLU(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.wt = WT(dim)
        self.iwt = IWT(dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + self.proj(x)
        ll, lh, hl, hh,original_h,original_w = self.wt(x)
        iwt = self.iwt(ll, lh, hl, hh,original_h,original_w)
        iwt = iwt.flatten(2).permute(0, 2, 1).reshape(B,N,C)
        x = x.flatten(2).permute(0, 2, 1).reshape(B,N,C)
        x = x + self.drop_path(self.attn(self.norms[0](iwt), H, W))
        x = x + self.drop_path(self.mlp(self.norms[1](x), H, W))
        return x

class PatchEmbed(nn.Module):
    def __init__(self,stage_idx,img_size=224, patch_size=2, in_c=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.stage_idx = stage_idx
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if(stage_idx==0):
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == x.shape[2] and W == x.shape[3], \
            f"Input image size ({H}*{W}) doesn't match model ({x.shape[2]}*{x.shape[3]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class TransWave(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[46,92,184,368], stem_channel=16, fc_dim=1280,
                 num_heads=[1,2,4,8], mlp_ratios=[2,2,2,2], qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 depths=[2,2,10,2], qk_ratio=1, sr_ratios=[8,4,2,1],  dp=0.1, embed_layer=PatchEmbed):
        super().__init__()
        self.patch_size = 2
        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_features = self.embed_dim = embed_dims[-1]
        
        # Stem block for the initial feature extraction
        self.stem = StemBlock(in_chans, stem_channel=stem_channel)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # Depthwise drop path rates
        
        # Initialize relative position embeddings for different stages
        self.relative_pos_embeds = [
            nn.Parameter(torch.randn(num_heads[0], (self.img_size//4)**2, (self.img_size//4)**2//sr_ratios[0]//sr_ratios[0])),
            nn.Parameter(torch.randn(num_heads[1], (self.img_size//8)**2, (self.img_size//8)**2//sr_ratios[1]//sr_ratios[1])),
            nn.Parameter(torch.randn(num_heads[2], (self.img_size//16)**2, (self.img_size//16)**2//sr_ratios[2]//sr_ratios[2])),
            nn.Parameter(torch.randn(num_heads[3], (self.img_size//32)**2, (self.img_size//32)**2//sr_ratios[3]//sr_ratios[3])) 
        ]
        
        # Define stages
        self.stages = nn.ModuleList()
        cur = 0
        for stage_idx in range(4):
            stage_blocks = []
            patch_embed = embed_layer(stage_idx, img_size=self.img_size//2, patch_size=self.patch_size, 
                                      in_c=stem_channel, embed_dim=self.embed_dims[stage_idx])
            # Add Transformer Blocks to each stage
            for i in range(depths[stage_idx]):
                stage_blocks.append(
                    Block(stage_idx, dim=embed_dims[stage_idx], num_heads=num_heads[stage_idx], mlp_ratio=mlp_ratios[stage_idx], 
                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                          drop_path=dpr[cur + i], norm_layer=norm_layer, qk_ratio=qk_ratio, 
                          sr_ratio=sr_ratios[stage_idx], related_posembed=self.relative_pos_embeds[stage_idx])
                )
            cur += depths[stage_idx]
            # Define the stage with patch embedding and blocks
            stage = nn.Sequential(
                patch_embed,  
                nn.Sequential(*stage_blocks),  # Transformer blocks
            )
            self.stages.append(stage)

        # Representation layer if needed
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(embed_dim, representation_size)), ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Linear(fc_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = self.stem(x)
        # Pass through each stage
        for stage in self.stages:
            x = stage(x)
            B, N, C = x.shape
            x = x.flatten(2).permute(0, 2, 1).reshape(B, C, int(N**0.5), int(N**0.5))
        x = self._fc(x)
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x
    
    # Initialize model weights
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# Resizing position embedding for different input sizes
def resize_pos_embed(posemb, posemb_new):
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

# Checkpoint filtering function to adjust pretrained model's state_dict
def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict

def _create_TransWave_model(pretrained=False, distilled=False, **kwargs):
    default_cfg = _cfg()
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]
    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = TransWave(img_size=img_size, num_classes=num_classes, representation_size=repr_size, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(
            model, num_classes=num_classes, in_chans=kwargs.get('in_chans', 3),
            filter_fn=partial(checkpoint_filter_fn, model=model))
    return model

# Register model as "TransWvae"
@register_model
def transwave(pretrained=False, **kwargs):
    model_kwargs = dict(
        qkv_bias=True, embed_dims=[76,152,304,608], stem_channel=38, num_heads=[1,2,4,8],
        depths=[4,4,20,4], mlp_ratios=[4,4,4,4], qk_ratio=1, sr_ratios=[8,4,2,1], dp=0.3, **kwargs)
    model = _create_TransWave_model(pretrained=pretrained, **model_kwargs)
    return model

def test():
    # 模拟输入张量，形状为 (batch_size, channels, height, width)
    input_tensor = torch.randn(32, 3, 224, 224)

    # 初始化模型
    model = transwave(pretrained=False)
    print(model)
    # 前向传播
    output = model(input_tensor)

    # 打印输出形状
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test()
