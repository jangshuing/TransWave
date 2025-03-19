import torch
import torch.nn as nn

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling

        # Shared MLP, using 1x1 convolutions instead of fully connected layers
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),  # Dimensionality reduction
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # Reconstruct original dimension
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input dimensions: [B, C, H, W]
        avg_out = self.fc(self.avg_pool(x))  # After average pooling: [B, C, 1, 1]
        max_out = self.fc(self.max_pool(x))  # After max pooling: [B, C, 1, 1]
        out = avg_out + max_out  # Element-wise sum, output dimensions: [B, C, 1, 1]
        return self.sigmoid(out)  # Output dimensions: [B, C, 1, 1]


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)  # 7x7 convolution
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input dimensions: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling, output dimensions: [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling, output dimensions: [B, 1, H, W]
        x = torch.cat([avg_out, max_out], dim=1)  # Concatenate along the channel dimension, output dimensions: [B, 2, H, W]
        x = self.conv1(x)  # 7x7 convolution, output dimensions: [B, 1, H, W]
        return self.sigmoid(x)  # Output dimensions: [B, 1, H, W]


# CBAM (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # Channel Attention Module
        self.sa = SpatialAttention(kernel_size)  # Spatial Attention Module

    def forward(self, x):
        # Channel Attention Module
        ca_out = self.ca(x)  # Output dimensions: [B, C, 1, 1]
        x = ca_out * x  # Element-wise weighting by channel, output dimensions: [B, C, H, W]

        # Spatial Attention Module
        sa_out = self.sa(x)  # Output dimensions: [B, 1, H, W]
        x = sa_out * x  # Element-wise weighting by spatial attention, output dimensions: [B, C, H, W]
        
        return x  # Final output dimensions: [B, C, H, W]
