import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

# Function to create wavelet filters for decomposition and reconstruction
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    
    # Initialize the db1 wavelet filter (decomposition filters)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL (low-low frequency)
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH (low-high frequency)
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL (high-low frequency)
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)  # HH (high-high frequency)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # Reconstruction filters
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    
    return dec_filters, rec_filters


# Perform wavelet transform on the input tensor using the given filters
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)

    groups = c  # Apply the corresponding filter to each input channel independently
    x = F.conv2d(x, filters, stride=2, groups=groups, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)  # Reshape the output into 4 frequency components
    return x


# Perform inverse wavelet transform to reconstruct the image from the wavelet components
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)  # Flatten the 4 components back into a single tensor
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
