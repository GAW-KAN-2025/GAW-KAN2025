import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pytorch_wavelets import DWT1DForward

class WavKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, wavelet_type='db1', dwt_level=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wavelet_type = wavelet_type
        self.dwt_level = dwt_level
        self.dwt = DWT1DForward(J=dwt_level, wave=wavelet_type, mode='zero')
        # 线性层输入维度需根据小波变换后长度动态调整
        test_arr = torch.zeros(1, 1, input_dim)
        cA, _ = self.dwt(test_arr)
        self.wave_len = cA.shape[-1]
        self.linear = nn.Linear(self.wave_len, output_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        batch_size = x.shape[0]
        # 变形为(batch, 1, input_dim)以适配DWT1DForward
        x_reshape = x.view(batch_size, 1, self.input_dim)
        cA, _ = self.dwt(x_reshape)
        # cA: (batch, 1, wave_len) -> (batch, wave_len)
        cA = cA.squeeze(1)
        out = self.linear(cA)
        out = F.softplus(out)
        return out

class WavKAN(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, wavelet_type='db1', dwt_level=1):
        super().__init__()
        self.wavkan_layer = WavKANLayer(input_dim, output_dim, wavelet_type, dwt_level)
        self.res = nn.Linear(input_dim, output_dim)

    def forward(self, x, occ=None):
        # x: (batch, seq_len, input_dim)
        res = self.res(x)
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.wavkan_layer.input_dim)  # (batch*seq_len, input_dim)
        out = self.wavkan_layer(x_flat)  # (batch*seq_len, output_dim)
        out = out.view(batch_size, seq_len, -1)
        return out + res  # (batch, seq_len, output_dim)