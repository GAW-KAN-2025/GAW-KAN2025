import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward

class GraphKANLayer(nn.Module):
    def __init__(self, feat_dim, out_dim, adj, wavelet_type='db1', dwt_level=1, dropout=0.1):
        super().__init__()
        self.adj = adj  # (node, node)
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.dwt = DWT1DForward(J=dwt_level, wave=wavelet_type, mode='zero')
        test_arr = torch.zeros(1, 1, 1)
        cA, _ = self.dwt(test_arr)
        self.wave_len = cA.shape[-1]
        self.nonlinear = nn.ModuleList([nn.Linear(self.wave_len, 1) for _ in range(feat_dim)])
        self.linear_out = nn.Linear(feat_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, node, feat_dim)
        batch, node, feat = x.shape
        agg = torch.einsum('ij,bjf->bif', self.adj, x)  # (batch, node, feat)
        outs = []
        for i in range(feat):
            xi = agg[:, :, i].contiguous().view(-1, 1)
            xi_wave = self._wavelet_feature(xi, i)
            xi_wave = xi_wave.view(batch, node, 1)
            outs.append(xi_wave)
        agg_nl = torch.cat(outs, dim=-1)  # (batch, node, feat)
        out = self.linear_out(agg_nl)
        out = self.dropout(out)
        return out

    def _wavelet_feature(self, x, i):
        batch = x.shape[0]
        x_reshape = x.view(batch, 1, 1)
        cA, _ = self.dwt(x_reshape)
        cA = cA.squeeze(1)
        out = self.nonlinear[i](cA)
        return out

class WGAK(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, hidden_dim=64, num_layers=2, wavelet_type='db1', dwt_level=1, adj=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.wavelet_type = wavelet_type
        self.dwt_level = dwt_level
        self.adj = adj
        self.dropout = dropout
        # 多层GraphKANLayer堆叠
        layer_in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        layer_out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            GraphKANLayer(layer_in_dims[i], layer_out_dims[i], adj, wavelet_type, dwt_level, dropout=dropout)
            for i in range(num_layers)
        ])
        self.res = nn.Linear(input_dim, output_dim)

    def forward(self, x, occ=None):
        # x: (batch, node, input_dim)
        res = self.res(x)
        for layer in self.layers:
            x = layer(x)
        return x + res