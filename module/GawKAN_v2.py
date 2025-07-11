import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward

class GraphKANLayer(nn.Module):
    def __init__(self, feat_dim, out_dim, adj, dropout=0.1):
        super().__init__()
        self.adj = adj  # (node, node)
        self.linear = nn.Linear(feat_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, node, feat_dim)
        agg = torch.einsum('ij,bjf->bif', self.adj, x)  # (batch, node, feat_dim)
        out = self.linear(agg)
        out = self.dropout(out)
        return out

class WaveletKANLayer(nn.Module):
    def __init__(self, feat_dim, out_dim, wavelet_type='db1', dwt_level=1, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.dwt_level = dwt_level
        self.dwt = DWT1DForward(J=dwt_level, wave=wavelet_type, mode='zero')
        test_arr = torch.zeros(1, 1, 2 ** dwt_level)
        cA, cD_list = self.dwt(test_arr)
        wavelet_lens = [cA.shape[-1]] + [cD.shape[-1] for cD in cD_list]
        self.wavelet_full_len = sum(wavelet_lens)
        self.nonlinear = nn.ModuleList([nn.Linear(self.wavelet_full_len, 1) for _ in range(feat_dim)])
        self.linear_out = nn.Linear(feat_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, node, feat_dim)
        batch, node, feat = x.shape
        outs = []
        for i in range(feat):
            xi = x[:, :, i].contiguous().view(-1, 1)
            xi_wave = self._wavelet_feature(xi, i)
            xi_wave = xi_wave.view(batch, node, 1)
            outs.append(xi_wave)
        agg_nl = torch.cat(outs, dim=-1)  # (batch, node, feat)
        out = self.linear_out(agg_nl)
        out = self.dropout(out)
        return out

    def _wavelet_feature(self, x, i):
        batch = x.shape[0]
        min_len = 2 ** self.dwt_level
        if x.shape[1] < min_len:
            pad = min_len - x.shape[1]
            x = F.pad(x, (0, pad))
        x_reshape = x.view(batch, 1, -1)
        cA, cD_list = self.dwt(x_reshape)
        features = [cA.squeeze(1)]
        for cD in cD_list:
            features.append(cD.squeeze(1))
        feat_cat = torch.cat(features, dim=-1)
        out = self.nonlinear[i](feat_cat)
        return out

class GawKAN(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, hidden_dim=64, num_layers=2, wavelet_type='db1', dwt_level=2, adj=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.wavelet_type = wavelet_type
        self.dwt_level = dwt_level
        self.adj = adj
        self.dropout = dropout
        # 先堆叠Graph层，再堆叠Wavelet层
        self.graph_layers = nn.ModuleList()
        self.wavelet_layers = nn.ModuleList()
        layer_in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        layer_out_dims = [hidden_dim] * (num_layers - 1) + [hidden_dim]
        for i in range(num_layers):
            self.graph_layers.append(GraphKANLayer(layer_in_dims[i], layer_out_dims[i], adj, dropout=dropout))
            self.wavelet_layers.append(WaveletKANLayer(layer_out_dims[i], hidden_dim if i < num_layers - 1 else output_dim, wavelet_type, dwt_level, dropout=dropout))
        self.res = nn.Linear(input_dim, output_dim)

    def forward(self, x, occ=None):
        # x: (batch, node, input_dim)
        res = self.res(x)
        for graph_layer, wavelet_layer in zip(self.graph_layers, self.wavelet_layers):
            x_in = x
            x = graph_layer(x)
            if x.shape == x_in.shape:
                x = x + x_in
            x_in2 = x
            x = wavelet_layer(x)
            if x.shape == x_in2.shape:
                x = x + x_in2
        return x + res