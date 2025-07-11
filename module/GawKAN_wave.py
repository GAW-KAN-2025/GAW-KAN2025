import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward
import csv
import os

class GraphKANLayer(nn.Module):
    def __init__(self, feat_dim, out_dim, adj, dropout=0.1):
        super().__init__()
        self.adj = adj  # (node, node)
        self.linear = nn.Linear(feat_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        agg = torch.einsum('ij,bjf->bif', self.adj, x)
        out = self.linear(agg)
        out = self.dropout(out)
        return out

class WaveletKANLayer(nn.Module):
    def __init__(self, feat_dim, out_dim, wavelet_type='db1', max_dwt_level=3, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.wavelet_type = wavelet_type
        self.max_dwt_level = max_dwt_level
        self.dropout = nn.Dropout(dropout)
        self.level_logits = nn.Parameter(torch.randn(max_dwt_level))  # [L]
        self.dwt_cache = {}
        # 预先计算最大级数下的特征长度
        test_arr = torch.zeros(1, 1, 2 ** max_dwt_level)
        dwt = DWT1DForward(J=max_dwt_level, wave=wavelet_type, mode='zero')
        cA, cD_list = dwt(test_arr)
        wavelet_lens = [cA.shape[-1]] + [cD.shape[-1] for cD in cD_list]
        self.wavelet_full_len = sum(wavelet_lens)
        self.nonlinear = nn.ModuleList([nn.Linear(self.wavelet_full_len, 1) for _ in range(feat_dim)])
        self.linear_out = nn.Linear(feat_dim, out_dim)

    def get_dwt(self, dwt_level, device=None):
        if dwt_level not in self.dwt_cache:
            self.dwt_cache[dwt_level] = DWT1DForward(J=dwt_level, wave=self.wavelet_type, mode='zero')
        dwt = self.dwt_cache[dwt_level]
        if device is not None:
            dwt = dwt.to(device)
        return dwt

    def forward(self, x):
        batch, node, feat = x.shape
        outs = []
        for i in range(feat):
            xi = x[:, :, i].contiguous().view(-1, 1)
            level_features = []
            for j in range(1, self.max_dwt_level + 1):
                xi_wave = self._wavelet_feature(xi, i, j)
                level_features.append(xi_wave)
            # Gumbel-Softmax采样one-hot权重
            weights = F.gumbel_softmax(self.level_logits, tau=1.0, hard=True)
            xi_wave_soft = sum(w * f for w, f in zip(weights, level_features))
            xi_wave_soft = xi_wave_soft.view(batch, node, 1)
            outs.append(xi_wave_soft)
        agg_nl = torch.cat(outs, dim=-1)
        out = self.linear_out(agg_nl)
        out = self.dropout(out)
        return out

    def _wavelet_feature(self, x, i, dwt_level):
        batch = x.shape[0]
        min_len = 2 ** dwt_level
        if x.shape[1] < min_len:
            pad = min_len - x.shape[1]
            x = F.pad(x, (0, pad))
        x_reshape = x.view(batch, 1, -1)
        device = x_reshape.device
        dwt = self.get_dwt(dwt_level, device=device)
        cA, cD_list = dwt(x_reshape)
        features = [cA.squeeze(1)]
        for cD in cD_list:
            features.append(cD.squeeze(1))
        feat_cat = torch.cat(features, dim=-1)
        # pad到最大长度
        if feat_cat.shape[-1] < self.wavelet_full_len:
            pad_len = self.wavelet_full_len - feat_cat.shape[-1]
            feat_cat = F.pad(feat_cat, (0, pad_len))
        out = self.nonlinear[i](feat_cat)
        return out

def dynamic_skip_fn(i, layer_type, x):
    if layer_type == 'wavelet' and x.std().item() < 1e-3:
        return True
    return False

class GawKAN(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, hidden_dim=64, num_layers=2, wavelet_type='db1', max_dwt_level=3, adj=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.wavelet_type = wavelet_type
        self.max_dwt_level = max_dwt_level
        self.adj = adj
        self.dropout = dropout
        self.graph_layers = nn.ModuleList()
        self.wavelet_layers = nn.ModuleList()
        self.res = nn.Linear(input_dim, output_dim)
        # 记录每层的输出维度
        layer_in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        graph_out_dims = [hidden_dim] * (num_layers - 1) + [hidden_dim]
        wavelet_out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.res_linears_graph = nn.ModuleList()
        self.res_linears_wavelet = nn.ModuleList()
        for i in range(num_layers):
            self.graph_layers.append(GraphKANLayer(layer_in_dims[i], graph_out_dims[i], adj, dropout=dropout))
            self.wavelet_layers.append(WaveletKANLayer(graph_out_dims[i], wavelet_out_dims[i], wavelet_type, max_dwt_level, dropout=dropout))
            self.res_linears_graph.append(nn.Linear(output_dim, graph_out_dims[i]))
            self.res_linears_wavelet.append(nn.Linear(output_dim, wavelet_out_dims[i]))
        self.theta_history = []

    def forward(self, x, occ=None):
        res = self.res(x)
        graph_outputs = [x]
        wavelet_outputs = []
        out = res
        num_layers = len(self.graph_layers)
        theta_list_graph = []
        theta_list_wavelet = []
        for i, (graph_layer, wavelet_layer, res_linear_graph, res_linear_wavelet) in enumerate(
                zip(self.graph_layers, self.wavelet_layers, self.res_linears_graph, self.res_linears_wavelet)):
            # Graph层
            x_in = graph_outputs[-1]
            x_g = graph_layer(x_in)
            if x_g.shape == x_in.shape:
                x_g = x_g + x_in
            res_proj_graph = res_linear_graph(res)
            diff_graph = F.mse_loss(x_g, res_proj_graph, reduction='mean')
            theta_graph = torch.sigmoid(diff_graph)
            x_g = theta_graph * x_g + (1 - theta_graph) * res_proj_graph
            theta_list_graph.append(theta_graph.item())
            graph_outputs.append(x_g)
            # Wavelet层
            x_in2 = x_g
            x_w = wavelet_layer(x_in2)
            if x_w.shape == x_in2.shape:
                x_w = x_w + x_in2
            res_proj_wavelet = res_linear_wavelet(res)
            diff_wavelet = F.mse_loss(x_w, res_proj_wavelet, reduction='mean')
            theta_wavelet = torch.sigmoid(diff_wavelet)
            x_w = theta_wavelet * x_w + (1 - theta_wavelet) * res_proj_wavelet
            theta_list_wavelet.append(theta_wavelet.item())
            wavelet_outputs.append(x_w)
            out = x_w
        self.theta_history.append({'graph': theta_list_graph, 'wavelet': theta_list_wavelet})
        return out
    
