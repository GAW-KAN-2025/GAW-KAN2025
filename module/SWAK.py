import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline
from pytorch_wavelets import DWT1DForward

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3, use_bspline=False, n_basis=8, adj=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.use_bspline = use_bspline
        self.n_basis = n_basis
        self.poly_coeff = nn.Parameter(torch.randn(input_dim, degree + 1))
        self.linear = nn.Linear(input_dim, output_dim)
        self.attn_fc = nn.Linear(input_dim, 1)  # 用于attention分数
        self.adj = adj  # 保存邻接矩阵
        if self.use_bspline:
            knots = np.linspace(0, 1, n_basis - degree + 1 + 2 * degree)
            self.register_buffer('knots', torch.tensor(knots, dtype=torch.float32))
            self.bspline_weights = nn.Parameter(torch.randn(input_dim, n_basis))

    def node_feature(self, x):
        # x: (batch, node, input_dim)
        if self.use_bspline:
            # 归一化到[0,1]，每个batch独立归一化
            x_min = x.min(dim=1, keepdim=True)[0]  # (batch, 1, input_dim)
            x_max = x.max(dim=1, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)  # (batch, node, input_dim)
            batch, node, input_dim = x_norm.shape
            basis_tensor = []
            knots = self.knots.cpu().numpy()
            for i in range(self.input_dim):
                basis_i = []
                for j in range(self.n_basis):
                    c = np.zeros(self.n_basis)
                    c[j] = 1
                    b = BSpline(knots, c, self.degree)
                    vals = b(x_norm[:, :, i].detach().cpu().numpy())  # (batch, node)
                    basis_i.append(torch.from_numpy(vals).to(x.device).float())
                # (n_basis, batch, node) -> (batch, node, n_basis)
                basis_i = torch.stack(basis_i, dim=0).permute(1, 2, 0)
                basis_tensor.append(basis_i)
            # (input_dim, batch, node, n_basis) -> (batch, node, input_dim, n_basis)
            basis_tensor = torch.stack(basis_tensor, dim=0).permute(1,2,0,3)
            # einsum: (batch, node, input_dim, n_basis), (input_dim, n_basis) -> (batch, node, input_dim)
            x_bspline = torch.einsum('bnij,ij->bni', basis_tensor, self.bspline_weights)
            return x_bspline  # (batch, node, input_dim)
        else:
            exponents = torch.arange(self.degree + 1, device=x.device)
            x_expanded = x.unsqueeze(-1) ** exponents  # (batch, node, input_dim, degree+1)
            x_nonlinear = torch.sum(x_expanded * self.poly_coeff, dim=-1)  # (batch, node, input_dim)
            return x_nonlinear

    def forward(self, x):
        # x: (batch, node, input_dim)
        batch, node, feat = x.shape
        h = self.node_feature(x)  # (batch, node, input_dim)
        h_proj = self.linear(h)   # (batch, node, output_dim)
        attn_scores = self.attn_fc(h).squeeze(-1)  # (batch, node)
        adj_mask = (self.adj > 0).float()  # (node, node)
        attn_matrix = attn_scores.unsqueeze(1).repeat(1, node, 1)  # (batch, node, node)
        attn_matrix = attn_matrix * adj_mask.unsqueeze(0)  # 只保留邻居分数
        attn_matrix = torch.softmax(attn_matrix.masked_fill(adj_mask.unsqueeze(0)==0, -1e9), dim=-1)
        out = torch.einsum('bij,bjd->bid', attn_matrix, h_proj)  # (batch, node, output_dim)
        return out

class SWAK(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, degree=3, use_bspline=False, n_basis=8, adj=None):
        super().__init__()
        self.dwt = DWT1DForward(J=1, wave='db1', mode='zero')
        # 推断小波变换后长度
        test_arr = torch.zeros(1, 1, input_dim)
        cA, _ = self.dwt(test_arr)
        self.wave_len = cA.shape[-1]
        self.res_wave = nn.Linear(self.wave_len, output_dim)
        self.kan_layer_wave = KANLayer(self.wave_len, output_dim, degree, use_bspline, n_basis, adj)
        self.adj = adj

    def forward(self, x, occ=None):
        x = x.float()
        batch, node, seq_len = x.shape
        x_reshape = x.reshape(batch * node, 1, seq_len)
        cA, _ = self.dwt(x_reshape)
        cA = cA.squeeze(1).reshape(batch, node, -1)
        wave_len = cA.shape[-1]
        # 动态创建/更新 res_wave 和 kan_layer_wave
        if (not hasattr(self, 'res_wave')) or (self.res_wave.in_features != wave_len):
            self.res_wave = nn.Linear(wave_len, self.kan_layer_wave.output_dim if hasattr(self, 'kan_layer_wave') else 1).to(x.device)
        if (not hasattr(self, 'kan_layer_wave')) or (self.kan_layer_wave.input_dim != wave_len):
            out_dim = self.kan_layer_wave.output_dim if hasattr(self, 'kan_layer_wave') else self.res_wave.out_features
            self.kan_layer_wave = KANLayer(wave_len, out_dim, degree=3, use_bspline=False, n_basis=8, adj=self.adj).to(x.device)
        cA_flat = cA.view(-1, wave_len)
        res = self.res_wave(cA_flat).view(batch, node, -1)
        out = self.kan_layer_wave(cA)
        return out + res  # (batch, node, output_dim)