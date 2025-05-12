import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3, use_bspline=False, n_basis=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.use_bspline = use_bspline
        self.n_basis = n_basis
        self.poly_coeff = nn.Parameter(torch.randn(input_dim, degree + 1))
        self.linear = nn.Linear(input_dim, output_dim)
        if self.use_bspline:
            knots = np.linspace(0, 1, n_basis - degree + 1 + 2 * degree)
            self.register_buffer('knots', torch.tensor(knots, dtype=torch.float32))
            self.bspline_weights = nn.Parameter(torch.randn(input_dim, n_basis))

    def forward(self, x):
        # x: (batch, input_dim)
        if self.use_bspline:
            # 归一化到[0,1]
            x_min = x.min(dim=0, keepdim=True)[0]
            x_max = x.max(dim=0, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)  # (batch, input_dim)
            batch, input_dim = x_norm.shape
            basis_tensor = []
            knots = self.knots.cpu().numpy()
            for i in range(self.input_dim):
                basis_i = []
                for j in range(self.n_basis):
                    c = np.zeros(self.n_basis)
                    c[j] = 1
                    b = BSpline(knots, c, self.degree)
                    basis_i.append(torch.from_numpy(b(x_norm[:, i].detach().cpu().numpy())).to(x.device).float())
                # (n_basis, batch) -> (batch, n_basis)
                basis_i = torch.stack(basis_i, dim=1)
                basis_tensor.append(basis_i)
            # (input_dim, batch, n_basis) -> (batch, input_dim, n_basis)
            basis_tensor = torch.stack(basis_tensor, dim=1)
            # einsum: (batch, input_dim, n_basis), (input_dim, n_basis) -> (batch, input_dim)
            x_bspline = torch.einsum('bin,in->bi', basis_tensor, self.bspline_weights)
            output = self.linear(x_bspline)
            return output
        else:
            exponents = torch.arange(self.degree + 1, device=x.device)
            x_expanded = x.unsqueeze(-1) ** exponents  # (batch, input_dim, degree+1)
            x_nonlinear = torch.sum(x_expanded * self.poly_coeff, dim=-1)  # (batch, input_dim)
            output = self.linear(x_nonlinear)  # (batch, output_dim)
            return output

class KAN(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, degree=3, use_bspline=False, n_basis=8):
        super().__init__()
        self.kan_layer = KANLayer(input_dim, output_dim, degree, use_bspline, n_basis)
        self.res = nn.Linear(input_dim, output_dim)

    def forward(self, x, occ=None):
        # x: (batch, seq_len, input_dim)
        res = self.res(x)
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.kan_layer.input_dim)  # (batch*seq_len, input_dim)
        output = self.kan_layer(x)  # (batch*seq_len, output_dim)
        return output.view(batch_size, seq_len, -1) + res  # (batch, seq_len, output_dim)