import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from module.WGAK import WGAK
from tools import data_switcher

# Grad-CAM for WGAK
class WGAK_GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def __call__(self, x_tensor, occ_tensor=None, class_idx=None):
        self.model.eval()
        x_tensor = x_tensor.requires_grad_()
        if occ_tensor is not None:
            output = self.model(x_tensor, occ=occ_tensor)
        else:
            output = self.model(x_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=-1)
        # 若output不是标量，取均值
        if output.numel() > 1:
            target = output.mean()
        else:
            target = output.squeeze()
        self.model.zero_grad()
        target.backward(retain_graph=True)
        # Grad-CAM权重
        weights = self.gradients.mean(dim=(2))  # (batch, node)
        cam = (weights.unsqueeze(-1) * self.activations).sum(dim=1)  # (batch, seq_len)
        cam = cam[0].cpu().numpy()  # 取第一个样本
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        return cam

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载邻接矩阵
    ext = os.path.splitext(args.adj_path)[-1]
    if ext in ['.pt', '.pth']:
        adj = torch.load(args.adj_path)
    elif ext == '.npy':
        adj = torch.tensor(np.load(args.adj_path))
    elif ext == '.pkl':
        import pickle
        with open(args.adj_path, 'rb') as f:
            adj = pickle.load(f, encoding='latin1')
    elif ext == '.csv':
        # 跳过表头和首列（如有node_id）
        adj_df = pd.read_csv(args.adj_path, header=0)
        # 如果第一列不是数字，自动跳过
        if not np.issubdtype(adj_df.iloc[:,0].dtype, np.number):
            adj_np = adj_df.iloc[:,1:].values.astype(float)
        else:
            adj_np = adj_df.values.astype(float)
        adj = torch.tensor(adj_np)
    else:
        raise ValueError(f"Unsupported adj file format: {ext}")
    # 加载模型
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict):
        model = WGAK(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, wavelet_type=args.wavelet_type, dwt_level=args.dwt_level, adj=adj, dropout=args.dropout)
        model.load_state_dict(state)
        model.to(device)
    else:
        model = state
        model.to(device)
    # 打印所有可分析的layer名称
    print("[可分析的layer名称]:")
    for name, module in model.named_modules():
        print(f"  {name}")
    # 加载数据
    train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense = data_switcher.get_data_loaders(
        args.dataset, args.seq_len, args.pre_len, device, args.batch_size
    )
    x_data = train_occupancy[:args.num_samples * args.seq_len]
    seq_len = args.seq_len
    if x_data.ndim == 3:
        N = args.num_samples
        node_num = x_data.shape[-1]
        x_data = x_data.reshape(N, seq_len, node_num)
        x_data = np.transpose(x_data, (0, 2, 1))
    elif x_data.ndim == 2:
        N = args.num_samples
        node_num = x_data.shape[1]
        x_data = x_data.reshape(N, seq_len, node_num)
        x_data = np.transpose(x_data, (0, 2, 1))
    else:
        raise ValueError(f"不支持的x_data维度: {x_data.shape}")
    # 选取样本
    idx = args.sample_idx
    x_tensor = torch.tensor(x_data[idx:idx+1], dtype=torch.float32).to(device)
    occ_tensor = None
    # Grad-CAM
    target_layer = args.target_layer  # 例如 'gconv1' 或 'layers.0' 等
    gradcam = WGAK_GradCAM(model, target_layer)
    cam = gradcam(x_tensor, occ_tensor)
    # 可视化
    plt.figure(figsize=(10,4))
    plt.imshow(cam[None, :], aspect='auto', cmap='jet')
    plt.colorbar(label='Grad-CAM')
    plt.xlabel('Time Step')
    plt.yticks([])
    plt.title(f'WGAK Grad-CAM ({args.dataset}, sample {idx}, layer {target_layer})')
    plt.tight_layout()
    plt.savefig(f'./analysis/{args.dataset}_gradcam_sample{idx}_{target_layer}.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for WGAK model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to WGAK model .pt file')
    parser.add_argument('--adj_path', type=str, required=True, help='Path to adjacency matrix file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name for data_switcher', default='PEMS-BAY')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pre_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=12)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--wavelet_type', type=str, default='db1')
    parser.add_argument('--dwt_level', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample to explain')
    parser.add_argument('--target_layer', type=str, default='layers.0', help='Target layer name for Grad-CAM')
    args = parser.parse_args()
    main(args)
