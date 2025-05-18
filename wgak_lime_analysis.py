import torch
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
import random
from pyheatmap.heatmap import HeatMap
from PIL import Image

# 固定随机种子，保证LIME可复现
random.seed(2025)
np.random.seed(2025)

# 假设WGAK模型和数据加载函数已在STkan2025.module和tools中
from module.GawKAN import GawKAN
from tools import data_switcher

# 示例数据加载函数（需根据实际数据格式修改）
def load_sample_data(data_path, num_samples=100):
    data = np.load(data_path)  # 假设为npy格式，shape: (N, node, feat)
    return data[:num_samples]

def model_predict_fn(model, device, occ_tensor, x_tensor):
    # x_tensor, occ_tensor: (batch, node, feat)
    model.eval()
    with torch.no_grad():
        if occ_tensor is not None:
            pred = model(x_tensor.to(device), occ=occ_tensor.to(device))
        else:
            pred = model(x_tensor.to(device))
    return pred.cpu().numpy().reshape((x_tensor.shape[0], -1))

def load_adj(adj_path):
    ext = os.path.splitext(adj_path)[-1]
    if ext in ['.pt', '.pth']:
        return torch.load(adj_path)
    elif ext == '.npy':
        return torch.tensor(np.load(adj_path))
    elif ext == '.pkl':
        with open(adj_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    elif ext == '.csv':
        # 支持邻接矩阵为csv格式，强制转为float类型
        adj_df = pd.read_csv(adj_path, header=None)
        adj_np = adj_df.values.astype(float)
        return torch.tensor(adj_np)
    else:
        raise ValueError(f"Unsupported adj file format: {ext}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'ST-EVCDP':
        # ST-EVCDP直接用data_switcher返回的adj_dense
        train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense = data_switcher.get_data_loaders(
            args.dataset, args.seq_len, args.pre_len, device, args.batch_size
        )
        adj = adj_dense
        # 加载模型或参数
        state = torch.load(args.model_path, map_location=device)
        if isinstance(state, dict):
            model = GawKAN(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, wavelet_type=args.wavelet_type, dwt_level=args.dwt_level, adj=adj, dropout=args.dropout)
            model.load_state_dict(state)
            model.to(device)
        else:
            model = state
            model.to(device)
    else:
        # 其他数据集从文件加载adj
        adj = load_adj(args.adj_path)
        train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense = data_switcher.get_data_loaders(
            args.dataset, args.seq_len, args.pre_len, device, args.batch_size
        )
        # 加载模型或参数
        state = torch.load(args.model_path, map_location=device)
        if isinstance(state, dict):
            model = GawKAN(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, wavelet_type=args.wavelet_type, dwt_level=args.dwt_level, adj=adj, dropout=args.dropout)
            model.load_state_dict(state)
            model.to(device)
        else:
            model = state
            model.to(device)
    # 使用validation集数据
    val_x_list, val_occ_list = [], []
    num_val = 0
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            x, occ = batch[0], batch[1]
        else:
            x = batch
            occ = None
        val_x_list.append(x.cpu().numpy())
        if occ is not None:
            val_occ_list.append(occ.cpu().numpy())
        num_val += x.shape[0]
        if num_val >= args.num_samples:
            break
    x_data = np.concatenate(val_x_list, axis=0)[:args.num_samples]
    if val_occ_list:
        occ_data = np.concatenate(val_occ_list, axis=0)[:args.num_samples]
    else:
        occ_data = np.zeros_like(x_data)
    print("原始x_data.shape:", x_data.shape)
    print("原始occ_data.shape:", occ_data.shape)

    # 计算每个节点在时间维度的方差
    time_var = np.var(x_data, axis=2)  # (N, node)
    print("每个节点在时间维度的方差均值:", np.mean(time_var))
    print("每个节点在时间维度的方差最大值:", np.max(time_var))
    print("每个节点在时间维度的方差的方差:", np.var(time_var))

    seq_len = args.seq_len
    # 自动判断x_data shape并转为(N, node, seq_len)
    if x_data.ndim == 3:
        if x_data.shape[2] == seq_len:
            # (N, node, seq_len)
            pass
        elif x_data.shape[1] == seq_len:
            # (N, seq_len, node) -> (N, node, seq_len)
            x_data = np.transpose(x_data, (0, 2, 1))
            occ_data = np.transpose(occ_data, (0, 2, 1))
        else:
            raise ValueError(f"不支持的x_data shape: {x_data.shape}")
    elif x_data.ndim == 2:
        # (N, node*seq_len)
        node = x_data.shape[1] // seq_len
        x_data = x_data.reshape(-1, node, seq_len)
        occ_data = occ_data.reshape(-1, node, seq_len)
    else:
        raise ValueError(f"不支持的x_data维度: {x_data.shape}")
    print("转置后x_data.shape:", x_data.shape)
    print("转置后occ_data.shape:", occ_data.shape)
    N, node, seq_len = x_data.shape

    x_flat = x_data.reshape(N, -1)
    input_flat = x_flat  # 只分析x，不分析occ

    # 生成友好的feature_names（只x）
    feature_names = []
    for n in range(node):
        for t in range(seq_len):
            feature_names.append(f'x[{n},{t}]')

    # 构造LIME解释器
    explainer = LimeTabularExplainer(input_flat, mode="regression", feature_names=feature_names)

    def lime_predict(input_flat):
        batch = input_flat.shape[0]
        expected_feat = node * seq_len
        if input_flat.shape[1] != expected_feat:
            raise ValueError(f"LIME input shape mismatch: got {input_flat.shape}, expected {node}*{seq_len}={expected_feat}")
        x_part = input_flat.reshape(batch, node, seq_len)
        x_tensor = torch.tensor(x_part, dtype=torch.float32)
        occ_tensor = None  # 忽略occ
        return model_predict_fn(model, device, occ_tensor, x_tensor)

    # 全局LIME解释
    num_global_samples = min(10, N)  # 可根据需要调整样本数
    global_node_importance = np.zeros(node)
    global_time_importance = np.zeros(seq_len)
    for idx in range(num_global_samples):
        exp = explainer.explain_instance(input_flat[idx], lime_predict, num_features=node*seq_len)
        all_weights = exp.local_exp[1]
        weights = np.zeros(node * seq_len)
        for idx_weight in all_weights:
            i, w = idx_weight
            weights[i] = w
        weights_matrix = weights.reshape(node, seq_len)
        global_node_importance += np.abs(weights_matrix).sum(axis=1)
        global_time_importance += np.abs(weights_matrix).sum(axis=0)
    global_node_importance /= num_global_samples
    global_time_importance /= num_global_samples
    print('[全局LIME] node_importance:', global_node_importance)
    print('[全局LIME] time_importance:', global_time_importance)

    # 全局节点重要性条形图
    plt.figure(figsize=(8,4))
    plt.bar(range(node), global_node_importance)
    plt.xlabel('Node')
    plt.ylabel('Global LIME Importance (sum over time)')
    plt.title('Global Node-wise LIME Importance')
    plt.tight_layout()
    plt.savefig(f'./analysis/{args.dataset}_lime_global_node_importance.png')
    plt.show()

    # 全局空间热力图（pyHeatMap）
    if args.dataset == 'PEMS-BAY':
        csv_path = './data/PEMS-BAY/graph_sensor_locations_bay.csv'
        loc_df = pd.read_csv(csv_path, header=None)
        lats = loc_df[1].values[:node]
        lons = loc_df[2].values[:node]
    elif args.dataset == 'ST-EVCDP':
        csv_path = './data/ST-EVCDP/information.csv'
        loc_df = pd.read_csv(csv_path)
        lons = loc_df['lon'].values[:node]
        lats = loc_df['la'].values[:node]
    else:
        raise ValueError(f"Unknown dataset for location info: {args.dataset}")
    # 需要安装pyHeatMap: pip install pyHeatMap
    # 经纬度归一化到像素坐标
    img_size = (600, 600)
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    def to_pixel(lon, lat):
        x = int((lon - lon_min) / (lon_max - lon_min) * (img_size[0] - 1))
        y = int((lat - lat_min) / (lat_max - lat_min) * (img_size[1] - 1))
        return [x, img_size[1] - 1 - y]  # y轴反向
    # 对节点重要性做对数归一化，使热力分布更均匀
    norm_importance = np.log1p(global_node_importance / global_node_importance.max() * 10)
    # 生成points时用norm_importance
    points = []
    for lon, lat, value in zip(lons, lats, norm_importance):
        pt = to_pixel(lon, lat)
        points.append(pt + [float(value)])
    # 生成热力图
    hm = HeatMap(points, width=img_size[0], height=img_size[1])
    heatmap_img = hm.heatmap()
    heatmap_img.save(f'./analysis/{args.dataset}_lime_global_node_spatial_heatmap_pyheatmap.png')
    # 用matplotlib显示
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap_img)
    plt.axis('off')
    plt.title('Global Spatial LIME Node Importance (pyHeatMap)')
    plt.tight_layout()
    plt.savefig(f'./analysis/{args.dataset}_lime_global_time_importance.png')
    plt.show()

    # 全局时间步重要性条形图
    plt.figure(figsize=(8,4))
    plt.bar(range(seq_len), global_time_importance)
    plt.xlabel('Time Step')
    plt.ylabel('Global LIME Importance (sum over nodes)')
    plt.title('Global Time-step-wise LIME Importance')
    plt.tight_layout()
    plt.savefig(f'./analysis/{args.dataset}_lime_global_time_importance.png')
    plt.show()

    # --- 逐未来时间步的全局时间重要性条形图 ---
    for t in range(seq_len):
        step_time_importance = np.zeros(seq_len)
        for idx in range(num_global_samples):
            exp = explainer.explain_instance(input_flat[idx], lime_predict, num_features=node*seq_len)
            all_weights = exp.local_exp[1]
            weights = np.zeros(node * seq_len)
            for idx_weight in all_weights:
                i, w = idx_weight
                weights[i] = w
            weights_matrix = weights.reshape(node, seq_len)
            # 所有节点在输入每个时间步对预测第t步的贡献
            step_time_importance += np.abs(weights_matrix).sum(axis=0)
        step_time_importance /= num_global_samples
        plt.figure(figsize=(8,4))
        bars = plt.bar(range(seq_len), step_time_importance)
        # 重要性前三名标红
        top3_idx = np.argsort(step_time_importance)[-3:]
        for idx in top3_idx:
            bars[idx].set_color('red')
        plt.xlabel('Input Time Step')
        plt.ylabel(f'LIME Importance for Output Step {t+1}')
        plt.title(f'Global Time-step-wise LIME Importance for Output Step {t+1}')
        plt.tight_layout()
        plt.savefig(f'./analysis/{args.dataset}_lime_global_time_importance_step{t+1}.png')
        plt.show()

        out_path = f'./analysis/{args.dataset}_lime_global_node_spatial_heatmap_pyheatmap_step{t+1}.png'
        heatmap_img.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIME analysis for WGAK model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to WGAK model .pt file')
    parser.add_argument('--adj_path', type=str, required=True, help='Path to adjacency matrix .pt file')
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
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample to explain')
    args = parser.parse_args()
    main(args) 