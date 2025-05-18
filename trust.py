import numpy as np
import torch
import os
import pickle
import pandas as pd 
from module.WGAK import WGAK
from tools import data_switcher
def predict_with_model(model, x_data, occ_data=None, device='cpu'):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
        if occ_data is not None:
            occ_tensor = torch.tensor(occ_data, dtype=torch.float32).to(device)
            pred = model(x_tensor, occ=occ_tensor)
        else:
            pred = model(x_tensor)
    return pred.cpu().numpy()

def bootstrap_ci(model, x_data, occ_data=None, device='cpu', n_bootstrap=1000, alpha=0.05):
    N = x_data.shape[0]
    preds = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(N, N, replace=True)
        x_bs = x_data[idx]
        occ_bs = occ_data[idx] if occ_data is not None else None
        pred_bs = predict_with_model(model, x_bs, occ_bs, device)
        preds.append(pred_bs)
    preds = np.stack(preds, axis=0)  # (n_bootstrap, N, ...)
    lower = np.percentile(preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    mean = np.mean(preds, axis=0)
    return mean, lower, upper

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

# 用法示例（放在main函数后面，或单独脚本运行）
# 假设x_data, occ_data, model, device已准备好
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = 'ST-EVCDP'
if dataset == 'ST-EVCDP':
    node_num = 247
    # ST-EVCDP直接用data_switcher返回的adj_dense
    train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense = data_switcher.get_data_loaders(
        dataset, 12, 12, device, 512
    )
    adj = adj_dense
    # 加载模型或参数
    state = torch.load('./checkpoints/WGAK_ST-EVCDP_12_bs512_completed.pt', map_location=device)
    if isinstance(state, dict):
        model = WGAK(input_dim=12, output_dim=12, hidden_dim=64, num_layers=2, wavelet_type='db1', dwt_level=2, adj=adj, dropout=0.1)
        model.load_state_dict(state)
        model.to(device)
    else:
        model = state
        model.to(device)
else:
    node_num = 315
    # 其他数据集从文件加载adj
    adj = load_adj('./data/PEMS-BAY/adj_mx_bay.pkl')
    train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense = data_switcher.get_data_loaders(
        dataset, 12, 12, device, 512
    )
    # 加载模型或参数
    state = torch.load('./checkpoints/WGAK_PEMS-BAY_12_bs512_completed.pt', map_location=device)
    if isinstance(state, dict):
        model = WGAK(input_dim=12, output_dim=12, hidden_dim=64, num_layers=2, wavelet_type='db1', dwt_level=2, adj=adj, dropout=0.1)
        model.load_state_dict(state)
        model.to(device)
    else:
        model = state
        model.to(device)

mean_pred, lower_pred, upper_pred = bootstrap_ci(model, train_occupancy.reshape(-1, node_num, 12), train_price.reshape(-1, node_num, 12), device=device, n_bootstrap=200, alpha=0.05)
print('预测均值 shape:', mean_pred.shape)
print('95%置信区间下界 shape:', lower_pred.shape)
print('95%置信区间上界 shape:', upper_pred.shape)

# 可视化某个节点的置信区间
import matplotlib.pyplot as plt
node_idx = 0  # 选择节点
plt.plot(mean_pred[:, node_idx], label='mean prediction')
plt.fill_between(range(mean_pred.shape[0]), lower_pred[:, node_idx], upper_pred[:, node_idx], color='gray', alpha=0.3, label='95% CI')
plt.legend()
plt.xlabel('样本')
plt.ylabel('预测值')
plt.title(f'节点{node_idx}预测及置信区间')
plt.show()
