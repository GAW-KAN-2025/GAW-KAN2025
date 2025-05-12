import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tools import functions as fn

def get_data_loaders(data_name, seq_l, pre_l, device, bs):
    if data_name == "ST-EVCDP":
        train_occupancy, train_price, valid_occupancy, valid_price, test_occupancy,  test_price, adj_dense, train_dataset, test_dataset, valid_dataset = get_ST_EVCDP(seq_l, pre_l, device)
    elif data_name == "PEMS-BAY":
        train_occupancy, train_price, valid_occupancy, valid_price, test_occupancy,  test_price, adj_dense, train_dataset, test_dataset, valid_dataset = get_PEMS(seq_l, pre_l, device)
        train_occupancy = train_occupancy.reshape(-1, train_occupancy.shape[-1])[:5000,...]
        train_price = train_price.reshape(-1, train_price.shape[-1])[:5000,...]
    else: 
        raise FileNotFoundError(f"Unexisting dataset {data_name}!")

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_occupancy), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_occupancy), shuffle=False)
    return train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense

def get_ST_EVCDP(seq_l, pre_l, device):
    # input data
    occ, prc, adj, col, dis, cap, tim, inf = fn.read_dataset()
    adj_dense = torch.Tensor(adj)

    # dataset division
    train_occupancy, valid_occupancy, test_occupancy = fn.division(occ, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
    nodes = train_occupancy.shape[-1]
    train_price, valid_price, test_price = fn.division(prc, train_rate=0.6, valid_rate=0.2, test_rate=0.2)

    train_dataset = fn.CreateDataset(train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
    test_dataset = fn.CreateDataset(test_occupancy, test_price, seq_l, pre_l, device, adj_dense)
    valid_dataset = fn.CreateDataset(valid_occupancy, valid_price, seq_l, pre_l, device, adj_dense)
    
    return train_occupancy, train_price, valid_occupancy, valid_price, test_occupancy,  test_price, adj_dense, train_dataset, test_dataset, valid_dataset

def get_PEMS(seq_l, pre_l, device):
    with np.load('./data/PEMS-BAY/test.npz') as data:
        test_x = data['x']
        test_y = data['y']

    with np.load('./data/PEMS-BAY/train.npz') as data:
        train_x = data['x']
        train_y = data['y']

    with np.load('./data/PEMS-BAY/val.npz') as data:
        val_x = data['x']
        val_y = data['y']

    with open("./data/PEMS-BAY/adj_mx_bay.pkl", "rb") as file:
        adj = pickle.load(file, encoding='latin1')[2]
        adj_dense = torch.Tensor(adj)

    train_occupancy = train_x[:5000,...,0]
    train_price = train_x[:5000,...,1]
    train_label = train_y[:5000,...,0]
    valid_occupancy = train_x[5000:7000,...,0]
    valid_price = train_x[5000:7000,...,1]
    valid_label = train_y[5000:7000,...,0]
    test_occupancy = train_x[7000:10000,...,0]
    test_price = train_x[7000:10000,...,1]
    test_label = train_y[7000:10000,...,0]

    # ====== min-max归一化（以训练集为基准）======
    occ_min = train_occupancy.min()
    occ_max = train_occupancy.max()
    prc_min = train_price.min()
    prc_max = train_price.max()
    label_min = train_label.min()
    label_max = train_label.max()

    train_occupancy = (train_occupancy - occ_min) / (occ_max - occ_min + 1e-8)
    valid_occupancy = (valid_occupancy - occ_min) / (occ_max - occ_min + 1e-8)
    test_occupancy = (test_occupancy - occ_min) / (occ_max - occ_min + 1e-8)

    train_price = (train_price - prc_min) / (prc_max - prc_min + 1e-8)
    valid_price = (valid_price - prc_min) / (prc_max - prc_min + 1e-8)
    test_price = (test_price - prc_min) / (prc_max - prc_min + 1e-8)

    train_label = (train_label - label_min) / (label_max - label_min + 1e-8)
    valid_label = (valid_label - label_min) / (label_max - label_min + 1e-8)
    test_label = (test_label - label_min) / (label_max - label_min + 1e-8)
    # ========================================

    train_dataset = CreatePEMSDataset(train_occupancy, train_price, train_label, seq_l, pre_l, device, adj)
    test_dataset = CreatePEMSDataset(test_occupancy, test_price, test_label, seq_l, pre_l, device, adj)
    valid_dataset = CreatePEMSDataset(valid_occupancy, valid_price, valid_label, seq_l, pre_l, device, adj)

    return train_occupancy, train_price, valid_occupancy, valid_price, test_occupancy,  test_price, adj_dense, train_dataset, test_dataset, valid_dataset

class CreatePEMSDataset(Dataset):
    def __init__(self, occ, prc, label, lb, pt, device, adj):
        self.occ = torch.Tensor(occ)[:1000,:lb,:]
        self.prc = torch.Tensor(prc)[:1000,:lb,:]
        self.label = torch.Tensor(label)[:1000,:pt,:]
        self.device = device

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):  # occ: batch, seq, node
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_label = torch.transpose(self.label[idx, :, :], 0, 1).to(self.device)
        return output_occ, output_prc, output_label