import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from module import baselines
from tools import learner
from tools import functions as fn
from tools import data_switcher
import time
from module import KAN
from module import GAF, SGCTN, WaveSTFTGAT, waveletGAT, FreTimeFusion, FourierGAT, CoupFourGAT, CoupFourGAT_v2
from module import STAK
from module import SWAK
from module import WavKAN
from module import WGAK

def choose_model(model_name, seq_l, pre_l, adj_dense, device, node_num=None, use_bspline=False):
    adj_dense_cuda = adj_dense.to(device)
    adj_sparse = adj_dense.to_sparse_coo().to(device)
    if model_name == "PAG":
        model = baselines.PAG(seq=seq_l, pred_len=pre_l, a_sparse=adj_sparse).to(device)
    elif model_name == "VAR":
        model = baselines.VAR(node=node_num, seq=seq_l, pre_l=pre_l).to(device)
    elif model_name == "FCN":
        model = baselines.FCN(node=node_num, seq=seq_l, pre_l=pre_l).to(device)
    elif model_name == "LSTM":
        model = baselines.LSTM(seq_l, 2, node_num=node_num, pre_l=pre_l).to(device)
    elif model_name == "TransformerModel":
        model = baselines.TransformerModel(seq_l, 16, 8, 2, 1, 2, 16, 0.1, pre_l=pre_l).to(device)
    elif model_name == "GCN":
        model = baselines.GCN(seq_l, 2, adj_dense_cuda, pre_l=pre_l).to(device)
    elif model_name == "GAT":
        model = baselines.GAT(seq_l, 2, adj_dense_cuda, pre_l=pre_l).to(device)
    elif model_name == "STGCN":
        model = baselines.STGCN(seq_l, 2, adj_dense_cuda, pre_l=pre_l).to(device)
    elif model_name == "LstmGcn":
        model = baselines.LstmGcn(seq_l, 2, adj_dense_cuda, pre_l=pre_l).to(device)
    elif model_name == "LstmGat":
        model = baselines.LstmGat(seq_l, 2, adj_dense_cuda, adj_sparse, pre_l=pre_l).to(device)
    elif model_name == "HSTGCN":
        model = baselines.HSTGCN(seq_l, 2, adj_dense_cuda, adj_dense_cuda, pre_l=pre_l).to(device)
    elif model_name == "TPA":
        model = baselines.TPA(seq_l, 2, node_num, pre_l=pre_l).to(device)
    elif model_name == "FGN":
        model = baselines.FGN(pre_length=pre_l, seq_length=seq_l).to(device)
    elif model_name == "KAN":
        model = KAN.KAN(input_dim=seq_l, output_dim=pre_l, degree=3, use_bspline=use_bspline, n_basis=8).to(device)
    elif model_name == "GAF":
        model = GAF.GATWithFourier(seq=seq_l, n_fea=2, adj_dense=adj_dense_cuda, pre_L=pre_l).to(device)
    elif model_name == "SGCTN":
        model = SGCTN.SGCTN(adj_matrix=adj_dense_cuda, lap_matrix=adj_dense_cuda, num_nodes=node_num, feat_dim=2, pre_L=pre_l).to(device)
    elif model_name == "WaveSTFTGAT":
        model = WaveSTFTGAT.WaveSTFTGAT(nfeat=seq_l, nhid=8, nclass=pre_l, dropout=0.1, alpha=0.2, nheads=2, adj=adj_dense_cuda, num_nodes=node_num, pre_L=pre_l).to(device)
    elif model_name == "waveletGAT":
        model = waveletGAT.WaveletGAT(nfeat=seq_l, nhid=8, nclass=pre_l, dropout=0.1, alpha=0.2, nheads=2, adj=adj_dense_cuda, num_nodes=node_num, pre_L=pre_l).to(device)
    elif model_name == "FreTimeFusion":
        model = FreTimeFusion.CoupFourGAT(nfeat=seq_l, nhid=8, nclass=pre_l, dropout=0.1, alpha=0.2, nheads=2, adj=adj_dense_cuda, num_nodes=node_num, pre_L=pre_l).to(device)
    elif model_name == "FourierGAT":
        model = FourierGAT.GAT_Fourier(nfeat=seq_l, nhid=8, nclass=pre_l, dropout=0.1, alpha=0.2, nheads=2, adj=adj_dense_cuda, num_nodes=node_num, pre_L=pre_l).to(device)
    elif model_name == "CoupFourGAT":
        model = CoupFourGAT.CoupFourGAT(nfeat=seq_l, nhid=8, nclass=pre_l, dropout=0.1, alpha=0.2, nheads=2, adj=adj_dense_cuda, num_nodes=node_num, pre_L=pre_l).to(device)
    elif model_name == "CoupFourGAT_v2":
        model = CoupFourGAT_v2.CoupFourGAT(nfeat=seq_l, nhid=8, nclass=pre_l, dropout=0.1, alpha=0.2, nheads=2, adj=adj_dense_cuda, num_nodes=node_num, pre_L=pre_l).to(device)
    elif model_name == "STAK":
        model = STAK.STAK(input_dim=seq_l, output_dim=pre_l, adj=adj_dense_cuda, use_bspline=use_bspline).to(device)
    elif model_name == "SWAK":
        model = SWAK.SWAK(input_dim=seq_l, output_dim=pre_l, degree=3, use_bspline=use_bspline, n_basis=8, adj=adj_dense_cuda).to(device)
    elif model_name == "WavKAN":
        model = WavKAN.WavKAN(input_dim=seq_l, output_dim=pre_l).to(device)
    elif model_name == "WGAK":
        model = WGAK.WGAK(input_dim=seq_l, output_dim=pre_l, adj=adj_dense_cuda).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model