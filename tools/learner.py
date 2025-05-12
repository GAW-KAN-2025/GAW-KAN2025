import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tools.functions as fn
import copy
from tqdm import tqdm

# 
def physics_informed_meta_learning(law_list, global_model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense):
    support_occ, query_occ = fn.meta_division(train_occupancy, support_rate=0.5, query_rate=0.5)
    support_prc, query_prc = fn.meta_division(train_price, support_rate=0.5, query_rate=0.5)

    # pre-training data generation
    n_laws = len(law_list)
    support_dataset_dict = dict()
    query_dataset_dict = dict()
    support_dataloader_dict = dict()
    query_dataloader_dict = dict()
    for n in range(n_laws):
        support_dataset_dict[n] = fn.PseudoDataset(support_occ, support_prc, seq_l, pre_l, device, adj_dense, law_list[n])
        query_dataset_dict[n] = fn.PseudoDataset(query_occ, query_prc, seq_l, pre_l, device, adj_dense, law_list[n])
        support_dataloader_dict[n] = DataLoader(support_dataset_dict[n], batch_size=bs, shuffle=True, drop_last=True)
        query_dataloader_dict[n] = DataLoader(query_dataset_dict[n], batch_size=query_occ.shape[0], shuffle=False)

    # meta-learning process    
    torch.save(global_model, './checkpoints' + '/meta_' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt')
    loss_function = torch.nn.MSELoss()
    # outer loop
    global_model.train()
    for epoch in tqdm(range(p_epoch), desc='Pre-training'):
        query_loss = 100
        global_grads = fn.zero_init_global_gradient(global_model)

        # inner loop
        for n in range(n_laws):
            temp_model = torch.load('./checkpoints' + '/meta_' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt').to(device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), weight_decay=0.00001)
            temp_model.train()
            # support
            for j, data in enumerate(support_dataloader_dict[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, pseudo_price, pseudo_label = data
                mix_ratio = (j+1) * occupancy.shape[0] / len(train_occupancy)
                mix_prc = fn.data_mix(price, pseudo_price, mix_ratio)
                mix_label = fn.data_mix(label, pseudo_label, mix_ratio)
                temp_optimizer.zero_grad()
                predict = temp_model(occupancy, mix_prc)
                loss = loss_function(predict, mix_label)
                loss.backward()
                temp_optimizer.step()
            # query
            for j, data in enumerate(query_dataloader_dict[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, pseudo_price, pseudo_label = data
                temp_optimizer.zero_grad()
                predict = temp_model(occupancy, price)
                loss = loss_function(predict, label)
                loss.backward()
                for name, param in temp_model.named_parameters():
                    if  param.grad is not None:
                        global_grads[name] += param.grad

        # global updating: BGD
        for name, param in global_model.named_parameters():
            param = param - 0.02 * global_grads[name] / n_laws

        if query_loss > loss:
            loss = query_loss
            torch.save(global_model, './checkpoints' + '/meta_' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt')
    
    return global_model


def fast_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense):
    n_laws = len(law_list)
    fast_datasets = dict()
    fast_loaders = dict()
    for n in range(n_laws):
        fast_datasets[n] = fn.CreateFastDataset(train_occupancy, train_price, seq_l, pre_l, law_list[n], device, adj_dense)
        fast_loaders[n] = DataLoader(fast_datasets[n], batch_size=bs, shuffle=True, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)
    loss_function = torch.nn.MSELoss()
    for epoch in tqdm(range(p_epoch), desc='Pre-training'):
        for n in range(n_laws):
            for j, data in enumerate(fast_loaders[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                predict = model(occupancy, prc_ch)
                loss = loss_function(predict, label_ch)
                loss.backward()
                optimizer.step()

            for j, data in enumerate(fast_loaders[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                predict = model(occupancy, prc_ch)
                loss = loss_function(predict, label_ch)
                loss.backward()
                optimizer.step()

    return model

def fast_diff_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense):
    n_laws = len(law_list)
    fast_datasets = dict()
    fast_loaders = dict()
    for n in range(n_laws):
        fast_datasets[n] = fn.CreateFastDataset(train_occupancy, train_price, seq_l, pre_l, law_list[n], device, adj_dense)
        fast_loaders[n] = DataLoader(fast_datasets[n], batch_size=bs, shuffle=True, drop_last=True)
    
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)
    loss_function = torch.nn.MSELoss()

    def train_batch(x_0):
        noise, pred_noise = pred_diff(x_0)
        
        # 计算损失
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()
        return loss

    def pred_diff(x_0):
        # x_0形状: (512,247,12)
        batch_size = x_0.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, T, (batch_size,))
        alpha_t = alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        
        # 正向加噪
        noise = torch.randn((x_0.shape[0],x_0.shape[1],1)).to(device)
        x_t = torch.sqrt(alpha_t).to(device) * x_0 + torch.sqrt(1 - alpha_t).to(device) * noise
        t = t.to(device)
        
        # 预测噪声
        pred_noise = model(x_t, t)
        return noise, pred_noise

    for epoch in tqdm(range(p_epoch), desc='Pre-training'):
        for n in range(n_laws):
            for j, data in enumerate(fast_loaders[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                loss = train_batch(occupancy)

            for j, data in enumerate(fast_loaders[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                loss = train_batch(occupancy)

    return model

