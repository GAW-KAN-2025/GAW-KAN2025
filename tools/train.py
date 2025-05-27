import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools import learner
from tools import functions as fn
from tools import data_switcher
from tools import model_switcher
import time

def train(args):
    # system configuration
    use_cuda = args.use_cuda
    cuda_id = "cuda:" + str(args.cuda_device)
    device = torch.device(cuda_id if use_cuda and torch.cuda.is_available() else "cpu")
    fn.set_seed(seed=args.random_seed, flag=True)
    dataset=args.dataset
    torch.cuda.empty_cache()

    # hyper params
    model_name = args.model_name
    seq_l = args.seq_len
    pre_l = args.pre_len
    bs = 512
    p_epoch = 200
    n_epoch = 1000
    patience = 30  # early stopping patience
    law_list = np.array([-1.48, -0.74])  # price elasticities of demand for EV charging. Recommend: up to 5 elements.
    is_train = True
    mode = 'completed'  # 'simplified' or 'completed'
    is_pre_train = args.is_pre_train

    train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense = data_switcher.get_data_loaders(dataset, seq_l, pre_l, device, bs)

    node_num = train_occupancy.shape[1]  # 自动获取节点数
    adj_dense_cuda = adj_dense.to(device)
    adj_sparse = adj_dense.to_sparse_coo().to(device)

    # training setting
    model = model_switcher.choose_model(model_name, seq_l, pre_l, adj_dense, device=device, node_num=node_num, use_bspline=args.use_bspline)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.05,
        steps_per_epoch=len(train_loader),
        epochs=n_epoch
    )
    loss_function = torch.nn.MSELoss()

    # 文件名后缀，STAK模型时注明bspline
    bspline_tag = ''
    if model_name == 'STAK':
        bspline_tag = '_bspline' if args.use_bspline else '_nobspline'

    print(f"----Starting training {model_name} model with prediction horizen {pre_l}----")

    if is_train is True:
        model.train()
        if is_pre_train is True:
            if mode == 'simplified':
                model = learner.fast_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
            elif mode == 'completed':
                model = learner.physics_informed_meta_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
            else:
                print("Mode error, skip the pre-training process.")

        # Early stopping variables
        best_val_loss = float('inf')
        best_model = None
        counter = 0
        best_epoch = 0

        for epoch in tqdm(range(n_epoch), desc='Fine-tuning'):
            for j, data in enumerate(train_loader):
                model.train()
                occupancy, price, label = data
                optimizer.zero_grad()
                predict = model(occupancy, price)
                loss = loss_function(predict, label)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # validation
            model.eval()
            v_loss = 0.0
            for j, data in enumerate(valid_loader):
                occupancy, price, label = data
                predict = model(occupancy, price)
                loss = loss_function(predict, label)
                v_loss += loss.item()
            v_loss /= len(valid_loader)
            
            # Check if validation loss improved
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model = model.state_dict().copy()
                counter = 0
                best_epoch = epoch
                print(f"Epoch {epoch}: Validation loss improved to {v_loss:.6f}")
                model_path = './checkpoints' + '/' + model_name + '_' + dataset + '_seq' + str(seq_l) + '_pre' + str(pre_l) + '_bs' + str(bs) + '_' + mode + bspline_tag + '_seed' + str(args.random_seed) + '.pt'
                torch.save(model, model_path)
            else:
                counter += 1
                print(f"Epoch {epoch}: Validation loss did not improve. Counter: {counter}/{patience}")
                
            # Early stopping check
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                # Load the best model
                model.load_state_dict(best_model)
                break

    print(f"----Training finished!----")
    
    model = torch.load('./checkpoints' + '/' + model_name + '_' + dataset + '_seq' + str(seq_l) + '_pre' + str(pre_l) + '_bs' + str(bs) + '_' + mode + bspline_tag + '_seed' + str(args.random_seed) + '.pt')
    print(f"----Model was saved into folder: {'./checkpoints' + '/' + model_name + '_' + dataset + '_seq' + str(seq_l) + '_pre' + str(pre_l) + '_bs' + str(bs) + '_' + mode + bspline_tag + '_seed' + str(args.random_seed) + '.pt'}")
    # test
    model.eval()
    result_list = []
    predict_list = []
    label_list = []

    # Initialize lists to store time and memory usage
    time_list = []
    memory_list = []

    for j, data in enumerate(test_loader):
        occupancy, price, label = data  # occupancy.shape = [batch, seq, node]
        print('occupancy:', occupancy.shape, 'price:', price.shape, 'label:', label.shape)
        with torch.no_grad():
            # Start time measurement
            start_time = time.time()
            
            # Start memory tracking
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Synchronize before measuring
                memory_before = torch.cuda.memory_allocated()

            predict = model(occupancy, price)

            # End memory tracking
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_usage = memory_after - memory_before
                memory_list.append(memory_usage / (1024 * 1024))  # Convert bytes to MB
            
            # End time measurement
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_list.append(elapsed_time)
            predict_list.append(predict)
            label_list.append(label)

    predict_list = torch.cat(predict_list, dim=0).view(-1, pre_l).cpu().detach().numpy()
    label_list = torch.cat(label_list, dim=0).view(-1, pre_l).cpu().detach().numpy()

    result_list = []
    for i in range(pre_l):
        output = fn.metrics(
            test_pre=predict_list[:, i],
            test_real=label_list[:, i]
        )
        # 在每行开头加上horizon步数（从1开始）
        result_list.append([i+1] + output)
    result_df = pd.DataFrame(
        columns=['horizon', 'MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'],
        data=result_list
    )
    result_df.to_csv('./results' + '/' + model_name + '_' + dataset + '_seq' + str(seq_l) + '_pre' + str(pre_l) + 'bs' + str(bs) + bspline_tag + '_seed' + str(args.random_seed) + '.csv', encoding='gbk', index=False)

    # Print average time and memory usage
    print(f'Average time per prediction: {np.mean(time_list)} seconds')
    print(f'Average memory usage per prediction: {np.mean(memory_list)} MB')