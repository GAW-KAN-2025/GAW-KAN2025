import argparse
from tools.train import train

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument received.")
    parser.add_argument('--model_name', type=str, default="KAN", help='Choose the model to be executed.')
    parser.add_argument('--use_cuda', type=str2bool, default=True, help='Decide whether u wanna use cuda or not.')
    parser.add_argument('--cuda_device', type=int, default=0, help='The index of cuda u wanna use.')
    parser.add_argument('--seq_len', type=int, default=12, help='An integer argument to determine the length of input sequence.')
    parser.add_argument('--pre_len', type=int, default=1)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--p_epoch', type=int, default=200)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default="ST-EVCDP")
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--is_pre_train', type=str2bool, default=True)
    parser.add_argument('--random_seed', type=int, default=2025)
    parser.add_argument('--mode', type=str, default='completed')
    parser.add_argument('--use_bspline', type=str2bool, default=False, help='Whether to use B-spline in KAN/STAN.')
    args = parser.parse_args()
    train(args)