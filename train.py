import os
import argparse
import torch

from logger import utils
from data_loaders import get_data_loaders
from solver import train
from SVM import SVM, HingeLoss


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    # load model
    model = SVM(input_dim=args.model.input_size, output_dim=args.model.output_size)

    
    # load parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=args.train.lr)
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = args.train.lr
    #    param_group['weight_decay'] = args.train.weight_decay
            
    # loss
    loss_func = HingeLoss()

    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    loss_func.to(args.device)

    # datas
    loader_train, loader_valid = get_data_loaders(args)
    
    # run
    train(args, initial_global_step, model, optimizer, loss_func, loader_train, loader_valid)
    
