import numpy as np
import pickle
import random
import os
import torch

from datetime import datetime

def flatten_list(lst: list):
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

def center_data(x: np.ndarray):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def preprocess_args(args):
    args.seeds = [int(s) for s in args.seeds.split('+')]
    if args.log_path is None:
        args.log_path = args.checkpoint_path
    args.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.checkpoint_dir = os.path.join('./checkpoint', args.timestamp)
    os.mkdir(args.checkpoint_dir)
    
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True