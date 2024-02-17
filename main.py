import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn import functional as F
import numpy as np

from loss import Loss

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from importlib import import_module

from utils import fix_seed, preprocess_args, AverageMeter, accuracy, Log
from train import train_babychillanto
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Descriptions')

    parser.add_argument('--data', type=str, default='BabyChillanto', help='Select dataset')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint file')  
    parser.add_argument('--log_path', type=str, default=None, help='Path to logging file')  
    parser.add_argument('--pattern', type=str, default=r'^\d+', help='Regex pattern of id extraction')
    parser.add_argument('--split_type', type=str, default='class', help='Type of data split')
    parser.add_argument('--n_mfcc_coeffs', type=int, default=20, help='Number of MFCC coefficients')
    parser.add_argument('--seg_len', type=int, default=4, help='Segment length of audio')
    parser.add_argument('--shift', type=float, default=1, help='Shift length of each segment')
    parser.add_argument('--val_rate', type=float, default=0.2, help='Validation data ratio')
    parser.add_argument('--n_dim',type=int, default=128, help='Transformer dimension')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save output')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load input')
    parser.add_argument('--seeds', type=str, default='42', help='Random seeds (for multiple training)')
    parser.add_argument('--seg_save_dir', type=str, default=None, help='Saving segmented audio files')
    parser.add_argument('--reweight', action='store_true', help='Reweighting')
    parser.add_argument('--add_labels', type=str, default='', help='Added labels')
    parser.add_argument('--center_data', type=bool, default=False, help='Center data for training or not')
    
    parser.add_argument('--loss', type=str, default='1*cross_entropy_loss+1*supervised_contrastive_loss', help='Loss function combination')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss function')
    parser.add_argument('--model', type=str, default='mlp', help='Model name')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    
    parser.add_argument('--gender_embed_dim', type=int, default=512, help='Embedding dim of gender') 
    parser.add_argument('--age_embed_dim', type=int, default=512, help='Embedding dim of age')
    
    args = parser.parse_args()
    

preprocess_args(args)
log = Log(args)
val_accuracies = []
test_accuracies = []
for seed in args.seeds:
    fix_seed(seed)
    
    data_module = getattr(import_module('data.' + args.data), args.data)
    dataset = data_module(args)
    if args.reweight:
        args.cls_weights = 1.0 / torch.Tensor(dataset.cls_num).cuda().float()
    
    module = getattr(import_module('models.' + args.model), args.model)
    args.in_feats = dataset.train_dataset.n_flatten_feats
    
    if args.model == 'mlp':
        model = module(args.in_feats, 2000, 64, len(args.labels), args.dropout)
    elif args.model == 'mlp_attention':
        model = module(args.in_feats, 64, 64, len(args.labels))
    elif args.model == 'Transformer_model':
        model = module(args.in_feats, args.n_dim, len(args.labels))
    else:
        model = module(args)
        
    if torch.cuda.is_available():
        model = model.cuda()
    
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
    
    trainer = getattr(import_module('train'), 'train_babychillanto')
    best_model, best_val_performance, best_test_performance = trainer(args, model, dataset, log, seed)
    val_accuracies.append(best_val_performance)
    test_accuracies.append(best_test_performance)
    
print('Val accuracies: {}, mean: {}, var: {}'.format(val_accuracies, np.mean(val_accuracies), np.var(val_accuracies)))
print('Test accuracies: {}, mean: {}, var: {}'.format(test_accuracies, np.mean(test_accuracies), np.var(test_accuracies)))