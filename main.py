import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import loss

from tqdm import tqdm
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from importlib import import_module

from data.BabyChillanto import BabyChillantoDataset
from data.DataCleaner import find_mistakes

    
def train(args, model, dataset):
    criterion = loss.Loss(args)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_loader = DataLoader(dataset.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=args.batch_size, shuffle=False)

    best_model = None
    best_performance = float('-inf') 
    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_data in train_loader:
            b_audios, b_labels, b_ids = batch_data
            optimizer.zero_grad()
            outputs = model(b_audios)
            loss = criterion(outputs, b_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        total_loss = 0
        correct = 0
        id2wrongCount = {id : 0 for id in dataset.val_dataset.data['id'].tolist()}
        wrong_ids = []
        with torch.no_grad():
            for vb_audios, vb_labels, vb_ids in val_loader:
                # Compute predictions and loss
                voutputs = model(vb_audios)
                loss = criterion(voutputs, vb_labels)
                total_loss += loss.item()
                
                pred = voutputs.argmax(dim=1, keepdim=True)
                correct += pred.squeeze(1).eq(vb_labels).sum().item()
                correct_indices = torch.nonzero(pred.squeeze(1).eq(vb_labels)).squeeze()
                mistake_indices = torch.nonzero(torch.ne(pred.squeeze(1), vb_labels)).squeeze()
                #wrong_ids = [vb_ids[mistake_indices]] if len(mistake_indices) == 1 else vb_ids[mistake_indices].tolist()
                if len(mistake_indices.shape) != 0:
                    wrong_ids.extend(vb_ids[mistake_indices].tolist())
                    
                for id in wrong_ids:
                    id2wrongCount[id] += 1
                #wrong_ids = torch.tensor([elem for i, elem in enumerate(vb_ids) if i not in correct_indices])
                #print(torch.unique(wrong_ids))
                
        avg_loss = total_loss / len(val_loader)
        val_accuracy = 100. * correct / len(dataset.val_dataset)

        # Check if current model has better performance than previous best model
        if val_accuracy > best_performance:
            best_performance = val_accuracy
            best_model = model.state_dict()  # Save the model's state dictionary
            torch.save(best_model, args.checkpoint_path)
            
        print(f"Epoch [{epoch+1}/{num_epochs}]: Traininn loss = {running_loss / len(train_loader): .3f}, \
            Val Accuracy = {val_accuracy: .3f}, {correct}/{len(dataset.val_dataset)}, Best Val Accuracy = {best_performance: .3f}")
    print(id2wrongCount)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--labels', nargs='+', default=['asphyxia', 'deaf', 'hunger', 'normal', 'pain'], help='List of labels')
    parser.add_argument('--data_dir', type=str, default='dataset/BabyChillanto/', help='Path to data directory')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/best_model.pth', help='Path to checkpoint file') 
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='Load the checkpoint file or not') 
    parser.add_argument('--pattern', type=str, default=r'^\d+', help='Regex pattern of id extraction')
    parser.add_argument('--split_type', type=str, default='class', help='Type of data split')
    parser.add_argument('--n_mfcc_coeffs', type=int, default=20, help='Number of MFCC coefficients')
    parser.add_argument('--seg_len', type=int, default=4, help='Segment length of audio')
    parser.add_argument('--shift', type=int, default=1, help='Shift length of each segment')
    parser.add_argument('--val_rate', type=float, default=0.2, help='Validation data ratio')
    parser.add_argument('--n_dim', type=int, default=128, help='Transformer dimension')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save output')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load input')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--loss', type=str, default='1*cross_entropy', help='Loss function combination')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss function')
    parser.add_argument('--model', type=str, default='mlp', help='Model name')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Parse the arguments
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    dataset = BabyChillantoDataset(args.data_dir, args.labels, args.seg_len, args.n_mfcc_coeffs, args.val_rate, \
        args.split_type, args.shift, args.save_path, args.load_path)

    module = getattr(import_module('models.' + args.model), args.model)
    
    if args.model == 'mlp':
        model = module(dataset.train_dataset.n_flatten_feats, 2000, 64, len(args.labels))
    elif args.model == 'mlp_attention':
        model = module(dataset.train_dataset.n_flatten_feats, 2000, 64, len(args.labels))
    else:
        model = module(args)
        
    if torch.cuda.is_available():
        model = model.cuda()
    
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint_path))
        
    
    train(args, model, dataset)
    
    
    