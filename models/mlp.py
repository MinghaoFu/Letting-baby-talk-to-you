import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, in_feats, mid_feats, out_feats, n_classes, dropout=0.5):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(in_feats, mid_feats)
        self.fc2 = nn.Linear(mid_feats, out_feats)
        self.fc3 = nn.Linear(out_feats, n_classes)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1, end_dim=len(x.shape) - 1)
        if hasattr(self, 'dropout'):
            x1 = self.relu(self.dropout(self.fc1(x)))
        else:
            x1 = self.relu(self.fc1(x))
            
        if hasattr(self, 'dropout'):
            x2 = self.relu(self.dropout(self.fc2(x1)))
        else:
            x2 = self.relu(self.fc2(x1))
            
        x3 = self.fc3(x2)
        
        return x3

