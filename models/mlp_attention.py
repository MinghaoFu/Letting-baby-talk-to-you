import torch
import torch.nn as nn
import torch.nn.functional as F

    
class mlp_attention(nn.Module):
    def __init__(self, in_feats, mid_feats, out_feats, n_classes):
        super(mlp_attention, self).__init__()
        self.fc1 = nn.Linear(in_feats, mid_feats)
        self.att1 = nn.Linear(in_feats, mid_feats)
        self.fc2 = nn.Linear(mid_feats, out_feats)
        self.att2 = nn.Linear(mid_feats, out_feats)
        self.fc3 = nn.Linear(out_feats, n_classes)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc4 = nn.Linear(out_feats, out_feats)
        self.fc5 = nn.Linear(out_feats, out_feats)
        

    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1, end_dim=len(x.shape) - 1)
        x1 = self.fc1(x)
        att1 = self.softmax(self.att1(x))
        x1 = self.relu(x1 * att1)
        x2 = self.fc2(x1) 
        att2 = self.softmax(self.att2(x1))
        x2 = self.relu(x2 * att2)
        x3 = self.fc3(x2)

        return x3
    
    def forward_constrative(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1, end_dim=len(x.shape) - 1)
        x1 = self.fc1(x)
        att1 = self.softmax(self.att1(x))
        x1 = self.relu(x1 * att1)
        x2 = self.fc2(x1) 
        att2 = self.softmax(self.att2(x1))
        x2 = self.relu(x2 * att2)
        x2 = self.fc4(x2)
        x2 = F.relu(x2)
        x2 = self.fc5(x2)
        # Normalize to unit hypersphere
        x2 = F.normalize(x2, dim=1)

        return x2

