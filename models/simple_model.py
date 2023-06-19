import torch
import torch.nn as nn
import torch.nn.functional as F

    
class MLP(nn.Module):
    def __init__(self, in_feats, mid_feats, out_feats, n_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feats, mid_feats)
        self.att1 = nn.Linear(in_feats, mid_feats)
        self.fc2 = nn.Linear(mid_feats, out_feats)
        self.att2 = nn.Linear(mid_feats, out_feats)
        self.fc3 = nn.Linear(out_feats, n_classes)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x1 = self.fc1(x)
        att1 = self.softmax(self.att1(x))
        x1 = self.relu(x1 * att1)
        x2 = self.fc2(x1) 
        att2 = self.softmax(self.att2(x1))
        x2 = self.relu(x2 * att2)
        x3 = self.fc3(x2)
        x3 = self.softmax(x3)
        return x3

class TransformerModel(nn.Module):
    def __init__(self, in_dim, n_dim, n_classes):
        super(TransformerModel, self).__init__()
        self.fc = nn.Linear(in_dim, n_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(n_dim, n_classes)

        
    def forward(self, x):
        h = self.fc(x)
        y = self.transformer_encoder(h)
        y = y.mean(dim=1)
        out = self.classifier(y)
        return out