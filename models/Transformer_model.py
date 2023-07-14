import torch
import torch.nn as nn

class Transformer_model(nn.Module):
    def __init__(self, in_dim, n_dim, n_classes):
        super(Transformer_model, self).__init__()
        self.fc = nn.Linear(in_dim, n_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(n_dim, n_classes)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1, end_dim=len(x.shape) - 1)
        h = self.fc(x)
        y = self.transformer_encoder(h)
        out = self.classifier(y)
        return out