import torch
import torch.nn as nn

class MelGanClassify(nn.Module):
    def __init__(self, ) -> None:
        super(MelGanClassify).__init__()
        self.melgan = torch.hub.load('seungwonpark/melgan', 'melgan').cuda()
        self.fc = nn.Linear()
        
    def forward(self, mel):
        audio = self.melgan.inference(mel)
        return audio
        
    def pad1D(self, x, dim):
        pass