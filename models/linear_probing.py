import torch.nn as nn
import torch
from models.vicreg import VICReg
from models.projector import Projector


class LinearProbing(nn.Module):
    def __init__(self, D=128, proj_dim=512, device='cuda'):
        super(LinearProbing, self).__init__()
        self.vicreg = VICReg(D=128, proj_dim=512, device='cuda')
        self.linear = nn.Linear(D, 10)

    def forward(self, x):
        with torch.no_grad():  # Freeze the backbone weights
            x = self.vicreg(x)
            x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    def encode(self, x):
        return self.encoder.encode(x)