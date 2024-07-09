import torch.nn as nn
from models.encoder import Encoder
from models.projector import Projector


class VICReg(nn.Module):
    def __init__(self, D=128, proj_dim=512, device='cuda'):
        super(VICReg, self).__init__()
        self.encoder = Encoder(D=D, device=device)
        self.projector = Projector(D=D, proj_dim=proj_dim)
        self.training = True

    def forward(self, x):
        x = self.encoder.encode(x)
        if self.training:
            x = self.projector(x)
        return x

    def encode(self, x):
        return self.encoder.encode(x)
