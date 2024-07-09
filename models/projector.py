import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )

    def forward(self, x):
        return self.model(x)