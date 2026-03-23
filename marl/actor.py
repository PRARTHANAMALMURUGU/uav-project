import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self,state_dim,action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_dim)
        )

    def forward(self,x):

        return torch.tanh(self.net(x))