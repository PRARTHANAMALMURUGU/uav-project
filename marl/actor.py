# marl/actor.py
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):

        x = self.net(x)

        mean = self.mean(x)

        # clamp mean (VERY IMPORTANT)
        mean = torch.clamp(mean, -1, 1)

        std = torch.clamp(torch.exp(self.log_std), 0.05, 0.3)

        return mean, std