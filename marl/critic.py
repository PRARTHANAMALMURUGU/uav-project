import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self,state_dim):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.ReLU(),

            nn.Linear(128,1)
        )


    def forward(self,x):

        return self.net(x)