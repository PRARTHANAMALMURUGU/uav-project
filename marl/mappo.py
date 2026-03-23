import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config

from marl.actor import Actor
from marl.critic import Critic


class MAPPO:

    def __init__(self, state_dim, action_dim):

        self.actor = Actor(state_dim, action_dim).to(config.DEVICE)
        self.critic = Critic(state_dim).to(config.DEVICE)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.LR)

        self.gamma = config.GAMMA


    def select_action(self, state):

        state = torch.tensor(state, dtype=torch.float32).to(config.DEVICE)

        action = self.actor(state)

        return action.detach().cpu().numpy()


    def update(self, states, rewards):

        # ===== Convert to tensors =====
        states = torch.tensor(np.array(states), dtype=torch.float32).to(config.DEVICE)

        # ===== Compute returns =====
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(config.DEVICE)

        # ===== Critic forward =====
        values = self.critic(states).squeeze()

        # ===== Advantage =====
        advantage = returns - values

        # =========================
        # Critic update
        # =========================
        critic_loss = (returns - values).pow(2).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # =========================
        # Actor update (SAFE)
        # =========================

        # forward again to keep graph
        actions = self.actor(states)

        # IMPORTANT: detach advantage ONLY
        advantage_detached = advantage.detach()

        actor_loss = -(actions * advantage_detached.unsqueeze(1)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        