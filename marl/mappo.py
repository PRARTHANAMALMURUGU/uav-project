import torch
import torch.optim as optim
import numpy as np
import config

class MAPPO:

    def __init__(self, state_dim, action_dim):
        from marl.actor import Actor
        from marl.critic import Critic

        self.actor = Actor(state_dim, action_dim).to(config.DEVICE)
        self.critic = Critic(config.GLOBAL_STATE_DIM).to(config.DEVICE)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.eps_clip = 0.2

    def select_action(self, state):

        state = torch.tensor(state, dtype=torch.float32).to(config.DEVICE)

        mean, std = self.actor(state)
        std = torch.clamp(std, 0.05, 0.3)

        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.detach().cpu().numpy(), log_prob.detach()

    def update(self, states, actions, old_log_probs, rewards, global_states):

        states = torch.tensor(np.array(states), dtype=torch.float32).to(config.DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(config.DEVICE)
        global_states = torch.tensor(np.array(global_states), dtype=torch.float32).to(config.DEVICE)

        T, N, D = states.shape

        states = states.view(T * N, D)
        actions = actions.view(T * N, -1)

        old_log_probs = torch.stack(old_log_probs).to(config.DEVICE).view(T * N)

        # ===== GLOBAL STATE PROCESS =====
        global_states = global_states.unsqueeze(1).repeat(1, N, 1)
        global_states = global_states.view(T * N, -1)

        # ===== RETURNS =====
        returns = []
        R = np.zeros(N)

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(config.DEVICE)
        returns = returns.view(T * N)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        old_log_probs = old_log_probs.detach()

        # ===== PPO =====
        for _ in range(10):

            mean, std = self.actor(states)
            std = torch.clamp(std, 0.05, 0.3)

            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            values = self.critic(global_states).squeeze()

            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            entropy = dist.entropy().sum(dim=1).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            critic_loss = (returns - values).pow(2).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_opt.step()