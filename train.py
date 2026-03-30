import numpy as np
import torch
import config

from envi.environment import UAVEnvironment
from marl.mappo import MAPPO
from reward.reward_function import compute_reward

env = UAVEnvironment()
agent = MAPPO(config.STATE_DIM, config.ACTION_DIM)

episodes = 1000

for ep in range(episodes):

    states = []
    global_states = []
    actions_list = []
    log_probs_list = []
    rewards = []

    for step in range(config.MAX_STEPS):

        states_per_uav = env.get_state()   # (num_uavs, state_dim)
        global_state = env.get_global_state()

        actions = []
        log_probs = []

        for i in range(config.NUM_UAVS):
            action, log_prob = agent.select_action(states_per_uav[i])
            actions.append(action)
            log_probs.append(log_prob)

        sinrs, throughputs, latencies, assignments = env.step(actions)

        reward = compute_reward(
            env.users,
            assignments,
            sinrs,
            throughputs,
            latencies
        )

       
        # ===== store =====
        states.append(states_per_uav)
        global_states.append(global_state)
        actions_list.append(np.array(actions))
        log_probs_list.append(torch.stack(log_probs))
        rewards.append(reward)

    agent.update(states, actions_list, log_probs_list, rewards, global_states)

    avg_rewards = np.mean(rewards, axis=0)

    print(f"Episode {ep}")
    print("  Per UAV:", avg_rewards)
    print("  Mean:", np.mean(avg_rewards))

    if ep == episodes-1:
        torch.save(agent.actor.state_dict(), f"saved_models/final_actor.pth")
