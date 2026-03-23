import numpy as np
import torch
import config

from envi.environment import UAVEnvironment
from marl.mappo import MAPPO
from reward.reward_function import compute_reward


env=UAVEnvironment()

agent=MAPPO(config.STATE_DIM,config.ACTION_DIM)

episodes=200


for ep in range(episodes):

    states=[]
    rewards=[]

    for step in range(config.MAX_STEPS):

        state=env.get_state()

        action=agent.select_action(state)

    actions = []

    for i in range(config.NUM_UAVS):
        a = agent.select_action(state)
        actions.append(a)
        sinr=env.step(actions)

        reward=compute_reward(env.users,sinr)

        states.append(state)
        rewards.append(reward)

    agent.update(states,rewards)

    print("Episode",ep,"Reward",sum(rewards))
    if ep  == 199:
        torch.save(agent.actor.state_dict(), f"saved_models/mappo_actor_ep{ep}.pth")