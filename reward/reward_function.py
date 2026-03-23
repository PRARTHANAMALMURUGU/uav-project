import config
import numpy as np

def compute_reward(users, sinr):

    reward = 0

    # normalize SINR
    sinr_norm = max(0, sinr) / 30

    for u in users:

        if u.type == "rescue":
            reward += config.W_RESCUE * sinr_norm

        elif u.type == "victim":
            reward += config.W_VICTIM * sinr_norm

        else:
            reward += config.W_CIVILIAN * sinr_norm

    return reward