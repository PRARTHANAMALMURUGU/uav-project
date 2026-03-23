import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import config

from envi.environment import UAVEnvironment
from marl.mappo import MAPPO


# ===== Load environment =====
env = UAVEnvironment()

# ===== Load trained model =====
agent = MAPPO(config.STATE_DIM, config.ACTION_DIM)

agent.actor.load_state_dict(
    torch.load("saved_models/mappo_actor_ep199.pth", map_location=config.DEVICE)
)

agent.actor.eval()

print("✅ Loaded trained model")

# ===== Setup plot =====
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# ===== Simulation loop =====
for step in range(100):

    state = env.get_state()

    action = agent.select_action(state)

    actions = [action for _ in range(config.NUM_UAVS)]

    sinr = env.step(actions)

    ax.clear()

    # ===== Plot UAVs =====
    for uav in env.uavs:
        ax.scatter(uav.x, uav.y, uav.z,
                   c='red', s=120, marker='^')

    # ===== Plot users =====
    for user in env.users:

        if user.type == "rescue":
            color = 'blue'
        elif user.type == "victim":
            color = 'yellow'
        else:
            color = 'white'

        ax.scatter(user.x, user.y, 0,
                   c=color, edgecolors='black', s=30)

    ax.set_xlim(0, config.GRID_SIZE)
    ax.set_ylim(0, config.GRID_SIZE)
    ax.set_zlim(0, 12)

    ax.set_title(f"Step {step} | SINR {sinr:.2f}")

    plt.pause(0.1)


plt.ioff()
plt.show()