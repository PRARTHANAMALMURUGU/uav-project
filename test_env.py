import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import config
import sys

sys.path.append(".")

from envi.environment import UAVEnvironment


env = UAVEnvironment()

steps = 100

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for step in range(steps):

    actions = []

    for _ in range(config.NUM_UAVS):

        dx = np.random.uniform(-0.5,0.5)
        dy = np.random.uniform(-0.5,0.5)
        dz = np.random.uniform(-0.2,0.2)

        actions.append((dx,dy,dz))

    sinr = env.step(actions)

    ax.clear()

    # plot UAVs
    for uav in env.uavs:
        ax.scatter(uav.x, uav.y, uav.z,
                   c='red',
                   s=120,
                   marker='^',
                   label='UAV')

    # plot users (on ground)
    for user in env.users:

        if user.type == "rescue":
            ax.scatter(user.x, user.y, 0, c='blue', s=30)

        elif user.type == "victim":
            ax.scatter(user.x, user.y, 0, c='yellow', s=30)

        else:
            ax.scatter(user.x, user.y, 0, c='white', edgecolors='black', s=30)

    ax.set_xlim(0, config.GRID_SIZE)
    ax.set_ylim(0, config.GRID_SIZE)
    ax.set_zlim(0, 12)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Altitude")

    ax.set_title(f"3D UAV Disaster Network | Step {step} | SINR {sinr:.2f}")

    plt.pause(0.1)


plt.ioff()
plt.show()