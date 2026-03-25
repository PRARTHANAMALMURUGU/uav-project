import numpy as np
import config

class UAV:

    def __init__(self):

        self.x = np.random.uniform(0, config.GRID_SIZE)
        self.y = np.random.uniform(0, config.GRID_SIZE)
        self.z = np.random.uniform(config.UAV_ALT_MIN, config.UAV_ALT_MAX)

        self.battery = 100


    def move(self, action):

        dx, dy, dz = action

        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)
        dz = np.clip(dz, -0.5, 0.5)

        self.x += dx
        self.y += dy
        self.z += dz

        # bounds
        self.x = np.clip(self.x, 0, config.GRID_SIZE)
        self.y = np.clip(self.y, 0, config.GRID_SIZE)
        self.z = np.clip(self.z, 1, config.MAX_HEIGHT)