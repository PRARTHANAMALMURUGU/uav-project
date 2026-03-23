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

        self.x += np.clip(dx,-1,1)
        self.y += np.clip(dy,-1,1)
        self.z += np.clip(dz,-0.5,0.5)

        self.z = np.clip(self.z,
                         config.UAV_ALT_MIN,
                         config.UAV_ALT_MAX)

        energy = 0.5*np.linalg.norm(action)

        self.battery -= energy