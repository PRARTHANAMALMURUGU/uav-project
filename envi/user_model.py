
import numpy as np
import random
import config

class User:

    def __init__(self):

        self.x = random.uniform(0, config.GRID_SIZE)
        self.y = random.uniform(0, config.GRID_SIZE)

        self.type = random.choice(
            ["rescue","victim","civilian"]
        )


    def move(self):

        dx = random.uniform(-0.3,0.3)
        dy = random.uniform(-0.3,0.3)

        self.x = np.clip(self.x+dx,0,config.GRID_SIZE)
        self.y = np.clip(self.y+dy,0,config.GRID_SIZE)