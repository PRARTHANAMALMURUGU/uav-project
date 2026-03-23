import numpy as np
import config

from envi.uav import UAV
from envi.user_model import User
from envi.channel import channel_gain, compute_sinr


class UAVEnvironment:

    def __init__(self):

        self.uavs = [UAV() for _ in range(config.NUM_UAVS)]
        self.users = [User() for _ in range(config.MAX_USERS)]


    def get_state(self):

        state=[]

        for uav in self.uavs:

            state.extend([
                uav.x,
                uav.y,
                uav.z,
                uav.battery
            ])

        return np.array(state,dtype=np.float32)


    def step(self,actions):

        for uav,a in zip(self.uavs,actions):
            uav.move(a)

        for user in self.users:
            user.move()

        sinrs=[]

        for user in self.users:

            best_signal=0

            for uav in self.uavs:

                d=np.sqrt(
                    (uav.x-user.x)**2 +
                    (uav.y-user.y)**2 +
                    uav.z**2
                )

                gain=channel_gain(d)

                best_signal=max(best_signal,gain)

            sinr=compute_sinr(best_signal)

            sinrs.append(sinr)

        return np.mean(sinrs)