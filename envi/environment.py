#environment.py
from turtle import distance

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

        states = []

        for uav in self.uavs:

            state = [
                uav.x / config.GRID_SIZE,
                uav.y / config.GRID_SIZE,
                uav.z / config.MAX_HEIGHT,
                uav.battery / 100
            ]

            # user centroid (global info)
            mean_x = np.mean([u.x for u in self.users]) / config.GRID_SIZE
            mean_y = np.mean([u.y for u in self.users]) / config.GRID_SIZE

            state.extend([mean_x, mean_y])

            states.append(state)

        return np.array(states, dtype=np.float32)  # (num_uavs, state_dim)
    def step(self, actions):

        # ===== Move UAVs =====
        for uav, a in zip(self.uavs, actions):
            uav.move(a)

        # ===== Move users =====
        for user in self.users:
            user.move()

        sinrs = []
        throughputs = []
        latencies = []
        user_assignments = []

        for user in self.users:

            signals = []
            distances = []

            for uav in self.uavs:

                d = np.sqrt(
                    (uav.x - user.x)**2 +
                    (uav.y - user.y)**2 +
                    uav.z**2
                )

                gain = channel_gain(d)

                signals.append(gain)
                distances.append(d)

            signals = np.array(signals)
            distances = np.array(distances)

            # ===== FIND BEST UAV =====
            best_idx = np.argmax(signals)

            # ===== STORE ASSIGNMENT (THIS IS THE KEY LINE) =====
            user_assignments.append(best_idx)

            signal = signals[best_idx]
            best_distance = distances[best_idx]

            # ===== INTERFERENCE =====
            interference = np.sum(signals) - signal

            # ===== SINR =====
            sinr_linear = signal / (interference + config.NOISE_POWER + 1e-9)
            sinr = 10 * np.log10(sinr_linear + 1e-9)

            sinr = np.clip(sinr, -10, 40)

            # ===== THROUGHPUT =====
            throughput = config.BANDWIDTH * np.log2(1 + sinr_linear)
            throughput = throughput / 1e6

            # ===== LATENCY =====
            latency = best_distance / 3e8 + 1 / (throughput + 1e-6)

            sinrs.append(sinr)
            throughputs.append(throughput)
            latencies.append(latency)

                # ===== Return averages =====
        return (
        sinrs,         # list per user
        throughputs,
        latencies,
        user_assignments
    )
    def get_global_state(self):

        state = []

        for uav in self.uavs:
            state.extend([uav.x, uav.y, uav.z])

        for user in self.users:
            state.extend([user.x, user.y])

        return np.array(state, dtype=np.float32)