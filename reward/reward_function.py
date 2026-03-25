import numpy as np
import config


def compute_reward(users, assignments, sinrs, throughputs, latencies):

    num_uavs = config.NUM_UAVS
    uav_rewards = np.zeros(num_uavs)

    # ===== NORMALIZATION CONSTANTS (IMPORTANT) =====
    SINR_SCALE = 20.0        # dB scaling
    THROUGHPUT_SCALE = 50.0  # Mbps scale (adjust if needed)
    LATENCY_SCALE = 0.05     # seconds (important)

    for i, user in enumerate(users):

        uav_id = assignments[i]

        sinr = sinrs[i]
        throughput = throughputs[i]
        latency = latencies[i]

        # ===== SMOOTH & SENSITIVE TERMS =====
        sinr_term = np.tanh(sinr / SINR_SCALE)
        throughput_term = np.tanh(throughput / THROUGHPUT_SCALE)

        # latency should penalize strongly but smoothly
        latency_term = -np.tanh(latency / LATENCY_SCALE)

        # ===== PRIORITY WEIGHTS =====
        if user.type == "rescue":
            w = 3.0
        elif user.type == "victim":
            w = 2.0
        else:
            w = 1.0

        # ===== BALANCED QoS =====
        qos = (
            0.5 * sinr_term +
            0.3 * throughput_term +
            0.2 * latency_term
        )

        uav_rewards[uav_id] += w * qos

    # ===== NORMALIZE PER UAV (VERY IMPORTANT) =====
    users_per_uav = len(users) / num_uavs
    uav_rewards = uav_rewards / (users_per_uav + 1e-6)

    # ===== FINAL STABILIZATION =====
    # keep rewards in reasonable range (-5 to +5 approx)
    uav_rewards = np.clip(uav_rewards, -5, 5)

    return uav_rewards