import torch

NUM_UAVS = 4
MAX_USERS = 40

GRID_SIZE = 10
MAX_STEPS = 200

UAV_ALT_MIN = 1
UAV_ALT_MAX = 10

NOISE_POWER = 1e-9
BANDWIDTH = 36e6

STATE_DIM = NUM_UAVS * 4
ACTION_DIM = 3

GAMMA = 0.99
LR = 0.0003

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_RESCUE = 4.5
W_VICTIM = 2.5
W_CIVILIAN = 1.5