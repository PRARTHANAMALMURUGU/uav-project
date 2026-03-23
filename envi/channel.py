import numpy as np
import config

def path_loss(d):

    f = 2e9
    c = 3e8
    wavelength = c/f

    return 20*np.log10(4*np.pi*d/wavelength)


def channel_gain(d):

    pl = path_loss(d)

    fading = np.random.rayleigh()

    gain = fading/(10**(pl/10))

    return gain


def compute_sinr(signal):

    sinr = signal/config.NOISE_POWER

    sinr_db = 10*np.log10(sinr+1e-9)

    return sinr_db