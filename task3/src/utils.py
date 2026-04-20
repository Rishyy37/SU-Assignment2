import numpy as np

def apply_prosody_warping(audio, aligned_f0):
    # naive amplitude modulation (proxy for energy)
    scaled_audio = audio * (1 + 0.1 * np.tanh(aligned_f0))

    return scaled_audio