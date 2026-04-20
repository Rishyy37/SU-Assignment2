import numpy as np
from fastdtw import fastdtw

def align_prosody(source_f0, target_len):
    if len(source_f0) == 0 or target_len == 0:
        return np.zeros(target_len)

    # create dummy target timeline
    target_f0 = np.linspace(0, 1, target_len)

    # Use absolute difference for scalar distance
    distance, path = fastdtw(source_f0, target_f0, dist=lambda a, b: abs(a - b))

    aligned_f0 = np.zeros(target_len)

    for i, j in path:
        if i < len(source_f0) and j < target_len:
            aligned_f0[j] = source_f0[i]

    return aligned_f0