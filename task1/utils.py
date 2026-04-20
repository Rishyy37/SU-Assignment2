import numpy as np

def split_frames(audio, sr, frame_ms=25, hop_ms=10):
    frame_size = int(sr * frame_ms / 1000)
    hop_size = int(sr * hop_ms / 1000)

    frames = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frames.append(audio[i:i+frame_size])
    return frames


def majority_vote(labels):
    return max(set(labels), key=labels.count)