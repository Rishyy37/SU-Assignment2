import numpy as np

def load_denoiser():
    return None, None

def denoise_audio(audio, model, state):
    # Simple normalization (acts as basic denoising)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    return audio