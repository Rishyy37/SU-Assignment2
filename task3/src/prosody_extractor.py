import librosa
import numpy as np

def extract_prosody(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    # F0 (pitch)
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    # Energy (RMS)
    energy = librosa.feature.rms(y=y)[0]

    # Clean NaNs
    f0 = np.nan_to_num(f0)

    return f0, energy