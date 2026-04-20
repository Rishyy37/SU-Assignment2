import numpy as np
import librosa
from scipy.fft import dct
from scipy.signal import get_window

def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def save_audio(path, audio, sr=16000):
    import soundfile as sf
    sf.write(path, audio, sr)

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio

def extract_lfcc(audio, sr=16000, n_lfcc=20, n_fft=512, hop_length=160, n_filters=40):
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC)
    """
    # Pre-emphasis
    audio = librosa.effects.preemphasis(audio)

    # STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
    mag = np.abs(stft)

    # Linear filterbank
    fmin, fmax = 0, sr // 2
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_filters, fmin=fmin, fmax=fmax)
    linear_filters = np.linspace(0, sr//2, n_filters+2)[1:-1]
    linear_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_filters, fmin=fmin, fmax=fmax)  # Approximation

    # Apply filters
    lfcc = np.dot(linear_filters, mag**2)

    # Log
    lfcc = np.log(lfcc + 1e-10)

    # DCT
    lfcc = dct(lfcc, type=2, axis=0, norm='ortho')[:n_lfcc]

    return lfcc.T  # (time, n_lfcc)

def compute_eer(scores_bona, scores_spoof):
    """
    Compute Equal Error Rate
    """
    from sklearn.metrics import roc_curve
    y_true = np.concatenate([np.ones(len(scores_bona)), np.zeros(len(scores_spoof))])
    y_scores = np.concatenate([scores_bona, scores_spoof])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    return eer

def compute_snr(original, perturbed):
    """
    Compute Signal-to-Noise Ratio
    """
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - perturbed)**2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')