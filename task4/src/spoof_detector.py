import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from utils import extract_lfcc, load_audio

class SpoofDetector:
    def __init__(self, n_components=8, n_lfcc=20):
        self.n_components = n_components
        self.n_lfcc = n_lfcc
        self.scaler = StandardScaler()
        self.gmm_bona = GaussianMixture(n_components=n_components, covariance_type='diag')
        self.gmm_spoof = GaussianMixture(n_components=n_components, covariance_type='diag')
        self.trained = False

    def extract_features(self, audio_path):
        audio = load_audio(audio_path)
        lfcc = extract_lfcc(audio, n_lfcc=self.n_lfcc)
        # Mean pooling over time
        features = np.mean(lfcc, axis=0)
        return features

    def train(self, bona_fide_paths, spoof_paths):
        bona_features = [self.extract_features(path) for path in bona_fide_paths]
        spoof_features = [self.extract_features(path) for path in spoof_paths]

        all_features = np.array(bona_features + spoof_features)
        self.scaler.fit(all_features)

        bona_scaled = self.scaler.transform(bona_features)
        spoof_scaled = self.scaler.transform(spoof_features)

        self.gmm_bona.fit(bona_scaled)
        self.gmm_spoof.fit(spoof_scaled)
        self.trained = True

    def predict_score(self, audio_path):
        if not self.trained:
            raise ValueError("Model not trained")
        features = self.extract_features(audio_path)
        scaled = self.scaler.transform([features])[0]

        score_bona = self.gmm_bona.score([scaled])
        score_spoof = self.gmm_spoof.score([scaled])

        # Return log-likelihood ratio: log P(bona) - log P(spoof)
        return score_bona[0] - score_spoof[0]

    def predict(self, audio_path, threshold=0):
        score = self.predict_score(audio_path)
        return score > threshold  # True if bona fide

    def evaluate_eer(self, bona_test_paths, spoof_test_paths):
        bona_scores = [self.predict_score(path) for path in bona_test_paths]
        spoof_scores = [self.predict_score(path) for path in spoof_test_paths]

        from utils import compute_eer
        eer = compute_eer(bona_scores, spoof_scores)
        return eer