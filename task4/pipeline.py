import os
import argparse
from src.spoof_detector import SpoofDetector
from src.adversarial_attack import LIDModel, run_adversarial_analysis
from utils import load_audio

def main():
    # Paths (adjust as needed)
    bona_fide_paths = ["data/bona_fide_1.wav", "data/bona_fide_2.wav"]  # Real human speech
    spoof_paths = ["../task_3/output/output_LRL_cloned.wav"]  # Synthesized speech
    test_audio_paths = ["data/test_segment.wav"]  # For adversarial analysis

    print("Step 1: Training Spoof Detector...")
    detector = SpoofDetector()
    detector.train(bona_fide_paths, spoof_paths)

    print("Step 2: Evaluating Spoof Detector...")
    eer = detector.evaluate_eer(bona_fide_paths, spoof_paths)
    print(f"EER: {eer:.4f}")

    print("Step 3: Running Adversarial Analysis...")
    model = LIDModel()
    results = run_adversarial_analysis(model, test_audio_paths, "output/")

    print("Results:")
    for res in results:
        print(f"File: {res['file']}, Epsilon: {res['epsilon']:.4f}, SNR: {res['snr']:.2f}dB")

    print("\n✅ Task 4 Completed!")

if __name__ == "__main__":
    main()