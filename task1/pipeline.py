import argparse
import librosa
import soundfile as sf
import whisper
from scipy.signal import medfilt

from utils import split_frames, majority_vote
from denoise import load_denoiser, denoise_audio
from lid import LIDModel


def smooth_predictions(preds):
    # Convert to numeric
    numeric = [0 if p == "Hindi" else 1 for p in preds]
    smooth = medfilt(numeric, kernel_size=5)
    return ["Hindi" if x == 0 else "English" for x in smooth]


def align_lid_with_segments(segments, lid_preds, hop_ms=10):
    aligned = []

    for seg in segments:
        start = seg["start"]
        end = seg["end"]

        start_idx = int(start * 1000 / hop_ms)
        end_idx = int(end * 1000 / hop_ms)

        segment_lid = lid_preds[start_idx:end_idx]

        if len(segment_lid) == 0:
            lang = "Unknown"
        else:
            lang = majority_vote(segment_lid)

        aligned.append({
            "start": start,
            "end": end,
            "lang": lang,
            "text": seg["text"]
        })

    return aligned


def main(audio_path):
    print("🔹 Loading audio...")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    print("🔹 Denoising...")
    model_df, df_state = load_denoiser()
    clean_audio = denoise_audio(audio, model_df, df_state)

    sf.write("data/output/clean.wav", clean_audio, 16000)

    print("🔹 Frame splitting...")
    frames = split_frames(clean_audio, sr)

    print("🔹 Running LID...")
    lid_model = LIDModel()
    lid_preds = lid_model.predict(frames)

    lid_preds = smooth_predictions(lid_preds)

    print("🔹 Running Whisper...")
    whisper_model = whisper.load_model("base")

    result = whisper_model.transcribe(
        "data/output/clean.wav",
        initial_prompt="Lecture includes terms like stochastic, cepstrum, spectrogram."
    )

    segments = result["segments"]

    print("🔹 Aligning LID with transcript...")
    aligned = align_lid_with_segments(segments, lid_preds)

    print("\n===== FINAL OUTPUT =====\n")
    for seg in aligned:
        print(f"[{seg['lang']}] ({seg['start']:.2f}-{seg['end']:.2f}) {seg['text']}")

    return aligned


if __name__ == "__main__":
    audio_path = "data/input/namo_input_clip.wav"

    #temp audio input file of 1 minute for quick testing
    audio_path_temp = "data/input/name_input_clip_temp.wav"
    
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    # parser.add_argument(audio_path, type=str, required=True, help="Path to input audio file")

    # args = parser.parse_args()

    main(audio_path)