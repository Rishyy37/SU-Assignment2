from src.speaker_embedding import extract_speaker_embedding
from src.prosody_extractor import extract_prosody
from src.dtw_alignment import align_prosody
from src.tts_synthesizer import synthesize_speech
from src.utils import apply_prosody_warping

import os
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_AUDIO = os.path.join(BASE_DIR, "data", "original_segment.wav")
VOICE_REF = os.path.join(BASE_DIR, "data", "rishicp_voice_ref.wav")
LRL_TEXT = os.path.join(BASE_DIR, "data", "lrl_text.txt")
OUTPUT = os.path.join(BASE_DIR, "output", "output_LRL_cloned.wav")


def main():
    print("Step 1: Extract speaker embedding...")
    embedding = extract_speaker_embedding(VOICE_REF)

    print("Step 2: Extract prosody from original...")
    f0, energy = extract_prosody(ORIGINAL_AUDIO)

    print("Step 3: Generate base TTS...")
    with open(LRL_TEXT, "r", encoding="utf-8") as f:
        text = f.read()

    tts_audio = synthesize_speech(text, VOICE_REF, OUTPUT)

    print("Step 4: Align prosody using DTW...")
    aligned_f0 = align_prosody(f0, len(tts_audio))

    print("Step 5: Apply prosody warping...")
    final_audio = apply_prosody_warping(tts_audio, aligned_f0)

    sf.write(OUTPUT, final_audio, 22050)

    print("\n✅ Task 3 Completed!")
    print(f"Output saved at: {OUTPUT}")


if __name__ == "__main__":
    main()