import os
import numpy as np
import soundfile as sf

try:
    from TTS.api import TTS
    _HAS_COQUI_TTS = True
except ImportError:
    _HAS_COQUI_TTS = False

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except ImportError:
    _HAS_PYTTSX3 = False

def synthesize_speech(text, speaker_wav, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if _HAS_COQUI_TTS:
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        wav = tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language="gu"
        )
        sf.write(output_path, wav, 22050)
        return wav

    if not _HAS_PYTTSX3:
        # Fallback: return the speaker_wav as dummy synthesized audio
        # This allows the pipeline to run prosody warping on the original voice
        # Replace with actual LRL synthesis later
        import librosa
        wav, _ = librosa.load(speaker_wav, sr=22050)
        sf.write(output_path, wav, 22050)
        return wav

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)

    engine.save_to_file(text, output_path)
    engine.runAndWait()

    import librosa
    wav, _ = librosa.load(output_path, sr=22050)
    sf.write(output_path, wav, 22050)

    return wav