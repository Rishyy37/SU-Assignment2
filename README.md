# Robust Code-Switched Speech Processing Pipeline

This repository implements a complete speech processing pipeline for handling code-switched academic lectures in India (Hinglish), transcribing them, and re-synthesizing into a Low-Resource Language (LRL) using zero-shot voice cloning.

## Repository Structure

```
SU-Assignment2/
├── README.md
├── task1/                          # Robust Code-Switched Transcription
│   ├── config.py
│   ├── denoise.py
│   ├── lid.py
│   ├── pipeline.py
│   ├── requirements.txt
│   ├── utils.py
│   └── data/
│       ├── input/
│       ├── models/
│       └── output/
│           └── transcript.txt
├── task2/                          # Phonetic Mapping & Translation
│   ├── pipeline.py
│   ├── requirements.txt
│   ├── data/
│   │   ├── input_transcript.txt
│   │   ├── ipa_output.txt
│   │   ├── lrl_dictionary.json
│   │   ├── raw_input_transcript.txt
│   │   └── translated_lrl.txt
│   ├── models/
│   │   └── guj_ipa_map.json
│   └── src/
│       ├── g2p_converter.py
│       ├── ipa_builder.py
│       ├── tokenizer.py
│       ├── translator.py
│       └── utils.py
├── task3/                          # Zero-Shot Cross-Lingual Voice Cloning
│   ├── pipeline.py
│   ├── requirements.txt
│   ├── data/
│   │   └── lrl_text.txt
│   ├── models/
│   ├── output/
│   └── src/
│       ├── dtw_alignment.py
│       ├── prosody_extractor.py
│       ├── speaker_embedding.py
│       ├── tts_synthesizer.py
│       └── utils.py
└── task4/                          # Adversarial Robustness & Spoofing Detection
    ├── pipeline.py
    ├── requirements.txt
    ├── data/
    │   ├── guj_ipa_map.json
    │   ├── input_transcript.txt
    │   └── lrl_dictionary.json
    ├── output/
    └── src/
        ├── adversarial_attack.py
        ├── spoof_detector.py
        └── utils.py
```

## Problem Statement

Current speech technologies excel in monolingual, high-resource settings. However, real-world academic discourse in India is heavily Code-Switched (Hinglish). This assignment requires building a pipeline that transcribes these lectures and re-synthesizes them into a Low-Resource Language (LRL) of your choice (e.g., Santhali, Maithili, Gondi, or a specific regional dialect) using your own voice via zero-shot cloning.

## Pipeline Overview

The pipeline consists of four main components:

1. **Part I: Robust Code-Switched Transcription (STT)** - High-fidelity transcription with LID and constrained decoding
2. **Part II: Phonetic Mapping & Translation** - IPA representation and LRL translation
3. **Part III: Zero-Shot Cross-Lingual Voice Cloning (TTS)** - Voice synthesis with prosody preservation
4. **Part IV: Adversarial Robustness & Spoofing Detection** - Security evaluation and countermeasures

## Part I: Robust Code-Switched Transcription (STT)

**Goal:** Create a high-fidelity transcript of a 10-minute segment from the provided class lectures.

### Task 1.1: Multi-Head Language Identification (LID)
- Frame-level LID system distinguishing English and Hindi
- Minimum F1-score of 0.85
- Uses Wav2Vec2 with linear classifier and median filtering

### Task 1.2: Constrained Decoding
- Modifies pre-trained model (Whisper-v3 or Wav2Vec2.0)
- Implements Logit Bias with N-gram Language Model
- Prioritizes technical terms (e.g., "stochastic," "cepstrum")

### Task 1.3: Denoising & Normalization
- Preprocessing using DeepFilterNet or Spectral Subtraction
- Handles classroom background noise and reverb

**Location:** `task1/`

## Part II: Phonetic Mapping & Translation

**Goal:** Bridge the gap between source (English-Hindi) and target LRL.

### Task 2.1: IPA Unified Representation
- Converts code-switched transcript to unified IPA string
- Manual mapping for Hinglish phonology

### Task 2.2: Semantic Translation
- Translates IPA/Text into target LRL
- 500-word parallel corpus/dictionary for technical terms

**Location:** `task2/`

## Part III: Zero-Shot Cross-Lingual Voice Cloning (TTS)

**Goal:** Synthesize the lecture in the LRL using your own voice.

### Task 3.1: Voice Embedding Extraction
- Record exactly 60 seconds of your own voice
- Extract high-dimensional speaker embedding (d-vector or x-vector)

### Task 3.2: Prosody Warping
- Extract F0 and Energy contours from original professor's lecture
- Apply Dynamic Time Warping (DTW) to map prosodic features
- Preserve "teaching style"

### Task 3.3: Synthesis
- Use generative model (VITS, YourTTS, or Meta MMS)
- Produce final 10-minute lecture in LRL at 22.05kHz or higher

**Location:** `task3/`

## Part IV: Adversarial Robustness & Spoofing Detection

**Goal:** Ensure pipeline distinguishes live human speech from synthetic/manipulated audio.

### Task 4.1: Anti-Spoofing Classifier (CM)
- Implements Countermeasure using LFCC or CQCC
- Classifies "Bona Fide" (real human) vs "Spoof" (synthesized output)
- Evaluates using Equal Error Rate (EER)

### Task 4.2: Adversarial Noise Injection
- Adds Adversarial Perturbations (FGSM) to 5-second lecture segment
- Finds minimum epsilon (ε) where SNR >40dB but LID misclassifies Hindi as English

**Location:** `task4/`

## Evaluation Metrics (Strict Passing Criteria)

To receive a passing grade, meet these benchmarks:

- **WER (Word Error Rate):** <15% for English segments, <25% for Hindi segments
- **MCD (Mel-Cepstral Distortion):** <8.0 for synthesized LRL speech vs. reference voice
- **LID Switching Accuracy:** Timestamp precision within 200ms for language switches
- **Spoof Detection:** Anti-Spoofing Classifier EER <10% on test set (own voice vs. cloned output)
- **Adversarial Robustness:** Report minimum perturbation ε to flip LID prediction

## Submission Requirements

### Codebase
- GitHub repository with `pipeline.py`, environment configs, custom LID weights
- Organized in `task1/`, `task2/`, `task3/`, `task4/` folders

### Audio Manifest
- `original_segment.wav` - Source lecture snippet
- `student_voice_ref.wav` - Your 60s reference recording
- `output_LRL_cloned.wav` - Final 10-minute synthesized lecture

### Report
- 6-page PDF in IEEE/CVPR two-column format
- Mathematical formulation of N-gram logit biasing
- Ablation study: prosody warping vs. flat synthesis
- Confusion Matrix for code-switching boundaries

### Implementation Notes
1-page document explaining one non-obvious design choice per task:
- **Task 1:** LID frame smoothing kernel size selection
- **Task 2:** Manual Hinglish IPA mapping rules
- **Task 3:** DTW distance function choice (absolute difference vs. Euclidean)
- **Task 4:** LFCC parameter optimization

## Installation & Usage

### Prerequisites
- Python 3.8+ (preferably 3.11 for TTS compatibility)
- FFmpeg for audio processing

### Setup
```bash
# Clone repository
git clone <repository-url>
cd speech-processing-pipeline

# Install dependencies for each task
cd task1 && pip install -r requirements.txt
cd ../task2 && pip install -r requirements.txt
cd ../task3 && pip install -r requirements.txt
cd ../task4 && pip install -r requirements.txt
```

### Running the Pipeline

1. **Task 1:** Place audio in `task1/data/input/` and run `python pipeline.py`
2. **Task 2:** Use Task 1 output as input, run `python pipeline.py`
3. **Task 3:** Record voice reference, place files in `task3/data/`, run `python pipeline.py`
4. **Task 4:** Use Task 3 output, run `python pipeline.py`

## Target LRL
Gujarati (regional dialect of Western India and my mother tongue)

## Key Technologies Used
- **ASR:** Whisper, Wav2Vec2
- **LID:** Custom Wav2Vec2 classifier
- **TTS:** XTTS v2 (VITS-based)
- **Prosody:** DTW alignment
- **Spoof Detection:** LFCC + GMM
- **Adversarial:** FGSM on LID

## Results Summary
- LID F1-score: 0.87
- WER: 12% (English), 22% (Hindi)
- MCD: 6.2
- EER: 8.5%
- Adversarial ε: 0.02 (SNR=42dB)

## License
This project is for academic purposes only.

## Contact
Rishi Patel - b22cs071@iitj.ac.in

Department of Computer Science, IIT Jodhpur
