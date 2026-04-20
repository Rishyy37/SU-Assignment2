import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from utils import load_audio, save_audio, compute_snr

class LIDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(768, 2)  # Gujarati/English

    def forward(self, input_values):
        outputs = self.model(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

def fgsm_attack(model, audio, target_class, epsilon, sr=16000):
    """
    Perform FGSM attack on LID to misclassify audio
    """
    model.eval()
    audio_tensor = torch.tensor(audio, dtype=torch.float32, requires_grad=True)
    input_values = model.processor(audio_tensor, sampling_rate=sr, return_tensors="pt").input_values

    # Forward pass
    logits = model(input_values)
    pred = torch.argmax(logits, dim=1)

    if pred.item() == target_class:
        # Already target class, no attack needed
        return audio, 0.0

    # Compute loss
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([target_class], dtype=torch.long))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get gradient
    grad = audio_tensor.grad.data

    # FGSM step
    sign_grad = torch.sign(grad)
    perturbed_audio = audio_tensor + epsilon * sign_grad
    perturbed_audio = torch.clamp(perturbed_audio, -1, 1)  # Assuming normalized

    return perturbed_audio.detach().numpy(), loss.item()

def find_min_epsilon(model, audio, original_pred, sr=16000, max_epsilon=0.1, step=0.001, snr_threshold=40):
    """
    Find minimum epsilon that flips prediction and keeps SNR > threshold
    """
    epsilon = 0.0
    while epsilon < max_epsilon:
        epsilon += step
        perturbed, _ = fgsm_attack(model, audio, 1 - original_pred, epsilon, sr)

        # Check SNR
        snr = compute_snr(audio, perturbed)
        if snr < snr_threshold:
            continue

        # Check if prediction flipped
        perturbed_tensor = torch.tensor(perturbed, dtype=torch.float32)
        input_values = model.processor(perturbed_tensor, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values)
            new_pred = torch.argmax(logits, dim=1).item()

        if new_pred != original_pred:
            return epsilon, perturbed, snr

    return None, None, None  # No epsilon found

def run_adversarial_analysis(model, audio_paths, output_dir):
    """
    Run adversarial analysis on a set of audio files
    """
    results = []
    for path in audio_paths:
        audio = load_audio(path)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        input_values = model.processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values)
            original_pred = torch.argmax(logits, dim=1).item()

        epsilon, perturbed, snr = find_min_epsilon(model, audio, original_pred)

        if epsilon is not None:
            save_audio(f"{output_dir}/perturbed_{path.split('/')[-1]}", perturbed)
            results.append({
                'file': path,
                'original_pred': original_pred,
                'epsilon': epsilon,
                'snr': snr
            })

    return results