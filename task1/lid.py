import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

class LIDModel:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Simple classifier (random initialized — replace if trained)
        self.classifier = torch.nn.Linear(768, 2)

    def predict_frame(self, frame):
        inputs = self.processor(frame, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embedding)
        pred = torch.argmax(logits, dim=1).item()

        return "Hindi" if pred == 0 else "English"

    def predict(self, frames):
        predictions = []
        for f in frames:
            pred = self.predict_frame(f)
            predictions.append(pred)

        return predictions