import torch
import torch.nn as nn

from project.image_encoders import Embeddings


class Predictor(nn.Module):
    def __init__(self, encoder: Embeddings, dropout_prob=0.5, n_classes=3):
        super().__init__()

        self.image_encoder = encoder

        self.ffn = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(encoder.out_dim, n_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.image_encoder(x)
        predictions = self.ffn(x)
        return predictions
