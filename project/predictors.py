from typing import Optional

import torch
import torch.nn as nn

from project.image_encoders import Embeddings


class Predictor(nn.Module):
    """Basic "binary" predictor: linear layer on top of
    pretrained image encoder outputting values in [0, 1].

    Attributes:
        image_encoder: Image encoder.
        ffn: dropout -> linear -> sigmoids.
    """

    def __init__(
        self,
        encoder: Embeddings,
        dropout_prob: Optional[float] = 0.5,
        n_tasks: Optional[int] = 3,
    ):
        """Init.

        Args:
            encoder: image encoder.
            dropout_prob: dropout probability.
            n_tasks: number of binary tasks.
        """

        super().__init__()

        self.image_encoder = encoder

        self.ffn = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(encoder.out_dim, n_tasks),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input images.

        Returns:
            [0, 1] predictions for each task.
        """
        x = self.image_encoder(x)
        predictions = self.ffn(x)
        return predictions
