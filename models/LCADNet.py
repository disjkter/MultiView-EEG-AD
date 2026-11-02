"""Light CNN baseline adapted from the public LCADNet notebook."""

from __future__ import annotations

import torch
import torch.nn as nn


class LCADNet(nn.Module):
    """Minimal LCADNet-style classifier.

    The public notebook defines a compact 2D CNN:
    - Conv2D(20, kernel=3x3) + MaxPool2D(2x21)
    - Conv2D(10, kernel=3x3)
    - Flatten -> Dense(50) -> Dense(80) -> Dense(num_classes)

    Input shape:
        x: [batch, channels, time]
    Output shape:
        logits: [batch, num_classes]
        features: [batch, hidden_dim]
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 21)),
            nn.Conv2d(20, 10, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, 128, dtype=torch.float32)
            flat_dim = int(self.features(dummy).reshape(1, -1).shape[1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 80),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(80, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""

        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.features(x)
        features = self.head(x)
        logits = self.classifier(features)
        return logits, features
