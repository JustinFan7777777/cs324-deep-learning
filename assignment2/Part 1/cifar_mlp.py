"""CIFAR10 MLP with optional dropout for a controlled regularization study."""

import torch
from torch import nn


class CIFARMLP(nn.Module):
    """Flatten images, then apply 3072 -> 512 -> 256 -> 10 linear layers.

    Dropout independently zeros hidden activations during training and rescales
    retained activations by 1/(1-p). In eval mode it is the identity. The final
    layer returns logits: CrossEntropyLoss already applies log-softmax.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError('dropout must lie in [0, 1).')
        # Keep PyTorch's default Linear initialization in all four variants.
        self.layers = nn.Sequential(
            nn.Flatten(),                         # (B,3,32,32) -> (B,3072)
            nn.Linear(3 * 32 * 32, 512), nn.ReLU(),
            nn.Dropout(dropout),                  # (B,512) -> (B,512)
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout),                  # (B,256) -> (B,256)
            nn.Linear(256, 10),                   # (B,256) -> (B,10)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError('Expected images with shape (B,3,32,32).')
        return self.layers(images)
