"""Reduced VGG specified in the assignment lecture slides."""

import torch
from torch import nn


class CNN(nn.Module):
    """Classify 32 x 32 images using eight convolutional blocks."""

    def __init__(
        self,
        n_channels: int = 3, # number of channels in the input image
        n_classes: int = 10, # number of classes of the classification problem
    ):
        super().__init__()
        # super calls the initializer of the parent class (nn.Module),
        # to ensure that the module is properly initialized.

        if n_channels <= 0 or n_classes <= 0:
            raise ValueError("Channel and class counts must be positive integers.")

        self.n_channels = n_channels

        # Each integer specifies the output channels of a convolution block.
        # Each block contains Conv2d -> BatchNorm2d -> ReLU.
        # "M" inserts a separate max-pooling layer at the specified position.
        configuration = [
            64, "M",
            128, "M",
            256, 256, "M",
            512, 512, "M",
            512, 512, "M"
        ]

        layers = []
        in_channels = n_channels

        for item in configuration:
            if item == "M":

                # for the spatial sizes used here:
                # 32 -> 16 -> 8 -> 4 -> 2 -> 1
                # pooling preserves the number of channels,
                # but reduces the spatial size of the feature maps.
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )

            else:

                # A 3x3 convolutional layer with stride 1 and padding 1:
                # preserves the spatial size of the feature maps.
                layers.extend([

                    # nn.Conv2d stores:
                    # weight: (out_channels, in_channels, kernel_size, kernel_size)
                    nn.Conv2d(
                        in_channels,
                        item,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),

                    nn.BatchNorm2d(item),
                    # batch normalization normalizes the output of the convolutional layer,
                    # which can help with training stability and convergence.

                    nn.ReLU(),
                    # ReLU changes values without changing tensor shape,
                    # introducing non-linearity to the network,
                    # allowing it to learn complex patterns in the data.
                ])

                in_channels = item
                # update in_channels to the current convolutional layer's output channels for the next iteration.

        # Sequential registers all convolution and BatchNorm parameters.
        self.features = nn.Sequential(*layers)

        # After five pools, each 32x32 image becomes a 512x1x1 tensor.
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits.

        Args:
            x: Images with shape (B, n_channels, 32, 32).

        Returns:
            Logits with shape (B, n_classes).
        """

        # This architecture's final linear layer assumes 32x32 inputs.
        if x.ndim != 4 or x.shape[1:] != (self.n_channels, 32, 32):
            raise ValueError(
                "Expected input shape (B, n_channels, 32, 32)."
            )

        x = self.features(x)
        # (B, 3, 32, 32) -> (B, 512, 1, 1)

        x = torch.flatten(x, start_dim=1)
        # (B, 512, 1, 1) -> (B, 512)
        # start_dim=1 preserves dimension 0, the batch dimension,
        # while flattening all other dimensions into a single dimension.

        return self.classifier(x)
        # (B, 512) -> (B, n_classes)
        # return raw logits for cross-entropy loss,
        # which combines softmax and negative log-likelihood loss in a single function.
              
