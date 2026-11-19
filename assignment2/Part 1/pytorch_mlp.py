from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

class MLP(nn.Module):
    
    def __init__(
        self,
        n_inputs: int, # number of inputs
        n_hidden: list[int], # list of integers, where each integer is the number of units in each linear layer
        n_classes: int, # number of classes of the classification problem
    ):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """

        super().__init__()
        # super calls the initializer of the parent class (nn.Module),
        # to ensure that the module is properly initialized.

        if n_inputs <= 0 or n_classes <= 0:
            raise ValueError("Input and output dimensions must be positive integers.")

        if any(width <= 0 for width in n_hidden):
            raise ValueError("Hidden-layer dimensions must be positive integers.")

        layers = []
        previous_width = n_inputs
        # previous_width is the number of units in the previous layer, starting with the input layer

        for hidden_width in n_hidden:

            # nn.Linear stores:
            # weight: (hidden_width, previous_width)
            # bias: (hidden_width)

            # for row-oriented batches, it computes:
            # output = input @ weight.T + bias
            
            # shape:
            # input: (batch_size, previous_width)
            # output: (batch_size, hidden_width)

            layers.append(nn.Linear(previous_width, hidden_width))

            layers.append(nn.ReLU())
            # ReLU introduces non-linearity to the network,
            # allowing it to learn complex patterns in the data,
            # without changing the shape of the data.

            previous_width = hidden_width
            # update previous_width to the current hidden layer's width for the next iteration.

        # the output layer produces one logit per class,
        # so it's shape is: (B, previous_width) -> (B, n_classes)
        layers.append(nn.Linear(previous_width, n_classes))

        self.layers = nn.Sequential(*layers)
        # nn.Sequential is a container that allows us to stack layers together in a sequential manner.
        # The * operator unpacks the list of layers into individual arguments for nn.Sequential.
        # This means that the layers will be applied in the order they were added to the list.

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """

        return self.layers(x)
        # the forward method defines how the input data flows through the network.
