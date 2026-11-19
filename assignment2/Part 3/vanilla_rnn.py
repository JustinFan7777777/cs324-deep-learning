from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """Predict the final digit using an explicit tanh recurrence."""

    def __init__(
        self,
        seq_length, # number of observed time steps, equal to T - 1
        input_dim, # features per time step.
        hidden_dim, # number of hidden-state features.
        output_dim, # number of output classes. e.g. Digits prediction has 10 classes (0 -9).
        batch_size, # number of sequences in a batch.
    ):
        
        super().__init__()
        # call the initializer of the parent class (nn.Module)

        if min(
            seq_length,
            input_dim,
            hidden_dim,
            output_dim,
            batch_size,
        ) <= 0:
            raise ValueError("All parameters must be positive integers.")

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # x_t @ W_hx.T + b_h
        # weight: (hidden_dim, input_dim)
        # bias = (hidden_dim,)
        # input: (B, input_dim)
        # output: (B, hidden_dim)
        self.input_to_hidden = nn.Linear(
            input_dim,
            hidden_dim,
            bias=True,
        )

        # h_(t-1) @ W_hh.T
        # there is already one hidden bias in input_to_hidden,
        # so we do not need a bias here.
        self.hidden_to_hidden = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        # h_final @ W_ph.T + b_o
        # input: (B, hidden_dim)
        # output: (B, output_dim)
        self.hidden_to_output = nn.Linear(
            hidden_dim,
            output_dim,
            bias=True,
        )

        bound = 1.0 / math.sqrt(hidden_dim)
        # bound is used to initialize the weights of the linear layers uniformly in the range [-bound, bound].

        for layer in (
            self.input_to_hidden,
            self.hidden_to_hidden,
            self.hidden_to_output,
        ):
            nn.init.uniform_(layer.weight, -bound, bound)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Return final-step logits, resetting the hidden state for each batch."""

        if x.ndim == 2 and self.input_dim == 1:
            x = x.unsqueeze(-1)
            # unsqueeze the last dimension to make it compatible with the expected input shape (B, T, input_dim).

        if (
            x.ndim != 3
            or x.shape[1:] != (self.seq_length, self.input_dim)
        ):
            raise ValueError(
                "Expected (B, seq_length, input_dim), "
                "or (B, seq_length) for scalar inputs."
            )

        hidden = x.new_zeros(
            x.size(0),
            self.hidden_dim,
        )

        for time_index in range(self.seq_length):
            # note that the same parameter matrices are reused at every step.
            # Autograd connects the hidden states across time.
            hidden = torch.tanh(
                self.input_to_hidden(x[:, time_index, :]) + self.hidden_to_hidden(hidden)
            )

        return self.hidden_to_output(hidden)
        # Return logits directly for CrossEntropyLoss.

