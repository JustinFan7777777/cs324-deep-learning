"""Generate random digit palindromes using the supplied dataset protocol."""

import math
import sys

import numpy as np
from torch.utils.data import Dataset


class PalindromeDataset(Dataset):
    """Generate a fresh palindrome each time an item is requested."""

    def __init__(self, seq_length):
        """
        Args:
            seq_length: Total palindrome length T, including the target.
        """
        if seq_length < 2:
            raise ValueError(
                "A palindrome must contain at least two digits."
            )

        self.seq_length = seq_length

    def __len__(self):
        # Preserve the supplied effectively unbounded sampling interface.
        #
        # This value is not the number of distinct possible palindromes.
        # Training must stop using an explicit optimizer-update budget.
        return sys.maxsize

    def __getitem__(self, idx):
        # idx is intentionally unused: each call generates a fresh sample.
        full_palindrome = self.generate_palindrome()

        # Inputs contain the first T - 1 digits.
        # The target is the final digit, represented as an integer class.
        return full_palindrome[:-1], int(full_palindrome[-1])

    def generate_palindrome(self):
        """
        Generate an even- or odd-length digit palindrome.

        Examples:
            T=4: left [3, 8]    -> [3, 8, 8, 3]
            T=5: left [3, 8, 2] -> [3, 8, 2, 8, 3]
        """
        # An odd-length palindrome includes its center in the left half.
        left = np.asarray(
            [
                np.random.randint(0, 10)
                for _ in range(math.ceil(self.seq_length / 2))
            ],
            dtype=np.float32,
        )

        if self.seq_length % 2 == 0:
            # Even length: mirror the entire left half.
            right = np.flip(left, axis=0)
        else:
            # Odd length: keep the center digit only once.
            right = np.flip(left[:-1], axis=0)

        return np.concatenate((left, right))