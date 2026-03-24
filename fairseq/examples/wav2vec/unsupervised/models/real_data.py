# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class RealData:
    """
    Provides "real" samples for the GAN discriminator in Wav2Vec-U.

    In the Wav2Vec-U framework, "real" samples are one-hot encoded phoneme
    sequences derived from unpaired text data. The discriminator learns to
    distinguish these real phoneme distributions from the "fake" distributions
    produced by the Generator.

    This class encapsulates the logic for:
    1. Converting token indices into one-hot representations
    2. Computing padding masks for the real samples
    """

    def __init__(self, output_dim, pad_index):
        """
        Args:
            output_dim: The number of phoneme classes (vocabulary size).
            pad_index: The index of the padding token in the dictionary.
        """
        self.output_dim = output_dim
        self.pad_index = pad_index

    def get_samples(self, tokens, device):
        """
        Convert token indices to one-hot encoded vectors ("real" samples).

        Args:
            tokens: Tensor of token indices (B, T) from unpaired text data.
            device: The device to create the tensor on.

        Returns:
            One-hot encoded tensor of shape (B, T, output_dim) representing
            the real phoneme distributions for the discriminator.
            Returns None if tokens is None.
        """
        if tokens is None:
            return None

        token_x = torch.zeros(
            tokens.numel(), self.output_dim, device=device
        )
        token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
        token_x = token_x.view(tokens.shape + (self.output_dim,))
        return token_x

    def get_padding_mask(self, tokens):
        """
        Compute the padding mask for real token sequences.

        Args:
            tokens: Tensor of token indices (B, T).

        Returns:
            Boolean tensor (B, T) where True indicates padded positions.
        """
        return tokens == self.pad_index
