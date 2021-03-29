"""
    This file is coppied from
    https://github.com/thuhcsi/tacotron/blob/master/model/attention_sma.py
"""

import torch
from torch import nn

from attention_base import BahdanauAttention


class StepwiseMonotonicAttention(BahdanauAttention):
    """
    StepwiseMonotonicAttention (SMA)

    This attention is described in:
        M. He, Y. Deng, and L. He, "Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS,"
        in Annual Conference of the International Speech Communication Association (INTERSPEECH), 2019, pp. 1293-1297.
        https://arxiv.org/abs/1906.00672

    See:
        https://gist.github.com/mutiann/38a7638f75c21479582d7391490df37c
        https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention
    """
    def __init__(self, query_dim, attn_dim, sigmoid_noise=2.0, score_mask_value=-float("inf")):
        """
        Args:
            sigmoid_noise: Standard deviation of pre-sigmoid noise.
                           Setting this larger than 0 will encourage the model to produce
                           large attention scores, effectively making the choosing probabilities
                           discrete and the resulting attention distribution one-hot.
        """
        super(StepwiseMonotonicAttention, self).__init__(query_dim, attn_dim, score_mask_value)

        self.alignment = None # alignment in previous query time step
        self.sigmoid_noise = sigmoid_noise

    def init_attention(self, processed_memory):
        # Initial alignment with [1, 0, ..., 0]
        b, t, c = processed_memory.size()
        self.alignment = processed_memory.new_zeros(b, t)
        self.alignment[:, 0:1] = 1

    def stepwise_monotonic_attention(self, p_i, prev_alignment):
        """
        Compute stepwise monotonic attention
            - p_i: probability to keep attended to the last attended entry
            - Equation (8) in section 3 of the paper
        """
        pad = prev_alignment.new_zeros(prev_alignment.size(0), 1)
        alignment = prev_alignment * p_i + torch.cat((pad, prev_alignment[:, :-1] * (1.0 - p_i[:, :-1])), dim=1)
        return alignment

    def get_selection_probability(self, e, std):
        """
        Compute selecton/sampling probability `p_i` from energies `e`
            - Equation (4) and the tricks in section 2.2 of the paper
        """
        # Add Gaussian noise to encourage discreteness
        if self.training:
            noise = e.new_zeros(e.size()).normal_()
            e = e + noise * std

        # Compute selecton/sampling probability p_i
        # (batch, max_time)
        return torch.sigmoid(e)

    def get_probabilities(self, energies):
        # Selecton/sampling probability p_i
        p_i = self.get_selection_probability(energies, self.sigmoid_noise)
        
        # Stepwise monotonic attention
        alignment = self.stepwise_monotonic_attention(p_i, self.alignment)

        # (batch, max_time)
        self.alignment = alignment
        return alignment
