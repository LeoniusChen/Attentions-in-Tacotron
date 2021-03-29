"""
    Referred from https://github.com/bshall/Tacotron
"""

import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import betabinom
import numpy as np

from attention_base import BahdanauAttention


class DynamicConvolutionAttention(BahdanauAttention):
    def __init__(self, query_dim, attn_dim, static_channels=8, static_kernel_size=21,
                dynamic_channels=8, dynamic_kernel_size=21, prior_length=11, 
                alpha=0.1, beta=0.9, score_mask_value=-float("inf")):
        super(DynamicConvolutionAttention, self).__init__(query_dim, attn_dim, score_mask_value)
        self.prior_length = prior_length
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size

        P = betabinom.pmf(np.arange(prior_length), prior_length - 1, alpha, beta)

        self.register_buffer("P", torch.FloatTensor(P).flip(0))
        self.W = nn.Linear(query_dim, attn_dim)
        self.V = nn.Linear(
            attn_dim, dynamic_channels * dynamic_kernel_size, bias=False
        )
        self.F = nn.Conv1d(
            1,
            static_channels,
            static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False,
        )
        self.U = nn.Linear(static_channels, attn_dim, bias=False)
        self.T = nn.Linear(dynamic_channels, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    
    def init_attention(self, processed_memory):
        b, t, _ = processed_memory.size()
        self.alignment_pre = F.one_hot(torch.zeros(b, dtype=torch.long), t).float().cuda()

    def get_energies(self, query, processed_memory):
        query = query.squeeze(1)
        p = F.conv1d(
            F.pad(self.alignment_pre.unsqueeze(1), (self.prior_length - 1, 0)), self.P.view(1, 1, -1)
        )
        p = torch.log(p.clamp_min_(1e-6)).squeeze(1)

        G = self.V(torch.tanh(self.W(query)))
        g = F.conv1d(
            self.alignment_pre.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=query.size(0),
        )
        g = g.view(query.size(0), self.dynamic_channels, -1).transpose(1, 2)

        f = self.F(self.alignment_pre.unsqueeze(1)).transpose(1, 2)

        e = self.v(torch.tanh(self.U(f) + self.T(g))).squeeze(-1) + p

        return e
    
    def get_probabilities(self, energies):
        # Current attention
        alignment = nn.Softmax(dim=1)(energies)

        # Update previous attention
        self.alignment_pre = alignment

        return alignment