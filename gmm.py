"""
    Referred from https://github.com/keithito/tacotron/issues/136
    and https://github.com/mozilla/TTS/blob/dev/TTS/tts/layers/attentions.py
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from attention_base import BahdanauAttention


class GMMAttention(BahdanauAttention):
    def __init__(self, query_dim, attn_dim, K, version, score_mask_value=1e-8):
        super(GMMAttention, self).__init__(query_dim, attn_dim, score_mask_value)
        self.gmm_version = version
        self.K = K  # num mixture
        self.eps = 1e-5
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, attn_dim, bias=True),
            nn.Tanh(),
            nn.Linear(attn_dim, 3*K))
    
    def init_attention(self, processed_memory):
        # No need to initialize alignment 
        # because GMM Attention is purely location based 
        # it has nothing to do with memory and t-1's alignment

        # Initial mu_pre with all zeros
        b, t, c = processed_memory.size()
        self.mu_prev = processed_memory.data.new(b, self.K, 1).zero_()
        j = torch.arange(0, processed_memory.size(1)).to(processed_memory.device)
        self.j = j.view(1, 1, processed_memory.size(1))  # [1, 1, T]

    def get_energies(self, query, processed_memory):
        '''
         Args:
            query: (batch, dim)
            processed_memory: (batch, max_time, dim)
        Returns:
            alignment: [batch, max_time]
        '''
        # Intermediate parameters (in Table 1)
        interm_params = self.mlp(query).view(query.size(0), -1, self.K)  # [B, 3, K]
        omega_hat, delta_hat, sigma_hat = interm_params.chunk(3, dim=1)  # Tuple

        # Each [B, K]
        omega_hat = omega_hat.squeeze(1)
        delta_hat = delta_hat.squeeze(1)
        sigma_hat = sigma_hat.squeeze(1)

        # Convert intermediate parameters to final mixture parameters
        # Choose version V0/V1/V2
        # Formula from https://arxiv.org/abs/1910.10288
        if self.gmm_version == '0':
            sigma = (torch.sqrt(torch.exp(-sigma_hat) / 2) + self.eps).unsqueeze(-1)  # [B, K, 1]
            delta = torch.exp(delta_hat).unsqueeze(-1)  # [B, K, 1]
            omega = torch.exp(omega_hat).unsqueeze(-1)  # [B, K, 1]
            Z = 1.0
        elif self.gmm_version == '1':
            sigma = (torch.sqrt(torch.exp(sigma_hat)) + self.eps).unsqueeze(-1)
            delta = torch.exp(delta_hat).unsqueeze(-1)
            omega = F.softmax(omega_hat, dim=-1).unsqueeze(-1)
            Z = torch.sqrt(2 * np.pi * sigma**2)
        elif self.gmm_version == '2':
            sigma = (F.softplus(sigma_hat) + self.eps).unsqueeze(-1)
            delta = F.softplus(delta_hat).unsqueeze(-1)
            omega = F.softmax(omega_hat, dim=-1).unsqueeze(-1)
            Z = torch.sqrt(2 * np.pi * sigma**2)

        mu = self.mu_prev + delta  # [B, K, 1]

        # Get alignment(phi in mathtype)
        alignment = omega / Z * torch.exp(-(self.j - mu)**2 / (sigma**2) / 2)  # [B, K ,T]
        alignment = torch.sum(alignment, 1)  # [B, T]

        # Update mu_prev
        self.mu_prev = mu

        return alignment
    
    def get_probabilities(self, energies):
        return energies
