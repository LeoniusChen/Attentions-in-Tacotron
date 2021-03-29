""" This file is copied from
    https://github.com/thuhcsi/tacotron/blob/master/model/attention.py
"""

import torch
from torch import nn


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length

    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).zero_().bool()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask


class BahdanauAttention(nn.Module):
    """
    BahdanauAttention

    This attention is described in:
        D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate,"
        in International Conference on Learning Representation (ICLR), 2015.
        https://arxiv.org/abs/1409.0473
    """
    def __init__(self, query_dim, attn_dim, score_mask_value=-float("inf")):
        super(BahdanauAttention, self).__init__()

        # Query layer to project query to hidden representation
        # (query_dim -> attn_dim)
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)

        # For computing alignment energies
        self.tanh = nn.Tanh()
        self.v = nn.Linear(attn_dim, 1, bias=False)

        # For computing weights
        self.score_mask_value = score_mask_value

    def forward(self, query, processed_memory, mask=None):
        """
        Get normalized attention weight

        Args:
            query: (batch, 1, dim) or (batch, dim)
            processed_memory: (batch, max_time, dim)
            mask: (batch, max_time)

        Returns:
            alignment: [batch, max_time]
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)

        # Alignment energies
        alignment = self.get_energies(query, processed_memory)

        if mask is not None:
            mask = mask.view(query.size(0), -1)
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Alignment probabilities (attention weights)
        alignment = self.get_probabilities(alignment)

        # (batch, max_time)
        return alignment

    def init_attention(self, processed_memory):
        # Nothing to do in the base module
        return

    def get_energies(self, query, processed_memory):
        """
        Compute the alignment energies
        """
        # Query (batch, 1, dim)
        processed_query = self.query_layer(query)

        # Alignment energies (batch, max_time, 1)
        alignment = self.v(self.tanh(processed_query + processed_memory))

        # (batch, max_time)
        return alignment.squeeze(-1)

    def get_probabilities(self, energies):
        """
        Compute the alignment probabilites (attention weights) from energies
        """
        return nn.Softmax(dim=1)(energies)


class LocationSensitiveAttention(BahdanauAttention):
    """
    LocationSensitiveAttention (LSA)

    This attention is described in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio, "Attention-based Models for Speech Recognition,"
        in Advances in Neural Information Processing Systems, 2015, pp. 577-585.
        https://arxiv.org/abs/1506.07503
    """
    def __init__(self, query_dim, attn_dim, filters=32, kernel_size=31, score_mask_value=-float("inf")):
        super(LocationSensitiveAttention, self).__init__(query_dim, attn_dim, score_mask_value)

        # Location layer: Conv1d followd by Linear
        self.conv = nn.Conv1d(1, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=True)
        self.L = nn.Linear(filters, attn_dim, bias=False)

        # Cumulative attentions
        self.cumulative = None

    def init_attention(self, processed_memory):
        # Initialize cumulative attention
        b, t, c = processed_memory.size()
        self.cumulative = processed_memory.data.new(b, t).zero_()

    def get_energies(self, query, processed_memory):
        # Query (batch, 1, dim)
        processed_query = self.query_layer(query)

        # Location feature
        location = self.cumulative.unsqueeze(1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))

        # Alignment energies (batch, max_time, 1)
        alignment = self.v(self.tanh(processed_query + processed_memory + processed_loc))

        # (batch, max_time)
        return alignment.squeeze(-1)

    def get_probabilities(self, energies):
        # Current attention
        alignment = nn.Softmax(dim=1)(energies)

        # Cumulative attention
        self.cumulative = self.cumulative + alignment

        # (batch, max_time)
        return alignment


class AttentionWrapper(nn.Module):
    def __init__(self, rnn_cell, attention_mechanism):
        super(AttentionWrapper, self).__init__()
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism

    def forward(self, query, attention, cell_state, memory,
                processed_memory=None, mask=None, memory_lengths=None):
        if processed_memory is None:
            processed_memory = memory
        if memory_lengths is not None and mask is None:
            mask = get_mask_from_lengths(memory, memory_lengths)

        # Concat input query and previous attention context
        cell_input = torch.cat((query, attention), -1)

        # Feed it to RNN
        cell_output = self.rnn_cell(cell_input, cell_state)

        # GRUCell or LSTMCell
        if type(self.rnn_cell) is nn.LSTMCell:
            query = cell_output[0]
        else:
            query = cell_output

        # Normalized attention weight
        # (batch, max_time)
        alignment = self.attention_mechanism(query, processed_memory, mask)

        # Attention context vector
        # (batch, 1, dim)
        attention = torch.bmm(alignment.unsqueeze(1), memory)

        # (batch, dim)
        attention = attention.squeeze(1)

        return cell_output, attention, alignment

