"""Configurable adaptation of the team's Drive Notebook 08 TinySASRec.

Not an independently validated reproduction of the original SASRec paper.
"""
import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=50, hidden_size=64, num_heads=4,
                 num_layers=2, dropout=0.2):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError('num_heads must divide hidden_size')
        self.num_heads = num_heads
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, 4 * hidden_size,
                    dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)

    def hidden(self, tokens):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        padding = tokens.eq(0)
        h = self.item_embedding(tokens) + self.position_embedding(positions)
        # Left-padding + causal attention otherwise gives padded queries no allowed
        # keys, causing NaNs in some attention backends. Let only padded queries
        # attend themselves; real queries still cannot attend padding or future tokens.
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        mask = mask.unsqueeze(0).expand(batch, -1, -1) | padding.unsqueeze(1)
        mask = mask & ~(padding.unsqueeze(2) & torch.eye(length, dtype=torch.bool, device=tokens.device))
        mask = mask.repeat_interleave(self.num_heads, dim=0)
        h = self.encoder(h, mask=mask)
        return self.norm(h).masked_fill(padding.unsqueeze(2), 0.0)

    def forward(self, tokens):
        return self.hidden(tokens) @ self.item_embedding.weight.T

    def next_item_scores(self, tokens):
        return self.hidden(tokens)[:, -1] @ self.item_embedding.weight.T

    def training_logits(self, tokens, targets):
        # Avoid materializing batch x length x catalog logits for padding positions.
        keep = targets.ne(0)
        return self.hidden(tokens)[keep] @ self.item_embedding.weight.T, targets[keep]
