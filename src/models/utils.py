"""Module for model utils."""

import torch


def pos_encoding(n, d_model):
    """Positional encoding."""
    assert d_model % 2 == 0, "d_model must be divisible by 2"
    wk = torch.tensor([1 / 10_000 ** (2 * i / d_model) for i in range(d_model // 2)])
    wk = wk.reshape((1, d_model // 2))
    t = torch.arange(n).reshape((n, 1))
    encoding = torch.zeros(n, d_model)
    encoding[:, ::2] = torch.sin(t * wk)
    encoding[:, 1::2] = torch.cos(t * wk)
    return encoding
