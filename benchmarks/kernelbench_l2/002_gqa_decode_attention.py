import torch
from torch import nn


class Model(nn.Module):
    def forward(self, query, key, value):
        ratio = query.shape[1] // key.shape[2]
        expanded_key = key.repeat_interleave(ratio, dim=2).transpose(1, 2)
        expanded_value = value.repeat_interleave(ratio, dim=2).transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(
            query.unsqueeze(2), expanded_key, expanded_value
        ).squeeze(2)


def get_inputs():
    batch, context = 1, 4096
    query = torch.randn(batch, 16, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(
        batch, context, 8, 128, device="cuda", dtype=torch.bfloat16
    )
    return [query, key, torch.randn_like(key)]


def get_init_inputs():
    return []
