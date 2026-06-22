import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, group_size=128):
        super().__init__()
        self.group_size = group_size

    def forward(self, x, packed_weight, scales):
        low = (packed_weight & 0xF).to(torch.int8) - 8
        high = ((packed_weight >> 4) & 0xF).to(torch.int8) - 8
        quant = torch.stack((low, high), dim=-1).flatten(-2).float()
        weight = quant * scales.repeat_interleave(self.group_size, dim=1).float()
        return torch.mv(weight, x.float()).to(x.dtype)


def get_inputs():
    n, k = 3072, 1024
    return [
        torch.randn(k, device="cuda", dtype=torch.float16),
        torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8),
        torch.rand(n, k // 128, device="cuda", dtype=torch.float16) * 0.02,
    ]


def get_init_inputs():
    return [128]
