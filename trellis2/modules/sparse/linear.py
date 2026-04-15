import torch
import torch.nn as nn
import torch.nn.functional as F
from . import VarLenTensor

__all__ = [
    'SparseLinear',
    'ROCM_SAFE_CHUNK',
    'rocm_safe_linear',
]

# ROCm GFX1201 (RX 9070 XT) bug workaround:
# hipBLASLt and rocBLAS GEMM kernels corrupt memory (→ NaN) when N > ~800k
# for shapes like [N, K] @ [K, M] with small K/M.  Chunking keeps each
# dispatch below the confirmed-safe threshold of 524288 rows.
ROCM_SAFE_CHUNK = 524_288


def rocm_safe_linear(feats: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """F.linear with ROCm large-N chunking workaround."""
    N = feats.shape[0]
    if N <= ROCM_SAFE_CHUNK:
        return F.linear(feats, weight, bias)
    out = torch.empty(N, weight.shape[0], device=feats.device, dtype=feats.dtype)
    for s in range(0, N, ROCM_SAFE_CHUNK):
        e = min(s + ROCM_SAFE_CHUNK, N)
        out[s:e] = F.linear(feats[s:e], weight, bias)
    return out


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    #def forward(self, input: VarLenTensor) -> VarLenTensor:
    #    return input.replace(super().forward(input.feats))

    def forward(self, input):
        feats = input.feats if hasattr(input, 'feats') else input
        out = rocm_safe_linear(feats, self.weight, self.bias)
        if hasattr(input, 'replace'):
            return input.replace(out)
        return out