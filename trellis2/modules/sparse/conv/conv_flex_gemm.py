import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
from .. import config as sparse_config
from ..linear import ROCM_SAFE_CHUNK
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction, SubMConv3dNeighborCache
from flex_gemm.ops import utils as flex_utils
import flex_gemm.kernels as flex_kernels


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def _sparse_conv3d_explicit_gemm_chunked(feats, neighbor_map, weight, bias, N, V, Co, Ci):
    """
    Chunked explicit-GEMM sparse conv: im2col + torch.mm in ROCM_SAFE_CHUNK-sized pieces.
    Avoids the flex_gemm Triton kernel for large N on ROCm GFX1201.
    """
    # weight: [Co, V, Ci] (reshaped from [Co, Kd, Kh, Kw, Ci])
    # neighbor_map: [N, V] uint32 - 0xffffffff means no neighbor
    weight_2d = weight.view(Co, V * Ci).t().contiguous()  # [V*Ci, Co]
    output = torch.zeros(N, Co, device=feats.device, dtype=feats.dtype)
    for s in range(0, N, ROCM_SAFE_CHUNK):
        e = min(s + ROCM_SAFE_CHUNK, N)
        chunk_size = e - s
        nm = neighbor_map[s:e].long()  # [chunk, V]
        # im2col: [chunk, V*Ci]
        im2col = torch.zeros(chunk_size * V, Ci, device=feats.device, dtype=feats.dtype)
        flat_nm = nm.view(-1)  # [chunk*V]
        valid = flat_nm != 0xffffffff
        # clamp invalid indices to 0 to avoid index-out-of-bounds, then mask
        safe_nm = flat_nm.clone()
        safe_nm[~valid] = 0
        im2col[valid] = feats[safe_nm[valid]]
        im2col = im2col.view(chunk_size, V * Ci)
        # GEMM: [chunk, V*Ci] @ [V*Ci, Co] -> [chunk, Co]
        output[s:e] = torch.mm(im2col, weight_2d)
    if bias is not None:
        output = output + bias
    return output


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    flex_gemm.ops.spconv.set_algorithm(config.FLEX_GEMM_ALGO)
    flex_gemm.ops.spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)

    Co, Kd, Kh, Kw, Ci = self.weight.shape
    N = x.feats.shape[0]
    V = Kd * Kh * Kw
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    # ROCm safe spconv: build neighbor map normally, then use chunked torch.mm instead of Triton
    if sparse_config.ROCM_SAFE_SPCONV and N > ROCM_SAFE_CHUNK:
        from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
        from flex_gemm.ops import utils as flex_utils
        import flex_gemm.kernels as flex_kernels
        from flex_gemm.ops.spconv import Algorithm
        if neighbor_cache is None:
            # Build neighbor map using the HIP hash kernel (small/fast operation)
            hashmap_keys, hashmap_vals = flex_utils.init_hashmap(
                torch.Size([*x.shape, *x.spatial_shape]),
                int(config.FLEX_GEMM_HASHMAP_RATIO * N),
                x.feats.device,
            )
            neighbor_map = flex_kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
                hashmap_keys, hashmap_vals,
                x.coords,
                x.spatial_shape[0], x.spatial_shape[1], x.spatial_shape[2],
                Kw, Kh, Kd,
                self.dilation[0], self.dilation[1], self.dilation[2],
            )
            # Store minimal cache so we skip rebuild next call
            from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dNeighborCache
            neighbor_cache_ = SubMConv3dNeighborCache(neighbor_map=neighbor_map)
            x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
        else:
            neighbor_map = neighbor_cache['neighbor_map']
        weight_flat = self.weight.reshape(Co, V, Ci)
        out = _sparse_conv3d_explicit_gemm_chunked(
            x.feats, neighbor_map, weight_flat, self.bias, N, V, Co, Ci
        )
        print(f"[ROCM_SAFE_SPCONV] N={N} used chunked explicit GEMM (V={V})")
        return x.replace(out)

    # Normal path: flex_gemm Triton kernel
    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        self.weight,
        self.bias,
        neighbor_cache,
        self.dilation
    )

    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)

    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')
