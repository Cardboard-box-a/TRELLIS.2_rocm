"""
Native PyTorch sparse convolution implementation.
No CUDA kernels, no Triton - works on any PyTorch backend (CUDA, ROCm, CPU).
MUCH SLOWER than optimized backends but numerically stable.

Use for debugging or when other backends fail.
"""

import math
import torch
import torch.nn as nn
from .. import SparseTensor


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    """
    Initialize sparse 3D convolution layer.
    """
    assert stride == 1 and padding is None, \
        'Native implementation only supports submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    
    # Weight shape: (out_channels, kernel_d, kernel_h, kernel_w, in_channels)
    # Matches FlexGEMM's layout for compatibility
    self.weight = nn.Parameter(torch.empty((out_channels, *self.kernel_size, in_channels)))
    
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)
    
    # Initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight.view(out_channels, -1, in_channels), a=math.sqrt(5))
    if self.bias is not None:
        fan_in = in_channels * math.prod(self.kernel_size)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)


def _build_coord_map(coords):
    """Build hash map from coordinates to indices."""
    coord_map = {}
    for i in range(coords.shape[0]):
        key = (int(coords[i, 0].item()), int(coords[i, 1].item()), 
               int(coords[i, 2].item()), int(coords[i, 3].item()))
        coord_map[key] = i
    return coord_map


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    """
    Forward pass for native sparse 3D convolution.
    Uses precomputed neighbor cache for efficiency.
    """
    coords = x.coords  # [N, 4] - (batch_idx, x, y, z)
    feats = x.feats    # [N, C_in]
    
    N = coords.shape[0]
    C_out = self.weight.shape[0]
    Kd, Kh, Kw = self.kernel_size
    dk, dh, dw = self.dilation
    device = feats.device
    dtype = feats.dtype
    
    # Center offsets
    kd_c, kh_c, kw_c = Kd // 2, Kh // 2, Kw // 2
    
    # Check for cached neighbor list
    cache_key = f'NativeConv3d_neighbors_{Kw}x{Kh}x{Kd}_d{dk}{dh}{dw}'
    neighbor_cache = x.get_spatial_cache(cache_key)
    
    if neighbor_cache is None:
        # Build coordinate map
        coord_map = _build_coord_map(coords)
        
        # Build neighbor lists for each voxel
        # neighbor_list[i] = [(kernel_idx, neighbor_feat_idx), ...]
        neighbor_lists = []
        
        for i in range(N):
            b = int(coords[i, 0].item())
            cx, cy, cz = int(coords[i, 1].item()), int(coords[i, 2].item()), int(coords[i, 3].item())
            neighbors = []
            
            for kd in range(Kd):
                for kh in range(Kh):
                    for kw in range(Kw):
                        nx = cx + (kd - kd_c) * dk
                        ny = cy + (kh - kh_c) * dh
                        nz = cz + (kw - kw_c) * dw
                        
                        key = (b, nx, ny, nz)
                        if key in coord_map:
                            # Store as flat kernel index and neighbor index
                            k_idx = kd * Kh * Kw + kh * Kw + kw
                            neighbors.append((k_idx, coord_map[key]))
            
            neighbor_lists.append(neighbors)
        
        neighbor_cache = neighbor_lists
        x.register_spatial_cache(cache_key, neighbor_cache)
    
    # Compute output
    output = torch.zeros(N, C_out, device=device, dtype=dtype)
    
    # Flatten weight for faster indexing: [Kd*Kh*Kw, C_out, C_in]
    weight_flat = self.weight.view(-1, C_out, self.in_channels)
    
    for i in range(N):
        for k_idx, n_idx in neighbor_cache[i]:
            # weight_flat[k_idx] has shape [C_out, C_in]
            output[i] += feats[n_idx] @ weight_flat[k_idx].T
    
    if self.bias is not None:
        output = output + self.bias
    
    return x.replace(output)


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    """
    Initialize sparse inverse 3D convolution (transposed/deconvolution).
    
    This is used in the decoder for upsampling.
    """
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    
    # Weight shape: (in_channels, kernel_d, kernel_h, kernel_w, out_channels)
    # Note: For transposed conv, we swap in/out channels
    self.weight = nn.Parameter(torch.empty((in_channels, *self.kernel_size, out_channels)))
    
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)
    
    # Initialize
    torch.nn.init.kaiming_uniform_(self.weight.view(in_channels, -1, out_channels), a=math.sqrt(5))
    if self.bias is not None:
        fan_in = in_channels * math.prod(self.kernel_size)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    """
    Forward pass for sparse inverse 3D convolution.
    
    For inverse convolution, each input voxel scatters to multiple output positions.
    This is essentially the transpose of the forward convolution.
    
    NOTE: This implementation assumes stride=1 (no actual upsampling).
    For stride>1 upsampling, the output coordinates would be different from input.
    """
    coords = x.coords  # [N, 4]
    feats = x.feats    # [N, C_in]
    
    N = coords.shape[0]
    C_out = self.weight.shape[-1]  # out_channels is last dim for inverse conv
    Kd, Kh, Kw = self.kernel_size
    dk, dh, dw = self.dilation
    device = feats.device
    dtype = feats.dtype
    
    kd_c, kh_c, kw_c = Kd // 2, Kh // 2, Kw // 2
    
    # Build coordinate map
    coord_map = _build_coord_map(coords)
    
    # For stride=1 inverse conv, output has same coordinates as input
    # Each output position accumulates from neighbors
    output = torch.zeros(N, C_out, device=device, dtype=dtype)
    
    # Flatten weight: [Kd*Kh*Kw, C_in, C_out]
    weight_flat = self.weight.view(-1, self.in_channels, C_out)
    
    # For each input voxel, scatter to its neighbors
    for i in range(N):
        b = int(coords[i, 0].item())
        cx, cy, cz = int(coords[i, 1].item()), int(coords[i, 2].item()), int(coords[i, 3].item())
        
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Neighbor coordinate
                    nx = cx + (kd - kd_c) * dk
                    ny = cy + (kh - kh_c) * dh
                    nz = cz + (kw - kw_c) * dw
                    
                    key = (b, nx, ny, nz)
                    if key in coord_map:
                        n_idx = coord_map[key]
                        k_idx = kd * Kh * Kw + kh * Kw + kw
                        # Scatter: output[neighbor] += feats[i] @ weight[k_idx]
                        output[n_idx] += feats[i] @ weight_flat[k_idx]
    
    if self.bias is not None:
        output = output + self.bias
    
    return x.replace(output)


# ============================================================================
# Vectorized implementation (faster but more memory)
# ============================================================================

def sparse_conv3d_forward_vectorized(self, x: SparseTensor) -> SparseTensor:
    """
    Vectorized implementation using batch operations.
    Faster than loop version but uses more memory.
    """
    coords = x.coords
    feats = x.feats
    
    N = coords.shape[0]
    C_in = feats.shape[1]
    C_out = self.weight.shape[0]
    Kd, Kh, Kw = self.kernel_size
    dk, dh, dw = self.dilation
    device = feats.device
    dtype = feats.dtype
    
    kd_c, kh_c, kw_c = Kd // 2, Kh // 2, Kw // 2
    
    # Build coordinate map
    coord_map = _build_coord_map(coords)
    
    # Build all neighbor pairs
    src_indices = []  # Input voxel index
    dst_indices = []  # Output voxel index (same as src for stride=1)
    kernel_indices = []  # Which kernel weight to use
    
    for i in range(N):
        b = int(coords[i, 0].item())
        cx, cy, cz = int(coords[i, 1].item()), int(coords[i, 2].item()), int(coords[i, 3].item())
        
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    nx = cx + (kd - kd_c) * dk
                    ny = cy + (kh - kh_c) * dh
                    nz = cz + (kw - kw_c) * dw
                    
                    key = (b, nx, ny, nz)
                    if key in coord_map:
                        n_idx = coord_map[key]
                        k_idx = kd * Kh * Kw + kh * Kw + kw
                        src_indices.append(n_idx)
                        dst_indices.append(i)
                        kernel_indices.append(k_idx)
    
    if len(src_indices) == 0:
        # No neighbors found
        output = torch.zeros(N, C_out, device=device, dtype=dtype)
        if self.bias is not None:
            output = output + self.bias
        return x.replace(output)
    
    # Convert to tensors
    src_indices = torch.tensor(src_indices, device=device, dtype=torch.long)
    dst_indices = torch.tensor(dst_indices, device=device, dtype=torch.long)
    kernel_indices = torch.tensor(kernel_indices, device=device, dtype=torch.long)
    
    # Gather features: [num_pairs, C_in]
    pair_feats = feats[src_indices]
    
    # Gather weights: [num_pairs, C_out, C_in]
    weight_flat = self.weight.view(-1, C_out, C_in)
    pair_weights = weight_flat[kernel_indices]
    
    # Compute contributions: [num_pairs, C_out]
    # pair_feats @ pair_weights.T -> but we need batch matmul
    contributions = torch.bmm(pair_weights, pair_feats.unsqueeze(-1)).squeeze(-1)
    
    # Scatter to output
    output = torch.zeros(N, C_out, device=device, dtype=dtype)
    output.scatter_add_(0, dst_indices.unsqueeze(-1).expand(-1, C_out), contributions)
    
    if self.bias is not None:
        output = output + self.bias
    
    return x.replace(output)