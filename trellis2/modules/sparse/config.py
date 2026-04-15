from typing import *

CONV = 'flex_gemm'
DEBUG = False
ATTN = 'flash_attn'
# ROCm GFX1201 workaround: when True, use chunked explicit-GEMM (im2col + torch.mm) instead
# of flex_gemm Triton kernels for any sparse conv where N > ROCM_SAFE_CHUNK.
# Set ROCM_SAFE_SPCONV=1 in env to enable, or call set_rocm_safe_spconv(True).
ROCM_SAFE_SPCONV = False

def __from_env():
    import os

    global CONV
    global DEBUG
    global ATTN
    global ROCM_SAFE_SPCONV
    
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']:
        ATTN = env_sparse_attn_backend
    if os.environ.get('ROCM_SAFE_SPCONV') == '1':
        ROCM_SAFE_SPCONV = True

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}; ROCM_SAFE_SPCONV: {ROCM_SAFE_SPCONV}")
        

__from_env()
    

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'sdpa']):
    global ATTN
    ATTN = backend

def set_rocm_safe_spconv(enabled: bool):
    global ROCM_SAFE_SPCONV
    ROCM_SAFE_SPCONV = enabled
