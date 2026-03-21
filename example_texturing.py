import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory

# --- STRIX HALO TRITON CHOKEPOINT (UNCHAINED) ---
import triton.runtime.jit

_original_run = triton.runtime.jit.JITFunction.run

def _amd_safe_triton_run(self, *args, **kwargs):
    # 1. Clamp warps to 8 (8 * 64 AMD threads = 512 threads per block)
    # This prevents the 2048-thread hardware rejection
    if kwargs.get('num_warps', 1) > 8:
        kwargs['num_warps'] = 8

    # 2. The AMD Zero-Grid Trap Bypass
    grid = kwargs.get('grid')
    grid_val = grid(kwargs) if callable(grid) else grid
    if grid_val and grid_val[0] == 0:
        return

    # Let everything else flow naturally
    return _original_run(self, *args, **kwargs)

triton.runtime.jit.JITFunction.run = _amd_safe_triton_run
# ------------------------------------------------

import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline

# 1. Load Pipeline
pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipeline.cuda()

# 2. Load Mesh, image & Run
mesh = trimesh.load("assets/example_texturing/the_forgotten_knight.ply")
image = Image.open("assets/example_texturing/image.webp")
output = pipeline.run(mesh, image)

# 3. Render Mesh
output.export("textured.glb", extension_webp=True)