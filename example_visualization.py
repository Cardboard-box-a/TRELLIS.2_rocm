"""
Visualization + render test for TRELLIS.2.

Run this instead of app.py to:
  1. Generate a mesh with full pipeline visualizations saved at every stage.
  2. Render the resulting mesh with render_utils.render_snapshot — the same
     call that app.py uses — and save every render frame to disk.

This is the SMOKING GUN test: if the mesh looks correct in the decode-stage
visualizations but the render images look wrong (15-30% coverage), the bug
is inside the renderer / nvdiffrast path, not in the pipeline.
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils

# ---------------------------------------------------------------------------
# Config — edit these to taste
# ---------------------------------------------------------------------------
IMAGE_PATH   = "assets/example_image/T2.png"
PIPELINE     = "1024_cascade"   # '512' | '1024' | '1024_cascade' | '1536_cascade'
SEED         = 42
NVIEWS       = 8                # number of render frames (matches app.py STEPS=8)
RENDER_RES   = 1024             # render resolution (matches app.py)
VIZ_DIR      = "visualizations_render_test"
# ---------------------------------------------------------------------------


def save_render_frames(images: dict, out_dir: str, prefix: str = "render"):
    """
    Save every key × every frame from render_snapshot output to disk.

    images is a dict like:
      {'shaded':     [np.uint8 (H,W,3), ...],
       'normal':     [...],
       'base_color': [...],
       'metallic':   [...],
       'roughness':  [...],
       'alpha':      [...]}
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for key, frames in images.items():
        for i, frame in enumerate(frames):
            path = os.path.join(out_dir, f"{prefix}_{key}_{i:03d}.png")
            img = Image.fromarray(frame)
            img.save(path)
            saved.append(path)
    return saved


def make_contact_sheet(images: dict, out_dir: str, prefix: str = "contact"):
    """
    Build one wide contact sheet per render key and save it.
    Useful for seeing all views at once without opening 40 files.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for key, frames in images.items():
        if not frames:
            continue
        row = np.concatenate(frames, axis=1)   # stack horizontally
        path = os.path.join(out_dir, f"{prefix}_{key}_all_views.png")
        Image.fromarray(row).save(path)
        paths.append(path)
        print(f"  Contact sheet [{key}]: {row.shape[1]}x{row.shape[0]}  -> {path}")
    return paths


def export_obj(mesh, path: str):
    """
    Export mesh vertices and faces directly to a Wavefront .obj file.
    Uses NO nvdiffrast, NO flex_gemm, NO ROCm GPU ops — pure Python/numpy.
    Load in Blender to verify 100% geometry completeness.
    """
    verts = mesh.vertices.detach().cpu().numpy()   # [N, 3]
    faces = mesh.faces.detach().cpu().numpy()       # [F, 3]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# TRELLIS.2 raw mesh export (no renderer, no GLB pipeline)\n")
        f.write(f"# {verts.shape[0]} vertices, {faces.shape[0]} faces\n\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for tri in faces:
            # .obj uses 1-based indices
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"  OBJ export: {verts.shape[0]} vertices, {faces.shape[0]} faces -> {path}")


def main():
    print("=" * 70)
    print("TRELLIS.2 render smoke-test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    print("\nLoading pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    # ------------------------------------------------------------------
    # Load HDR environment maps (same as app.py)
    # ------------------------------------------------------------------
    print("Loading environment maps...")
    envmap = {
        'forest': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED),
                         cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'sunset': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED),
                         cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'courtyard': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED),
                         cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
    }

    image = Image.open(IMAGE_PATH)

    # ------------------------------------------------------------------
    # Run pipeline with ALL stage visualizations enabled
    # ------------------------------------------------------------------
    print(f"\nRunning pipeline (type={PIPELINE}, seed={SEED}) ...")
    print(f"Stage visualizations will be saved to: {VIZ_DIR}/\n")

    mesh = pipeline.run(
        image,
        seed=SEED,
        pipeline_type=PIPELINE,
        visualize_sparse_structure=False,
        visualize_save_dir=VIZ_DIR,
    )

    print("\nPipeline complete. Mesh object:", type(mesh[0]).__name__)
    print(f"  vertices : {mesh[0].vertices.shape}")
    print(f"  faces    : {mesh[0].faces.shape}")
    if hasattr(mesh[0], 'coords'):
        print(f"  vox coords: {mesh[0].coords.shape}")
    if hasattr(mesh[0], 'attrs'):
        print(f"  vox attrs : {mesh[0].attrs.shape}")

    # ------------------------------------------------------------------
    # BYPASS TEST: export raw .obj — zero GPU rendering, zero nvdiffrast
    # Load in Blender to verify geometry is 100% complete before blaming
    # the renderer or GLB pipeline.
    # ------------------------------------------------------------------
    obj_path = os.path.join(VIZ_DIR, "raw_mesh.obj")
    print(f"\n{'='*70}")
    print("Exporting raw .obj (no renderer, no nvdiffrast) ...")
    print(f"{'='*70}")
    export_obj(mesh[0], obj_path)
    print(f"  -> Open {obj_path} in Blender to check geometry completeness.")
    print(f"     If this is complete but renders are 15-30%, the bug is in nvdiffrast/rasterizer.\n")

    # ------------------------------------------------------------------
    # SMOKING GUN: render with render_snapshot — identical to app.py
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Rendering {NVIEWS} views at {RENDER_RES}x{RENDER_RES} ...")
    print(f"{'='*70}\n")

    render_out_dir = os.path.join(VIZ_DIR, "render_frames")
    os.makedirs(render_out_dir, exist_ok=True)

    images = render_utils.render_snapshot(
        mesh[0],                  # same single-mesh call as app.py
        resolution=RENDER_RES,
        r=2,
        fov=36,
        nviews=NVIEWS,
        envmap=envmap,
    )

    print(f"\nRender keys returned: {list(images.keys())}")
    for key, frames in images.items():
        print(f"  {key}: {len(frames)} frames, each {frames[0].shape}")

    # Save every individual frame
    print(f"\nSaving individual frames to {render_out_dir}/ ...")
    saved = save_render_frames(images, render_out_dir, prefix="render")
    print(f"  Saved {len(saved)} frame files.")

    # Save contact sheets (one image per render key, all views side by side)
    print(f"\nBuilding contact sheets ...")
    make_contact_sheet(images, render_out_dir, prefix="contact")

    print(f"\n{'='*70}")
    print(f"All done.  Check {render_out_dir}/ for render output.")
    print(f"  contact_shaded_all_views.png      <- the key one to look at")
    print(f"  contact_base_color_all_views.png  <- color without lighting")
    print(f"  contact_normal_all_views.png      <- surface normals")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
