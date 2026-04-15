from typing import *
import torch
from ..voxel import Voxel
import cumesh
from flex_gemm.ops.grid_sample import grid_sample_3d
from ...utils.pipeline_logger import get_logger, log_mesh, elapsed


class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs
        
    @property
    def device(self):
        return self.vertices.device
        
    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )
        
    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)
        
    def cpu(self):
        return self.to('cpu')
    
    def fill_holes(self, max_hole_perimeter=3e-2):
        import os, numpy as np
        L = get_logger()
        log_mesh(self.vertices, self.faces, "fill_holes:before")
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()

        # ------------------------------------------------------------------ #
        # Debug helpers: per-step .obj dump + stats print
        # ------------------------------------------------------------------ #
        _dbg_dir = os.environ.get("CUMESH_DEBUG_DIR", "cumesh_debug")
        _dbg_step = [0]

        def _snap(label, v_tensor, f_tensor):
            return
            """Dump vertex/face data to an OBJ and print min/max/nan stats."""
            v = v_tensor.detach().cpu().float().numpy()  # [N, 3]
            f = f_tensor.detach().cpu().int().numpy()    # [M, 3]
            step = _dbg_step[0]
            _dbg_step[0] += 1

            vmin = v.min(axis=0) if len(v) else [float('nan')]*3
            vmax = v.max(axis=0) if len(v) else [float('nan')]*3
            all_zero_v = bool((v == 0).all()) if len(v) else True
            all_zero_f = bool((f == 0).all()) if len(f) else True
            nan_v = bool(np.isnan(v).any())

            print(f"[CUMESH_DBG] step={step:02d} {label}")
            print(f"  verts : {v.shape[0]}  min={vmin}  max={vmax}  all_zero={all_zero_v}  nan={nan_v}")
            print(f"  faces : {f.shape[0]}  all_zero={all_zero_f}")

            os.makedirs(_dbg_dir, exist_ok=True)
            obj_path = os.path.join(_dbg_dir, f"step{step:02d}_{label.replace(':', '_').replace('/', '_')}.obj")
            with open(obj_path, "w") as fp:
                fp.write(f"# step={step} {label}\n")
                fp.write(f"# {v.shape[0]} vertices, {f.shape[0]} faces\n\n")
                for row in v:
                    fp.write(f"v {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
                fp.write("\n")
                for row in f:
                    fp.write(f"f {row[0]+1} {row[1]+1} {row[2]+1}\n")
            print(f"  -> {obj_path}")

        def _snap_mesh(label):
            return
            """Read current CuMesh state and dump it."""
            v, f = mesh.read()
            _snap(label, v, f)
        # ------------------------------------------------------------------ #

        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        _snap("00_after_init", vertices, faces)

        mesh.get_edges()
        _snap_mesh("01_after_get_edges")

        mesh.get_boundary_info()
        L.info(f"  {elapsed()} fill_holes: num_boundaries={mesh.num_boundaries}")
        _snap_mesh("02_after_get_boundary_info")

        if mesh.num_boundaries == 0:
            L.info(f"  {elapsed()} fill_holes: no boundaries, skipping")
            return

        mesh.get_vertex_edge_adjacency()
        _snap_mesh("03_after_get_vertex_edge_adjacency")

        mesh.get_vertex_boundary_adjacency()
        _snap_mesh("04_after_get_vertex_boundary_adjacency")

        mesh.get_manifold_boundary_adjacency()
        _snap_mesh("05_after_get_manifold_boundary_adjacency")

        mesh.read_manifold_boundary_adjacency()
        _snap_mesh("06_after_read_manifold_boundary_adjacency")

        mesh.get_boundary_connected_components()
        _snap_mesh("07_after_get_boundary_connected_components")

        mesh.get_boundary_loops()
        L.info(f"  {elapsed()} fill_holes: num_boundary_loops={mesh.num_boundary_loops}")
        _snap_mesh("08_after_get_boundary_loops")

        if mesh.num_boundary_loops == 0:
            return

        mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
        _snap_mesh("09_after_fill_holes")

        new_vertices, new_faces = mesh.read()
        _snap("10_final_read", new_vertices, new_faces)
        log_mesh(new_vertices, new_faces, "fill_holes:after")

        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)
        
    def remove_faces(self, face_mask: torch.Tensor):
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()
        
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.remove_faces(face_mask)
        new_vertices, new_faces = mesh.read()
        
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)
        
    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):
        L = get_logger()
        log_mesh(self.vertices, self.faces, f"simplify:before(target={target})")
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()

        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(target, verbose=verbose, options=options)
        new_vertices, new_faces = mesh.read()
        log_mesh(new_vertices, new_faces, "simplify:after")

        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)


class TextureFilterMode:
    CLOSEST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


class AlphaMode:
    OPAQUE = 0
    MASK = 1
    BLEND = 2


class Texture:
    def __init__(
        self,
        image: torch.Tensor,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        wrap_mode: TextureWrapMode = TextureWrapMode.REPEAT
    ):
        self.image = image
        self.filter_mode = filter_mode
        self.wrap_mode = wrap_mode

    def to(self, device, non_blocking=False):
        return Texture(
            self.image.to(device, non_blocking=non_blocking),
            self.filter_mode,
            self.wrap_mode,
        )


class PbrMaterial:
    def __init__(
        self,
        base_color_texture: Optional[Texture] = None,
        base_color_factor: Union[torch.Tensor, List[float]] = [1.0, 1.0, 1.0],
        metallic_texture: Optional[Texture] = None,
        metallic_factor: float = 1.0,
        roughness_texture: Optional[Texture] = None,
        roughness_factor: float = 1.0,
        alpha_texture: Optional[Texture] = None,
        alpha_factor: float = 1.0,
        alpha_mode: AlphaMode = AlphaMode.OPAQUE,
        alpha_cutoff: float = 0.5,
    ):
        self.base_color_texture = base_color_texture
        self.base_color_factor = torch.tensor(base_color_factor, dtype=torch.float32)[:3]
        self.metallic_texture = metallic_texture
        self.metallic_factor = metallic_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.alpha_texture = alpha_texture
        self.alpha_factor = alpha_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff

    def to(self, device, non_blocking=False):
        return PbrMaterial(
            base_color_texture=self.base_color_texture.to(device, non_blocking=non_blocking) if self.base_color_texture is not None else None,
            base_color_factor=self.base_color_factor.to(device, non_blocking=non_blocking),
            metallic_texture=self.metallic_texture.to(device, non_blocking=non_blocking) if self.metallic_texture is not None else None,
            metallic_factor=self.metallic_factor,
            roughness_texture=self.roughness_texture.to(device, non_blocking=non_blocking) if self.roughness_texture is not None else None,
            roughness_factor=self.roughness_factor,
            alpha_texture=self.alpha_texture.to(device, non_blocking=non_blocking) if self.alpha_texture is not None else None,
            alpha_factor=self.alpha_factor,
            alpha_mode=self.alpha_mode,
            alpha_cutoff=self.alpha_cutoff,
        )


class MeshWithPbrMaterial(Mesh):
    def __init__(self,
        vertices,
        faces,
        material_ids,
        uv_coords,
        materials: List[PbrMaterial],
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.material_ids = material_ids    # [M]
        self.uv_coords = uv_coords          # [M, 3, 2]
        self.materials = materials
        self.layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    def to(self, device, non_blocking=False):
        return MeshWithPbrMaterial(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.material_ids.to(device, non_blocking=non_blocking),
            self.uv_coords.to(device, non_blocking=non_blocking),
            [material.to(device, non_blocking=non_blocking) for material in self.materials],
        )


class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )
        
    def query_attrs(self, xyz):
        grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
        vertex_attrs = grid_sample_3d(
            self.attrs,
            torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
            self.voxel_shape,
            grid,
            mode='trilinear'
        )[0]
        return vertex_attrs
        
    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)
