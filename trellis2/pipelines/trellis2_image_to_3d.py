from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel
from ..utils.pipeline_logger import get_logger, log_sparse, log_mesh, log_tensor, section, elapsed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        visualize_hr_coords: bool = False,
        visualize_save_dir: str = None,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            visualize_hr_coords (bool): Whether to visualize high-resolution coordinates after upsampling.
            visualize_save_dir (str): Directory to save visualization images. If None, displays interactively.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        get_logger().debug(f"DEBUG SLAT: coords={slat.coords.shape}, spatial_shape={slat.spatial_shape}, "
              f"coords_max={slat.coords[:,1:].max(dim=0).values}, dtype={slat.feats.dtype}")
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        get_logger().debug(f"DEBUG SLAT[after *std + mean]: coords={slat.coords.shape}, spatial_shape={slat.spatial_shape}, "
              f"coords_max={slat.coords[:,1:].max(dim=0).values}, dtype={slat.feats.dtype}")
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        get_logger().debug(f"DEBUG CASCADE: hr_coords shape={hr_coords.shape}, max={hr_coords[:,1:].max(dim=0).values}, unique_x={hr_coords[:,1].unique().shape[0]}, unique_y={hr_coords[:,2].unique().shape[0]}, unique_z={hr_coords[:,3].unique().shape[0]}")
        
        # Visualize high-resolution coordinates if requested
        if visualize_hr_coords:
            print("\n=== High-Resolution Coordinates Visualization (After Upsampling) ===")
            self.analyze_sparse_structure(hr_coords)
            
            # Calculate effective resolution for visualization
            effective_resolution = lr_resolution * 4  # upsample_times=4
            
            if visualize_save_dir:
                import os
                os.makedirs(visualize_save_dir, exist_ok=True)
                base_path = os.path.join(visualize_save_dir, f"hr_coords_{resolution}_upsampled")
                
                self.visualize_sparse_structure_matplotlib(
                    hr_coords, 
                    title=f"HR Coordinates - Upsampled {resolution} (effective res: {effective_resolution})",
                    save_path=f"{base_path}_3d.png"
                )
                
                self.visualize_sparse_structure_voxel(
                    hr_coords,
                    resolution=effective_resolution,
                    title=f"HR Voxel Grid - Upsampled {resolution} (effective res: {effective_resolution})",
                    save_path=f"{base_path}_voxel.png"
                )
                
                self.visualize_sparse_structure_projections(
                    hr_coords,
                    resolution=effective_resolution,
                    title=f"HR Projections - Upsampled {resolution} (effective res: {effective_resolution})",
                    save_path=f"{base_path}_projections.png"
                )
                
                self.visualize_sparse_structure_multi_view(
                    hr_coords,
                    title=f"HR Multi-View - Upsampled {resolution} (effective res: {effective_resolution})",
                    save_path=f"{base_path}_multi_view.png"
                )
            else:
                # Interactive visualization (no saving)
                self.visualize_sparse_structure_matplotlib(
                    hr_coords, 
                    title=f"HR Coordinates - Upsampled {resolution} (effective res: {effective_resolution})"
                )
                
                self.visualize_sparse_structure_voxel(
                    hr_coords,
                    resolution=effective_resolution,
                    title=f"HR Voxel Grid - Upsampled {resolution} (effective res: {effective_resolution})"
                )
                
                self.visualize_sparse_structure_projections(
                    hr_coords,
                    resolution=effective_resolution,
                    title=f"HR Projections - Upsampled {resolution} (effective res: {effective_resolution})"
                )
                
                self.visualize_sparse_structure_multi_view(
                    hr_coords,
                    title=f"HR Multi-View - Upsampled {resolution} (effective res: {effective_resolution})"
                )
            print("=== HR Coordinates Visualization Complete ===\n")
        
        coord_set = set(map(tuple, hr_coords[:, 1:].cpu().numpy().tolist()))
        has_neighbor = sum(1 for c in coord_set if any(
            (c[0]+dx, c[1]+dy, c[2]+dz) in coord_set 
            for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        )) / len(coord_set)
        get_logger().debug(f"DEBUG TOPOLOGY: coords={len(coord_set)}, neighbor_coverage={has_neighbor:.3f}")

        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            get_logger().debug(f"DEBUG COORDS: num_tokens={coords.shape[0]}, max={coords[:,1:].max(dim=0).values}")
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Visualize quantized coordinates if requested
        if visualize_hr_coords:
            print("\n=== Quantized Coordinates Visualization (After Resolution Adjustment) ===")
            self.analyze_sparse_structure(coords)
            
            if visualize_save_dir:
                import os
                os.makedirs(visualize_save_dir, exist_ok=True)
                base_path = os.path.join(visualize_save_dir, f"quantized_coords_{hr_resolution}")
                
                self.visualize_sparse_structure_matplotlib(
                    coords, 
                    title=f"Quantized Coords - Resolution {hr_resolution}",
                    save_path=f"{base_path}_3d.png"
                )
                
                self.visualize_sparse_structure_voxel(
                    coords,
                    resolution=hr_resolution // 16,
                    title=f"Quantized Voxel Grid - Resolution {hr_resolution}",
                    save_path=f"{base_path}_voxel.png"
                )
                
                self.visualize_sparse_structure_projections(
                    coords,
                    resolution=hr_resolution // 16,
                    title=f"Quantized Projections - Resolution {hr_resolution}",
                    save_path=f"{base_path}_projections.png"
                )
                
                self.visualize_sparse_structure_multi_view(
                    coords,
                    title=f"Quantized Multi-View - Resolution {hr_resolution}",
                    save_path=f"{base_path}_multi_view.png"
                )
            else:
                # Interactive visualization (no saving)
                self.visualize_sparse_structure_matplotlib(
                    coords, 
                    title=f"Quantized Coords - Resolution {hr_resolution}"
                )
                
                self.visualize_sparse_structure_voxel(
                    coords,
                    resolution=hr_resolution // 16,
                    title=f"Quantized Voxel Grid - Resolution {hr_resolution}"
                )
                
                self.visualize_sparse_structure_projections(
                    coords,
                    resolution=hr_resolution // 16,
                    title=f"Quantized Projections - Resolution {hr_resolution}"
                )
                
                self.visualize_sparse_structure_multi_view(
                    coords,
                    title=f"Quantized Multi-View - Resolution {hr_resolution}"
                )
            print("=== Quantized Coordinates Visualization Complete ===\n")
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples

        

        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        get_logger().debug(f"CASCADE final slat: nan={torch.isnan(slat.feats).any().item()} inf={torch.isinf(slat.feats).any().item()} max={slat.feats.abs().max().item():.4f} dtype={slat.feats.dtype}")
        
        # Visualize final SLat features if requested
        if visualize_hr_coords:
            print("\n=== Final SLat Features Visualization (After Denormalization) ===")
            self.analyze_slat_features(slat)
            
            if visualize_save_dir:
                import os
                os.makedirs(visualize_save_dir, exist_ok=True)
                base_path = os.path.join(visualize_save_dir, f"final_slat_{hr_resolution}")
                
                # Visualize first few features
                for i in range(min(3, slat.feats.shape[1])):
                    self.visualize_slat_features(
                        slat,
                        title=f"Final SLat Feature {i} - Resolution {hr_resolution}",
                        save_path=f"{base_path}_feature{i}.png",
                        feature_idx=i
                    )
            else:
                # Interactive visualization (no saving)
                for i in range(min(3, slat.feats.shape[1])):
                    self.visualize_slat_features(
                        slat,
                        title=f"Final SLat Feature {i} - Resolution {hr_resolution}",
                        feature_idx=i
                    )
            print("=== Final SLat Features Visualization Complete ===\n")
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        visualize: bool = False,
        visualize_save_dir: str = None,
        pipeline_type: str = 'unknown',
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
            visualize (bool): Whether to visualize shape + colored texture slat.
            visualize_save_dir (str): Directory to save visualizations. None = interactive.
            pipeline_type (str): Pipeline name used in visualization titles.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_norm = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat_norm.replace(feats=torch.randn(shape_slat_norm.coords.shape[0], in_channels - shape_slat_norm.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat_norm,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples

        if self.low_vram:
            flow_model.cpu()

        # Visualize: shape structure + colored texture slat
        if visualize:
            import os
            print("\n=== Texture SLat Visualization ===")
            self.analyze_slat_features(slat)

            if visualize_save_dir:
                os.makedirs(visualize_save_dir, exist_ok=True)
                base_path = os.path.join(visualize_save_dir, f"tex_slat_{pipeline_type}")

                # 1. Shape-only structure (occupancy/geometry)
                self.visualize_sparse_structure_projections(
                    shape_slat.coords,
                    title=f"Shape Structure - {pipeline_type}",
                    save_path=f"{base_path}_shape_projections.png",
                )

                # 2. Combined: shape colored by tex-slat latent features (pseudo-RGB from first 3 dims)
                self.visualize_tex_slat_colored(
                    slat,
                    title=f"Tex SLat Colored - {pipeline_type}",
                    save_path=f"{base_path}_colored.png",
                )

                # 3. Per-feature projections (first 3 latent dims)
                for i in range(min(3, slat.feats.shape[1])):
                    self.visualize_slat_features(
                        slat,
                        title=f"Tex Feature {i} - {pipeline_type}",
                        save_path=f"{base_path}_feature{i}.png",
                        feature_idx=i,
                    )
            else:
                self.visualize_sparse_structure_projections(
                    shape_slat.coords,
                    title=f"Shape Structure - {pipeline_type}",
                )
                self.visualize_tex_slat_colored(
                    slat,
                    title=f"Tex SLat Colored - {pipeline_type}",
                )
                for i in range(min(3, slat.feats.shape[1])):
                    self.visualize_slat_features(
                        slat,
                        title=f"Tex Feature {i} - {pipeline_type}",
                        feature_idx=i,
                    )
            print("=== Texture SLat Visualization Complete ===\n")

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    def visualize_sparse_structure_matplotlib(self, coords: torch.Tensor, title: str = "Sparse Structure", save_path: str = None):
        """
        Visualize sparse structure coordinates using matplotlib 3D scatter plot.
        
        Args:
            coords: torch.Tensor of shape [N, 4] with [batch, x, y, z]
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        # Convert to numpy and extract spatial coordinates (drop batch index)
        coords_np = coords.cpu().numpy()
        x = coords_np[:, 1]  # x coordinate
        y = coords_np[:, 2]  # y coordinate  
        z = coords_np[:, 3]  # z coordinate
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title}\n{len(coords)} occupied voxels')
        
        # Add colorbar
        plt.colorbar(scatter, label='Z coordinate')
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved matplotlib visualization to {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_sparse_structure_voxel(self, coords: torch.Tensor, resolution: int = 32, title: str = "Sparse Structure", save_path: str = None):
        """
        Visualize sparse structure as a 3D voxel grid.
        
        Args:
            coords: torch.Tensor of shape [N, 4] with [batch, x, y, z]
            resolution: Grid resolution (e.g., 32 for 32³ grid)
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        # Create empty 3D grid
        grid = np.zeros((resolution, resolution, resolution), dtype=bool)
        
        # Fill in occupied voxels
        coords_np = coords.cpu().numpy()
        for coord in coords_np:
            _, x, y, z = coord
            if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                grid[x, y, z] = True
        
        # Get coordinates of occupied voxels
        x, y, z = np.where(grid)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot voxels
        ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.3)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title}\n{len(coords)} occupied voxels / {resolution**3} total')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved voxel visualization to {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_sparse_structure_projections(self, coords: torch.Tensor, resolution: int = 32, title: str = "Sparse Structure", save_path: str = None):
        """
        Visualize sparse structure using 2D projections (XY, XZ, YZ planes).
        
        Args:
            coords: torch.Tensor of shape [N, 4] with [batch, x, y, z]
            resolution: Grid resolution
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        coords_np = coords.cpu().numpy()
        x = coords_np[:, 1]
        y = coords_np[:, 2]
        z = coords_np[:, 3]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY projection (looking down Z axis)
        axes[0].scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('XY Projection (Top View)')
        axes[0].set_xlim(0, resolution)
        axes[0].set_ylim(0, resolution)
        axes[0].set_aspect('equal')
        
        # XZ projection (looking down Y axis)
        axes[1].scatter(x, z, c=y, cmap='viridis', s=1, alpha=0.5)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        axes[1].set_title('XZ Projection (Side View)')
        axes[1].set_xlim(0, resolution)
        axes[1].set_ylim(0, resolution)
        axes[1].set_aspect('equal')
        
        # YZ projection (looking down X axis)
        axes[2].scatter(y, z, c=x, cmap='viridis', s=1, alpha=0.5)
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        axes[2].set_title('YZ Projection (Front View)')
        axes[2].set_xlim(0, resolution)
        axes[2].set_ylim(0, resolution)
        axes[2].set_aspect('equal')
        
        plt.suptitle(f'{title}\n{len(coords)} occupied voxels', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved projections visualization to {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_sparse_structure_multi_view(self, coords: torch.Tensor, title: str = "Sparse Structure", save_path: str = None):
        """
        Visualize sparse structure with multiple views (3D + 2D projections).
        
        Args:
            coords: torch.Tensor of shape [N, 4] with [batch, x, y, z]
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        coords_np = coords.cpu().numpy()
        x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
        
        # Create multi-view visualization
        fig = plt.figure(figsize=(18, 6))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
        ax1.set_title('3D View')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        
        # XY projection
        ax2 = fig.add_subplot(132)
        ax2.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
        ax2.set_title('XY Projection')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        
        # XZ projection
        ax3 = fig.add_subplot(133)
        ax3.scatter(x, z, c=y, cmap='viridis', s=1, alpha=0.5)
        ax3.set_title('XZ Projection')
        ax3.set_xlabel('X'); ax3.set_ylabel('Z')
        ax3.set_aspect('equal')
        
        plt.suptitle(f'{title}\n{len(coords)} occupied voxels', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved multi-view visualization to {save_path}")
        
        plt.show()
        plt.close()
    
    def analyze_sparse_structure(self, coords: torch.Tensor):
        """
        Analyze and print statistics about the sparse structure.
        
        Args:
            coords: torch.Tensor of shape [N, 4]
        """
        coords_np = coords.cpu().numpy()
        x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
        
        print(f"Sparse Structure Analysis:")
        print(f"  Total occupied voxels: {len(coords)}")
        print(f"  X range: [{x.min()}, {x.max()}]")
        print(f"  Y range: [{y.min()}, {y.max()}]")
        print(f"  Z range: [{z.min()}, {z.max()}]")
        print(f"  Center: [{x.mean():.1f}, {y.mean():.1f}, {z.mean():.1f}]")
        print(f"  Std dev: [{x.std():.1f}, {y.std():.1f}, {z.std():.1f}]")
        print(f"  Bounding box volume: {(x.max()-x.min()) * (y.max()-y.min()) * (z.max()-z.min())}")
    
    def visualize_slat_features(self, slat: SparseTensor, title: str = "SLat Features", save_path: str = None, feature_idx: int = 0):
        """
        Visualize features from a SparseTensor (shape SLat).
        
        Args:
            slat: SparseTensor with features at sparse coordinates
            title: Title for the plot
            save_path: Optional path to save the figure
            feature_idx: Which feature dimension to visualize (default: 0)
        """
        coords_np = slat.coords.cpu().numpy()
        feats_np = slat.feats.cpu().numpy()
        
        # Extract coordinates and selected feature
        x = coords_np[:, 1]
        y = coords_np[:, 2]
        z = coords_np[:, 3]
        feature_values = feats_np[:, feature_idx]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points colored by feature value
        scatter = ax.scatter(x, y, z, c=feature_values, cmap='viridis', s=1, alpha=0.6)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title}\nFeature {feature_idx} | Range: [{feature_values.min():.3f}, {feature_values.max():.3f}]')
        
        # Add colorbar
        plt.colorbar(scatter, label=f'Feature {feature_idx} Value')
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved SLat feature visualization to {save_path}")
        
        plt.show()
        plt.close()

    def visualize_tex_slat_colored(self, slat: SparseTensor, title: str = "Tex SLat Colored", save_path: str = None):
        """
        Visualize texture SLat with points colored by the first 3 latent feature dimensions
        mapped to R, G, B — giving a pseudo-color view of the texture distribution across the shape.

        Also shows three 2D projection panels (XY/XZ/YZ) beside the 3D view so you can see
        coverage completeness at a glance.

        Args:
            slat: SparseTensor with texture latent features [N, C]
            title: Title for the plot
            save_path: Optional path to save the figure. None = interactive display.
        """
        import numpy as np

        coords_np = slat.coords.cpu().float().numpy()
        feats_np = slat.feats.cpu().float().numpy()

        x = coords_np[:, 1]
        y = coords_np[:, 2]
        z = coords_np[:, 3]

        # Build per-point RGB from first 3 feature dims, normalised to [0, 1]
        n_color_dims = min(3, feats_np.shape[1])
        rgb = feats_np[:, :n_color_dims].copy()
        for ch in range(n_color_dims):
            lo, hi = rgb[:, ch].min(), rgb[:, ch].max()
            rgb[:, ch] = (rgb[:, ch] - lo) / (hi - lo + 1e-8)
        if n_color_dims < 3:
            pad = np.ones((rgb.shape[0], 3 - n_color_dims))
            rgb = np.concatenate([rgb, pad], axis=1)
        rgb = np.clip(rgb, 0.0, 1.0)

        fig = plt.figure(figsize=(22, 6))
        fig.suptitle(f'{title}  ({len(x)} voxels, {feats_np.shape[1]} feat dims)', fontsize=13)

        # 3D scatter coloured by pseudo-RGB
        ax3d = fig.add_subplot(141, projection='3d')
        ax3d.scatter(x, y, z, c=rgb, s=1, alpha=0.6)
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        ax3d.set_title('3D (pseudo-RGB)')

        # XY projection
        ax_xy = fig.add_subplot(142)
        ax_xy.scatter(x, y, c=rgb, s=1, alpha=0.5)
        ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
        ax_xy.set_title('XY (top)')
        ax_xy.set_aspect('equal')

        # XZ projection
        ax_xz = fig.add_subplot(143)
        ax_xz.scatter(x, z, c=rgb, s=1, alpha=0.5)
        ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z')
        ax_xz.set_title('XZ (side)')
        ax_xz.set_aspect('equal')

        # YZ projection
        ax_yz = fig.add_subplot(144)
        ax_yz.scatter(y, z, c=rgb, s=1, alpha=0.5)
        ax_yz.set_xlabel('Y'); ax_yz.set_ylabel('Z')
        ax_yz.set_title('YZ (front)')
        ax_yz.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved tex-slat colored visualization to {save_path}")

        plt.show()
        plt.close()

    def visualize_decoded_mesh(self, mesh, title: str = "Decoded Mesh", save_path_prefix: str = None):
        """
        Visualize a decoded triangle mesh (vertices + faces).

        Renders four panels:
          - 3D scatter of vertices coloured by Z (subsampled to ≤50k points so matplotlib doesn't choke)
          - XY / XZ / YZ 2D projections

        Saves four separate PNGs when save_path_prefix is given (one per panel style matches
        the naming convention used elsewhere in the pipeline):
          <prefix>_3d.png, <prefix>_projections.png
        """
        import numpy as np
        import os

        verts = mesh.vertices.cpu().float().numpy()  # [V, 3]
        n_verts = verts.shape[0]
        n_faces = mesh.faces.shape[0]

        MAX_SCATTER = 50_000
        if n_verts > MAX_SCATTER:
            idx = np.random.choice(n_verts, MAX_SCATTER, replace=False)
            v = verts[idx]
        else:
            v = verts
        x, y, z = v[:, 0], v[:, 1], v[:, 2]

        subtitle = f"{n_verts:,} vertices  {n_faces:,} faces" + (
            f"  (scatter: {len(x):,} sampled)" if n_verts > MAX_SCATTER else "")

        # --- 3D scatter ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{title}\n{subtitle}')
        plt.tight_layout()
        if save_path_prefix:
            p = f"{save_path_prefix}_3d.png"
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"Saved decoded mesh 3D to {p}")
        plt.show(); plt.close()

        # --- 3-panel 2D projections ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_title('XY (top)')
        axes[0].set_aspect('equal')
        axes[1].scatter(x, z, c=y, cmap='viridis', s=1, alpha=0.5)
        axes[1].set_xlabel('X'); axes[1].set_ylabel('Z'); axes[1].set_title('XZ (side)')
        axes[1].set_aspect('equal')
        axes[2].scatter(y, z, c=x, cmap='viridis', s=1, alpha=0.5)
        axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z'); axes[2].set_title('YZ (front)')
        axes[2].set_aspect('equal')
        plt.suptitle(f'{title}\n{subtitle}', fontsize=13)
        plt.tight_layout()
        if save_path_prefix:
            p = f"{save_path_prefix}_projections.png"
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"Saved decoded mesh projections to {p}")
        plt.show(); plt.close()

    def visualize_mesh_with_voxel(self, mv, title: str = "MeshWithVoxel", save_path_prefix: str = None):
        """
        Visualize a MeshWithVoxel: overlays mesh vertices (grey) and texture voxel positions
        (coloured by pseudo-RGB from first 3 attr dims) in one 5-panel figure.

        Panels: 3D overlay, XY / XZ / YZ 2D projections.
        """
        import numpy as np

        verts = mv.vertices.cpu().float().numpy()
        n_verts = verts.shape[0]
        n_faces = mv.faces.shape[0]
        coords = mv.coords.cpu().float().numpy()   # [N, 3]  (already stripped of batch dim)
        attrs = mv.attrs.cpu().float().numpy()     # [N, C]
        n_vox = coords.shape[0]

        MAX_SCATTER = 50_000
        if n_verts > MAX_SCATTER:
            vi = np.random.choice(n_verts, MAX_SCATTER, replace=False)
            vp = verts[vi]
        else:
            vp = verts
        if n_vox > MAX_SCATTER:
            ci = np.random.choice(n_vox, MAX_SCATTER, replace=False)
            cp = coords[ci]; ap = attrs[ci]
        else:
            cp = coords; ap = attrs

        # Build pseudo-RGB from first 3 attr dims
        n_color = min(3, ap.shape[1])
        rgb = ap[:, :n_color].copy()
        for ch in range(n_color):
            lo, hi = rgb[:, ch].min(), rgb[:, ch].max()
            rgb[:, ch] = (rgb[:, ch] - lo) / (hi - lo + 1e-8)
        if n_color < 3:
            rgb = np.concatenate([rgb, np.ones((rgb.shape[0], 3 - n_color))], axis=1)
        rgb = np.clip(rgb, 0, 1)

        subtitle = (f"Mesh: {n_verts:,}v {n_faces:,}f  |  Voxels: {n_vox:,}"
                    + ("  (both subsampled)" if n_verts > MAX_SCATTER or n_vox > MAX_SCATTER else ""))

        # Voxel coords are integer indices; convert to world space for overlay
        vox_world = cp * mv.voxel_size + mv.origin.cpu().numpy()
        vx, vy, vz = vox_world[:, 0], vox_world[:, 1], vox_world[:, 2]
        mx, my, mz = vp[:, 0], vp[:, 1], vp[:, 2]

        # --- 3D overlay ---
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mx, my, mz, c='lightgrey', s=1, alpha=0.3, label='mesh verts')
        ax.scatter(vx, vy, vz, c=rgb, s=2, alpha=0.6, label='tex voxels')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{title}\n{subtitle}')
        plt.tight_layout()
        if save_path_prefix:
            p = f"{save_path_prefix}_3d.png"
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"Saved MeshWithVoxel 3D to {p}")
        plt.show(); plt.close()

        # --- 4-panel 2D projections ---
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        def proj(ax_, hx, hy, hz, label):
            ax_.scatter(hx, hy, c='lightgrey', s=1, alpha=0.25)
            ax_.scatter(vx if label == 'XY' else (vx if label == 'XZ' else vy),
                        vy if label == 'XY' else (vz if label == 'XZ' else vz),
                        c=rgb, s=1, alpha=0.5)
            ax_.set_aspect('equal')

        axes[0].scatter(mx, my, c='lightgrey', s=1, alpha=0.25)
        axes[0].scatter(vx, vy, c=rgb, s=1, alpha=0.5)
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_title('XY (top)'); axes[0].set_aspect('equal')

        axes[1].scatter(mx, mz, c='lightgrey', s=1, alpha=0.25)
        axes[1].scatter(vx, vz, c=rgb, s=1, alpha=0.5)
        axes[1].set_xlabel('X'); axes[1].set_ylabel('Z'); axes[1].set_title('XZ (side)'); axes[1].set_aspect('equal')

        axes[2].scatter(my, mz, c='lightgrey', s=1, alpha=0.25)
        axes[2].scatter(vy, vz, c=rgb, s=1, alpha=0.5)
        axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z'); axes[2].set_title('YZ (front)'); axes[2].set_aspect('equal')

        # 4th panel: voxel coverage ratio as bar chart per axis
        axes[3].axis('off')
        info = (f"Mesh vertices : {n_verts:,}\n"
                f"Mesh faces    : {n_faces:,}\n"
                f"Tex voxels    : {n_vox:,}\n"
                f"Voxel size    : {mv.voxel_size:.5f}\n"
                f"Voxel world X : [{vx.min():.3f}, {vx.max():.3f}]\n"
                f"Voxel world Y : [{vy.min():.3f}, {vy.max():.3f}]\n"
                f"Voxel world Z : [{vz.min():.3f}, {vz.max():.3f}]\n"
                f"Attr dims     : {mv.attrs.shape[1]}\n"
                f"Attr range    : [{mv.attrs.min().item():.4f}, {mv.attrs.max().item():.4f}]")
        axes[3].text(0.05, 0.95, info, transform=axes[3].transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[3].set_title('Stats')

        plt.suptitle(f'{title}\n{subtitle}', fontsize=13)
        plt.tight_layout()
        if save_path_prefix:
            p = f"{save_path_prefix}_projections.png"
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"Saved MeshWithVoxel projections to {p}")
        plt.show(); plt.close()

    def analyze_slat_features(self, slat: SparseTensor):
        """
        Analyze and print statistics about SLat features.
        
        Args:
            slat: SparseTensor with features
        """
        coords_np = slat.coords.cpu().numpy()
        feats_np = slat.feats.cpu().numpy()
        
        print(f"\nSLat Features Analysis:")
        print(f"  Number of tokens: {slat.coords.shape[0]}")
        print(f"  Feature dimensions: {slat.feats.shape[1]}")
        print(f"  Feature statistics:")
        for i in range(min(5, slat.feats.shape[1])):  # Show first 5 features
            feat = feats_np[:, i]
            print(f"    Feature {i}: min={feat.min():.4f}, max={feat.max():.4f}, mean={feat.mean():.4f}, std={feat.std():.4f}")
        
        print(f"  NaN values: {np.isnan(feats_np).any()}")
        print(f"  Inf values: {np.isinf(feats_np).any()}")
        print(f"  Coordinate range: X=[{coords_np[:, 1].min()}, {coords_np[:, 1].max()}], "
              f"Y=[{coords_np[:, 2].min()}, {coords_np[:, 2].max()}], "
              f"Z=[{coords_np[:, 3].min()}, {coords_np[:, 3].max()}]")
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
        visualize: bool = False,
        visualize_save_dir: str = None,
        pipeline_type: str = 'unknown',
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        L = get_logger()
        section(f"decode_latent  resolution={resolution}")

        section("decode_shape_slat")
        log_sparse(shape_slat, "shape_slat-in")
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        L.info(f"  {elapsed()} decode_shape_slat produced {len(meshes)} mesh(es)")
        for i, m in enumerate(meshes):
            log_mesh(m.vertices, m.faces, f"shape_mesh[{i}]")

        # Visualize decoded shape meshes
        if visualize:
            import os
            for i, m in enumerate(meshes):
                base = (os.path.join(visualize_save_dir, f"decoded_mesh_{pipeline_type}_s{i}")
                        if visualize_save_dir else None)
                if base:
                    os.makedirs(visualize_save_dir, exist_ok=True)
                self.visualize_decoded_mesh(m, title=f"Decoded Shape Mesh [{i}] - {pipeline_type}",
                                            save_path_prefix=base)

        section("decode_tex_slat")
        log_sparse(tex_slat, "tex_slat-in")
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        L.info(f"  {elapsed()} decode_tex_slat produced {len(tex_voxels)} voxel set(s)")

        #Commented temporarily for speed.
        """
        # Visualize texture voxels
        if visualize:
            import os
            for i, v in enumerate(tex_voxels):
                base = (os.path.join(visualize_save_dir, f"tex_voxels_{pipeline_type}_s{i}")
                        if visualize_save_dir else None)
                if base:
                    os.makedirs(visualize_save_dir, exist_ok=True)
                self.visualize_tex_slat_colored(v,
                    title=f"Tex Voxels [{i}] - {pipeline_type}",
                    save_path=f"{base}_colored.png" if base else None)
        """

        section("build MeshWithVoxel")
        out_mesh = []
        for i, (m, v) in enumerate(zip(meshes, tex_voxels)):
            L.info(f"  {elapsed()} sample {i}:")
            log_sparse(v, f"tex_voxels[{i}]")
            L.info(f"    spatial_shape={v.spatial_shape}  "
                   f"coords_max={v.coords.max(dim=0).values.tolist()}")

            log_mesh(m.vertices, m.faces, f"before-fill_holes[{i}]")

            # CPU simplification via pyfqmr (QEM) before fill_holes to avoid
            # GPU OOM from CuMesh's O(F*3) edge buffers on large meshes.
            import pyfqmr, time
            _target = 4_000_000
            if m.faces.shape[0] > _target:
                _v_np = m.vertices.detach().cpu().float().numpy()
                _f_np = m.faces.detach().cpu().int().numpy()
                L.info(f"    [pyfqmr] simplify {m.faces.shape[0]} → {_target} faces ...")
                _t0 = time.perf_counter()
                _simplifier = pyfqmr.Simplify()
                _simplifier.setMesh(_v_np, _f_np)
                _simplifier.simplify_mesh(_target, aggressiveness=7, verbose=False)
                _sv, _sf, _sn = _simplifier.getMesh()
                _dt = time.perf_counter() - _t0
                L.info(f"    [pyfqmr] done in {_dt:.2f}s  →  {len(_sv)} verts  {len(_sf)} faces")
                m.vertices = torch.from_numpy(_sv).to(dtype=torch.float32, device=m.vertices.device)
                m.faces = torch.from_numpy(_sf).to(dtype=torch.int32, device=m.faces.device)

            m.fill_holes()
            log_mesh(m.vertices, m.faces, f"after-fill_holes[{i}]")

            coords_xyz = v.coords[:, 1:]
            L.info(f"    coords_xyz: {list(coords_xyz.shape)}  "
                   f"range={[coords_xyz.min().item(), coords_xyz.max().item()]}")
            L.info(f"    attrs: {list(v.feats.shape)}  "
                   f"range=[{v.feats.min().item():.4g},{v.feats.max().item():.4g}]  "
                   f"NaN={torch.isnan(v.feats).any().item()}")
            L.info(f"    voxel_size={1/resolution:.6f}  origin=[-0.5,-0.5,-0.5]")

            mv = MeshWithVoxel(
                m.vertices, m.faces,
                origin = [-0.5, -0.5, -0.5],
                voxel_size = 1 / resolution,
                coords = coords_xyz,
                attrs = v.feats,
                voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                layout=self.pbr_attr_layout
            )
            L.info(f"    MeshWithVoxel.voxel_shape={mv.voxel_shape}  "
                   f"voxel_size={mv.voxel_size}  origin={mv.origin}")

            # Visualize final MeshWithVoxel
            if visualize:
                import os
                base = (os.path.join(visualize_save_dir, f"mesh_with_voxel_{pipeline_type}_s{i}")
                        if visualize_save_dir else None)
                if base:
                    os.makedirs(visualize_save_dir, exist_ok=True)
                self.visualize_mesh_with_voxel(mv,
                    title=f"MeshWithVoxel [{i}] - {pipeline_type}",
                    save_path_prefix=base)

            out_mesh.append(mv)

        section("decode_latent complete")
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        visualize_sparse_structure: bool = False,
        visualize_save_dir: str = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            visualize_sparse_structure (bool): Whether to visualize the sparse structure.
            visualize_save_dir (str): Directory to save visualization images. If None, displays interactively.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        
        # Visualize sparse structure if requested
        if visualize_sparse_structure:
            print("\n=== Sparse Structure Visualization ===")
            self.analyze_sparse_structure(coords)
            
            if visualize_save_dir:
                import os
                os.makedirs(visualize_save_dir, exist_ok=True)
                base_path = os.path.join(visualize_save_dir, f"sparse_structure_{pipeline_type}_seed{seed}")
                
                self.visualize_sparse_structure_matplotlib(
                    coords, 
                    title=f"Sparse Structure - {pipeline_type} (seed={seed})",
                    save_path=f"{base_path}_3d.png"
                )
                
                self.visualize_sparse_structure_voxel(
                    coords,
                    resolution=ss_res,
                    title=f"Voxel Grid - {pipeline_type} (seed={seed})",
                    save_path=f"{base_path}_voxel.png"
                )
                
                self.visualize_sparse_structure_projections(
                    coords,
                    resolution=ss_res,
                    title=f"Projections - {pipeline_type} (seed={seed})",
                    save_path=f"{base_path}_projections.png"
                )
                
                self.visualize_sparse_structure_multi_view(
                    coords,
                    title=f"Multi-View - {pipeline_type} (seed={seed})",
                    save_path=f"{base_path}_multi_view.png"
                )
            else:
                # Interactive visualization (no saving)
                self.visualize_sparse_structure_matplotlib(
                    coords, 
                    title=f"Sparse Structure - {pipeline_type} (seed={seed})"
                )
                
                self.visualize_sparse_structure_voxel(
                    coords,
                    resolution=ss_res,
                    title=f"Voxel Grid - {pipeline_type} (seed={seed})"
                )
                
                self.visualize_sparse_structure_projections(
                    coords,
                    resolution=ss_res,
                    title=f"Projections - {pipeline_type} (seed={seed})"
                )
                
                self.visualize_sparse_structure_multi_view(
                    coords,
                    title=f"Multi-View - {pipeline_type} (seed={seed})"
                )
            print("=== Visualization Complete ===\n")

        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params,
                visualize=visualize_sparse_structure,
                visualize_save_dir=visualize_save_dir,
                pipeline_type=pipeline_type,
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
                visualize=visualize_sparse_structure,
                visualize_save_dir=visualize_save_dir,
                pipeline_type=pipeline_type,
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                visualize_hr_coords=visualize_sparse_structure,
                visualize_save_dir=visualize_save_dir,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
                visualize=visualize_sparse_structure,
                visualize_save_dir=visualize_save_dir,
                pipeline_type=pipeline_type,
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                visualize_hr_coords=visualize_sparse_structure,
                visualize_save_dir=visualize_save_dir,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
                visualize=visualize_sparse_structure,
                visualize_save_dir=visualize_save_dir,
                pipeline_type=pipeline_type,
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res,
                                      visualize=visualize_sparse_structure,
                                      visualize_save_dir=visualize_save_dir,
                                      pipeline_type=pipeline_type)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
