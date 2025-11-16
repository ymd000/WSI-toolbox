"""
Command-based processors for WSI analysis pipeline.

Design pattern: __init__ for configuration, __call__ for execution
No inheritance - simple standalone command classes.
"""

import os
import gc
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from pydantic import BaseModel, Field

from .models import DEFAULT_MODEL, create_model
from .wsi_files import create_wsi_file
from .utils.helpers import is_white_patch, safe_del
from .utils.progress import tqdm_or_st
from .utils.analysis import leiden_cluster

from sklearn.preprocessing import StandardScaler
import umap
from PIL import Image, ImageFont
from matplotlib import pyplot as plt, colors as mcolors

from .utils import create_frame, get_platform_font


# === Global Configuration (Pydantic) ===
class Config(BaseModel):
    """Global configuration for commands"""
    progress: str = Field(default='tqdm', description="Progress bar backend")
    model_name: str = Field(default=DEFAULT_MODEL, description="Default model name")
    verbose: bool = Field(default=True, description="Verbose output")
    device: str = Field(default='cuda', description="Device for computation")


# Global config instance
_config = Config()


def set_default_progress(backend: str):
    """Set global default progress backend ('tqdm', 'streamlit', etc.)"""
    _config.progress = backend


def set_default_model(model_name: str):
    """Set global default model ('uni', 'gigapath', 'virchow2')"""
    _config.model_name = model_name


def set_default_device(device: str):
    """Set global default device ('cuda', 'cpu')"""
    _config.device = device


def set_verbose(verbose: bool):
    """Set global verbosity"""
    _config.verbose = verbose


def _get(key: str, value):
    """Get value or fall back to global default"""
    if value is not None:
        return value
    return getattr(_config, key)


def _progress(iterable, **kwargs):
    """Wrapper for tqdm_or_st that uses global config"""
    return tqdm_or_st(iterable, backend=_config.progress, **kwargs)


class Wsi2HDF5Command:
    """
    Convert WSI image to HDF5 format with patch extraction

    Usage:
        # Set global config once
        commands.set_default_progress('tqdm')

        # Create and run command
        cmd = Wsi2HDF5Command(patch_size=256, engine='auto')
        result = cmd(input_path='image.ndpi', output_path='output.h5')
    """

    def __init__(self,
                 patch_size: int = 256,
                 engine: str = 'auto',
                 mpp: float = 0,
                 rotate: bool = False):
        """
        Initialize WSI to HDF5 converter

        Args:
            patch_size: Size of patches to extract
            engine: WSI reader engine ('auto', 'openslide', 'tifffile', 'standard')
            mpp: Microns per pixel (for standard images)
            rotate: Whether to rotate patches 180 degrees

        Note:
            progress and verbose are controlled by global config:
            - commands.set_default_progress('tqdm')
            - commands.set_verbose(True/False)
        """
        self.patch_size = patch_size
        self.engine = engine
        self.mpp = mpp
        self.rotate = rotate

    def __call__(self, input_path: str, output_path: str) -> dict:
        """
        Execute WSI to HDF5 conversion

        Args:
            input_path: Path to input WSI file
            output_path: Path to output HDF5 file

        Returns:
            dict: Metadata including mpp, scale, patch_count
        """
        # Create WSI reader
        wsi = create_wsi_file(input_path, engine=self.engine, mpp=self.mpp)

        # Calculate scale based on mpp
        original_mpp = wsi.get_mpp()

        if 0.360 < original_mpp < 0.500:
            scale = 1
        elif original_mpp < 0.360:
            scale = 2
        else:
            raise RuntimeError(f'Invalid mpp: {original_mpp:.6f}')

        mpp = original_mpp * scale

        # Get image dimensions
        W, H = wsi.get_original_size()
        S = self.patch_size  # Scaled patch size
        T = S * scale        # Original patch size

        x_patch_count = W // T
        y_patch_count = H // T
        width = (W // T) * T
        row_count = H // T

        if _config.verbose and _config.progress == 'tqdm':
            print(f'Original mpp: {original_mpp:.6f}')
            print(f'Image mpp: {mpp:.6f}')
            print(f'Target resolutions: {W} x {H}')
            print(f'Obtained resolutions: {x_patch_count*S} x {y_patch_count*S}')
            print(f'Scale: {scale}')
            print(f'Patch size: {T}')
            print(f'Scaled patch size: {S}')
            print(f'Row count: {y_patch_count}')
            print(f'Col count: {x_patch_count}')

        coordinates = []

        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Write metadata
            f.create_dataset('metadata/original_mpp', data=original_mpp)
            f.create_dataset('metadata/original_width', data=W)
            f.create_dataset('metadata/original_height', data=H)
            f.create_dataset('metadata/image_level', data=0)
            f.create_dataset('metadata/mpp', data=mpp)
            f.create_dataset('metadata/scale', data=scale)
            f.create_dataset('metadata/patch_size', data=S)
            f.create_dataset('metadata/cols', data=x_patch_count)
            f.create_dataset('metadata/rows', data=y_patch_count)

            # Create patches dataset
            total_patches = f.create_dataset(
                'patches',
                shape=(x_patch_count * y_patch_count, S, S, 3),
                dtype=np.uint8,
                chunks=(1, S, S, 3),
                compression='gzip',
                compression_opts=9
            )

            # Extract patches row by row
            cursor = 0
            tq = _progress(range(row_count))
            for row in tq:
                # Read one row
                image = wsi.read_region((0, row * T, width, T))
                image = cv2.resize(image, (width // scale, S),
                                 interpolation=cv2.INTER_LANCZOS4)

                # Reshape into patches
                patches = image.reshape(1, S, x_patch_count, S, 3)  # (y, h, x, w, 3)
                patches = patches.transpose(0, 2, 1, 3, 4)          # (y, x, h, w, 3)
                patches = patches[0]

                # Filter white patches and collect valid ones
                batch = []
                for col, patch in enumerate(patches):
                    if is_white_patch(patch):
                        continue

                    if self.rotate:
                        patch = cv2.rotate(patch, cv2.ROTATE_180)
                        coordinates.append((
                            (x_patch_count - 1 - col) * S,
                            (y_patch_count - 1 - row) * S
                        ))
                    else:
                        coordinates.append((col * S, row * S))

                    batch.append(patch)

                # Write batch
                batch = np.array(batch)
                total_patches[cursor:cursor + len(batch), ...] = batch
                cursor += len(batch)

                tq.set_description(
                    f'Selected {len(batch)}/{len(patches)} patches '
                    f'(row {row}/{y_patch_count})'
                )
                tq.refresh()

            # Resize to actual patch count and save coordinates
            patch_count = len(coordinates)
            f.create_dataset('coordinates', data=coordinates)
            f['patches'].resize((patch_count, S, S, 3))
            f.create_dataset('metadata/patch_count', data=patch_count)

        if _config.verbose and _config.progress == 'tqdm':
            print(f'{patch_count} patches were selected.')

        return {
            'mpp': mpp,
            'original_mpp': original_mpp,
            'scale': scale,
            'patch_count': patch_count,
            'patch_size': S,
            'cols': x_patch_count,
            'rows': y_patch_count,
            'output_path': output_path
        }


class TileEmbeddingCommand:
    """
    Extract embeddings from patches using foundation models

    Usage:
        # Set global config once
        commands.set_default_model('gigapath')
        commands.set_default_device('cuda')

        # Create and run command
        cmd = TileEmbeddingCommand(batch_size=256, with_latent=False)
        result = cmd(hdf5_path='data.h5')
    """

    def __init__(self,
                 batch_size: int = 256,
                 with_latent: bool = False,
                 overwrite: bool = False,
                 model_name: str | None = None,
                 device: str | None = None):
        """
        Initialize tile embedding extractor

        Args:
            batch_size: Batch size for inference
            with_latent: Whether to extract latent features
            overwrite: Whether to overwrite existing features
            model_name: Model name (None to use global default)
            device: Device (None to use global default)

        Note:
            progress and verbose are controlled by global config
        """
        self.batch_size = batch_size
        self.with_latent = with_latent
        self.overwrite = overwrite
        self.model_name = _get('model_name', model_name)
        self.device = _get('device', device)

        # Validate model
        if self.model_name not in ['uni', 'gigapath', 'virchow2']:
            raise ValueError(f'Invalid model: {self.model_name}')

        # Dataset paths
        self.feature_name = f'{self.model_name}/features'
        self.latent_feature_name = f'{self.model_name}/latent_features'

    def __call__(self, hdf5_path: str) -> dict:
        """
        Execute embedding extraction

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            dict: Result metadata (feature_dim, patch_count, skipped, etc.)
        """
        # Load model
        model = create_model(self.model_name)
        model = model.eval().to(self.device)

        # Normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        done = False

        try:
            with h5py.File(hdf5_path, 'r+') as f:
                latent_size = model.patch_embed.proj.kernel_size[0]

                # Check if already exists
                if not self.overwrite:
                    if self.with_latent:
                        if (self.feature_name in f) and (self.latent_feature_name in f):
                            if _config.verbose:
                                print('Already extracted. Skipped.')
                            return {'skipped': True}
                        if (self.feature_name in f) or (self.latent_feature_name in f):
                            raise RuntimeError(
                                f'Either {self.feature_name} or {self.latent_feature_name} exists.'
                            )
                    else:
                        if self.feature_name in f:
                            if _config.verbose:
                                print('Already extracted. Skipped.')
                            return {'skipped': True}

                # Delete if overwrite
                if self.overwrite:
                    safe_del(f, self.feature_name)
                    safe_del(f, self.latent_feature_name)

                # Get patch count
                patch_count = f['metadata/patch_count'][()]

                # Create batch indices
                batch_idx = [
                    (i, min(i + self.batch_size, patch_count))
                    for i in range(0, patch_count, self.batch_size)
                ]

                # Create datasets
                f.create_dataset(
                    self.feature_name,
                    shape=(patch_count, model.num_features),
                    dtype=np.float32
                )
                if self.with_latent:
                    f.create_dataset(
                        self.latent_feature_name,
                        shape=(patch_count, latent_size**2, model.num_features),
                        dtype=np.float16
                    )

                # Process batches
                tq = _progress(batch_idx)
                for i0, i1 in tq:
                    # Load batch
                    x = f['patches'][i0:i1]
                    x = (torch.from_numpy(x) / 255).permute(0, 3, 1, 2)  # BHWC->BCHW
                    x = x.to(self.device)
                    x = (x - mean) / std

                    # Forward pass
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        h_tensor = model.forward_features(x)

                    # Extract features
                    h = h_tensor.cpu().detach().numpy()  # [B, T+L, H]
                    latent_index = h.shape[1] - latent_size**2
                    cls_feature = h[:, 0, ...]
                    latent_feature = h[:, latent_index:, ...]

                    # Save features
                    f[self.feature_name][i0:i1] = cls_feature
                    if self.with_latent:
                        f[self.latent_feature_name][i0:i1] = latent_feature.astype(np.float16)

                    # Cleanup
                    del x, h_tensor
                    torch.cuda.empty_cache()

                    tq.set_description(f'Processing {i0}-{i1} (total={patch_count})')
                    tq.refresh()

                if _config.verbose:
                    print(f'Embeddings dimension: {f[self.feature_name].shape}')

                done = True

                return {
                    'feature_dim': model.num_features,
                    'patch_count': patch_count,
                    'model': self.model_name,
                    'with_latent': self.with_latent
                }

        finally:
            if done and _config.verbose:
                print(f'Wrote {self.feature_name}')
            elif not done:
                # Cleanup on error
                with h5py.File(hdf5_path, 'a') as f:
                    safe_del(f, self.feature_name)
                    if self.with_latent:
                        safe_del(f, self.latent_feature_name)
                if _config.verbose:
                    print(f'ABORTED! Deleted {self.feature_name}')

            # Cleanup
            del model, mean, std
            torch.cuda.empty_cache()
            gc.collect()


class ClusteringCommand:
    """
    Perform clustering on extracted features

    Usage:
        # Set global config once
        commands.set_default_model('gigapath')

        # Create and run command
        cmd = ClusteringCommand(resolution=1.0)
        result = cmd(hdf5_paths=['data1.h5', 'data2.h5'])
    """

    def __init__(self,
                 resolution: float = 1.0,
                 cluster_name: str = '',
                 cluster_filter: list[int] | None = None,
                 use_umap: bool = False,
                 overwrite: bool = False,
                 model_name: str | None = None):
        """
        Initialize clustering command

        Args:
            resolution: Leiden clustering resolution
            cluster_name: Name for multi-file clustering
            cluster_filter: Filter to specific clusters for sub-clustering
            use_umap: Whether to use UMAP embeddings for clustering
            overwrite: Whether to overwrite existing clusters
            model_name: Model name (None to use global default)
        """
        self.resolution = resolution
        self.cluster_name = cluster_name
        self.cluster_filter = cluster_filter or []
        self.use_umap = use_umap
        self.overwrite = overwrite
        self.model_name = _get('model_name', model_name)

        # Validate model
        if self.model_name not in ['uni', 'gigapath', 'virchow2']:
            raise ValueError(f'Invalid model: {self.model_name}')

        self.sub_clustering = len(self.cluster_filter) > 0

        # Internal state (initialized in __call__)
        self.hdf5_paths = []
        self.masks = []
        self.features = None
        self.total_clusters = None
        self._umap_embeddings = None

    def __call__(self, hdf5_paths: str | list[str]) -> dict:
        """
        Execute clustering

        Args:
            hdf5_paths: Single HDF5 path or list of paths

        Returns:
            dict: Result metadata (cluster_count, etc.)
        """
        # Normalize to list
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]

        self.hdf5_paths = hdf5_paths
        multi = len(hdf5_paths) > 1

        # Validate multi-file clustering
        if multi and not self.cluster_name:
            raise RuntimeError('Multiple files provided but cluster_name was not specified.')

        # Determine cluster path
        if multi:
            clusters_path = f'{self.model_name}/clusters_{self.cluster_name}'
        else:
            clusters_path = f'{self.model_name}/clusters'

        # Load features
        self._load_features(clusters_path)

        # Check if already exists
        if not self.sub_clustering and hasattr(self, 'has_clusters') and self.has_clusters and not self.overwrite:
            if _config.verbose:
                print('Skip clustering (already exists)')
            return {'skipped': True}

        # Perform clustering
        self.total_clusters = leiden_cluster(
            self.features,
            umap_emb_func=self._get_umap_embeddings if self.use_umap else None,
            resolution=self.resolution,
            progress=_config.progress
        )

        # Write results
        target_path = clusters_path
        if self.sub_clustering:
            suffix = '_sub' + '-'.join(map(str, self.cluster_filter))
            target_path = target_path + suffix

        if _config.verbose:
            print(f'Writing to {target_path}')

        cursor = 0
        for hdf5_path, mask in zip(self.hdf5_paths, self.masks):
            count = np.sum(mask)
            clusters = self.total_clusters[cursor:cursor + count]
            cursor += count

            with h5py.File(hdf5_path, 'a') as f:
                if target_path in f:
                    del f[target_path]

                # Fill with -1 for filtered patches
                full_clusters = np.full(len(mask), -1, dtype=clusters.dtype)
                full_clusters[mask] = clusters
                f.create_dataset(target_path, data=full_clusters)

        cluster_count = len(np.unique(self.total_clusters))

        return {
            'cluster_count': cluster_count,
            'feature_count': len(self.features),
            'target_path': target_path
        }

    def _load_features(self, clusters_path: str):
        """Load features from HDF5 files"""
        featuress = []
        clusterss = []
        self.masks = []

        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                patch_count = f['metadata/patch_count'][()]

                # Check existing clusters
                if clusters_path in f:
                    clusters = f[clusters_path][:]
                else:
                    clusters = None

                # Create mask
                if self.cluster_filter:
                    if clusters is None:
                        raise RuntimeError('Sub-clustering requires pre-computed clusters')
                    mask = np.isin(clusters, self.cluster_filter)
                else:
                    mask = np.ones(patch_count, dtype=bool)

                self.masks.append(mask)

                # Load features
                feature_path = f'{self.model_name}/features'
                features = f[feature_path][mask]
                featuress.append(features)

                # Store existing clusters
                if clusters is not None:
                    clusterss.append(clusters[mask])

        # Concatenate and normalize
        features = np.concatenate(featuress)
        scaler = StandardScaler()
        self.features = scaler.fit_transform(features)

        # Store existing clusters state
        if len(clusterss) == len(self.hdf5_paths):
            self.has_clusters = True
            self.total_clusters = np.concatenate(clusterss)
        elif len(clusterss) == 0:
            self.has_clusters = False
            self.total_clusters = None
        else:
            raise RuntimeError(
                f'Cluster count mismatch: {len(clusterss)} vs {len(self.hdf5_paths)}'
            )

    def _get_umap_embeddings(self):
        """Get UMAP embeddings (lazy evaluation)"""
        if self._umap_embeddings is not None:
            return self._umap_embeddings

        reducer = umap.UMAP(n_components=2)
        self._umap_embeddings = reducer.fit_transform(self.features)
        return self._umap_embeddings


class BasePreviewCommand:
    """
    Base class for preview commands using Template Method Pattern
    
    Subclasses must implement:
    - _prepare(f, **kwargs): Prepare data (frames, scores, etc.)
    - _get_frame(index, data, f): Get frame for specific patch
    """
    
    def __init__(self, size: int = 64, font_size: int = 16, 
                 model_name: str | None = None):
        """
        Initialize preview command
        
        Args:
            size: Thumbnail patch size
            font_size: Font size for labels
            model_name: Model name (None to use global default)
        """
        self.size = size
        self.font_size = font_size
        self.model_name = _get('model_name', model_name)
    
    def __call__(self, hdf5_path: str, **kwargs) -> Image.Image:
        """
        Template method - common workflow for all preview commands
        
        Args:
            hdf5_path: Path to HDF5 file
            **kwargs: Subclass-specific arguments
            
        Returns:
            PIL.Image: Thumbnail image
        """
        S = self.size
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load metadata
            cols, rows, patch_count, patch_size = self._load_metadata(f)
            
            # Subclass-specific preparation
            data = self._prepare(f, **kwargs)
            
            # Create canvas
            canvas = Image.new('RGB', (cols * S, rows * S), (0, 0, 0))
            
            # Render all patches (common loop)
            tq = _progress(range(patch_count))
            for i in tq:
                coord = f['coordinates'][i]
                patch_array = f['patches'][i]
                
                # Get subclass-specific frame
                frame = self._get_frame(i, data, f)
                
                # Render patch
                x, y = coord // patch_size * S
                patch = Image.fromarray(patch_array).resize((S, S))
                if frame:
                    patch.paste(frame, (0, 0), frame)
                canvas.paste(patch, (x, y, x + S, y + S))
        
        return canvas
    
    def _load_metadata(self, f: h5py.File):
        """Load common metadata"""
        cols = f['metadata/cols'][()]
        rows = f['metadata/rows'][()]
        patch_count = f['metadata/patch_count'][()]
        patch_size = f['metadata/patch_size'][()]
        return cols, rows, patch_count, patch_size
    
    def _prepare(self, f: h5py.File, **kwargs):
        """
        Prepare data for rendering (implemented by subclass)
        
        Args:
            f: HDF5 file handle
            **kwargs: Subclass-specific arguments
            
        Returns:
            Any data structure needed for _get_frame()
        """
        raise NotImplementedError
    
    def _get_frame(self, index: int, data, f: h5py.File):
        """
        Get frame for specific patch (implemented by subclass)
        
        Args:
            index: Patch index
            data: Data prepared by _prepare()
            f: HDF5 file handle
            
        Returns:
            PIL.Image or None: Frame overlay
        """
        raise NotImplementedError


class PreviewClustersCommand(BasePreviewCommand):
    """
    Generate thumbnail with cluster visualization

    Usage:
        cmd = PreviewClustersCommand(size=64)
        image = cmd(hdf5_path='data.h5', cluster_name='test')
    """
    
    def _prepare(self, f: h5py.File, cluster_name: str = ''):
        """
        Prepare cluster frames
        
        Args:
            f: HDF5 file handle
            cluster_name: Cluster name suffix
            
        Returns:
            dict with 'clusters' and 'frames'
        """
        # Load clusters
        cluster_path = f'{self.model_name}/clusters'
        if cluster_name:
            cluster_path += f'_{cluster_name}'
        if cluster_path not in f:
            raise RuntimeError(f'{cluster_path} does not exist in HDF5 file')
        
        clusters = f[cluster_path][:]
        
        # Prepare frames for each cluster
        font = ImageFont.truetype(font=get_platform_font(), size=self.font_size)
        cmap = plt.get_cmap('tab20')
        frames = {}
        
        for cluster in np.unique(clusters).tolist() + [-1]:
            color = mcolors.rgb2hex(cmap(cluster)[:3]) if cluster >= 0 else '#111'
            frames[cluster] = create_frame(self.size, color, f'{cluster}', font)
        
        return {'clusters': clusters, 'frames': frames}
    
    def _get_frame(self, index: int, data, f: h5py.File):
        """Get frame for cluster at index"""
        cluster = data['clusters'][index]
        return data['frames'][cluster] if cluster >= 0 else None


class PreviewScoresCommand(BasePreviewCommand):
    """
    Generate thumbnail with score visualization

    Usage:
        cmd = PreviewScoresCommand(size=64)
        image = cmd(hdf5_path='data.h5', score_name='pca')
    """
    
    def _prepare(self, f: h5py.File, score_name: str):
        """
        Prepare score visualization data
        
        Args:
            f: HDF5 file handle
            score_name: Score dataset name
            
        Returns:
            dict with 'scores', 'cmap', and 'font'
        """
        # Load scores
        score_path = f'{self.model_name}/scores_{score_name}'
        scores = f[score_path][()]
        
        # Prepare font and colormap
        font = ImageFont.truetype(font=get_platform_font(), size=self.font_size)
        cmap = plt.get_cmap('viridis')
        
        return {'scores': scores, 'cmap': cmap, 'font': font}
    
    def _get_frame(self, index: int, data, f: h5py.File):
        """Get frame for score at index"""
        score = data['scores'][index]
        
        if np.isnan(score):
            return None
        
        color = mcolors.rgb2hex(data['cmap'](score)[:3])
        return create_frame(self.size, color, f'{score:.3f}', data['font'])



class DziExportCommand:
    """
    Export HDF5 patches to DZI (Deep Zoom Image) format
    
    Usage:
        cmd = DziExportCommand(jpeg_quality=90, fill_empty=False)
        cmd(hdf5_path='data.h5', output_dir='output', name='slide')
    """
    
    def __init__(self,
                 jpeg_quality: int = 90,
                 fill_empty: bool = False):
        """
        Initialize DZI export command
        
        Args:
            jpeg_quality: JPEG compression quality (0-100)
            fill_empty: Fill missing tiles with black tiles
        """
        self.jpeg_quality = jpeg_quality
        self.fill_empty = fill_empty
    
    def __call__(self, hdf5_path: str, output_dir: str, name: str) -> dict:
        """
        Export to DZI format with full pyramid
        
        Args:
            hdf5_path: Path to HDF5 file
            output_dir: Output directory
            name: Base name for DZI files
            
        Returns:
            dict: Export metadata
        """
        import math
        import shutil
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read HDF5
        with h5py.File(hdf5_path, 'r') as f:
            patches = f['patches'][:]
            coords = f['coordinates'][:]
            original_width = f['metadata/original_width'][()]
            original_height = f['metadata/original_height'][()]
            tile_size = f['metadata/patch_size'][()]
        
        # Validate tile_size (256 or 512 only)
        if tile_size not in [256, 512]:
            raise ValueError(f'Unsupported patch_size: {tile_size}. Only 256 or 512 are supported.')
        
        # Calculate grid and levels
        cols = (original_width + tile_size - 1) // tile_size
        rows = (original_height + tile_size - 1) // tile_size
        max_dimension = max(original_width, original_height)
        max_level = math.ceil(math.log2(max_dimension))
        
        if _config.verbose:
            print(f'Original size: {original_width}x{original_height}')
            print(f'Tile size: {tile_size}')
            print(f'Grid: {cols}x{rows}')
            print(f'Total patches in HDF5: {len(patches)}')
            print(f'Max zoom level: {max_level} (Level 0 = 1x1, Level {max_level} = original)')
        
        coord_to_idx = {(int(x // tile_size), int(y // tile_size)): idx
                        for idx, (x, y) in enumerate(coords)}
        
        # Setup directories
        dzi_path = output_dir / f'{name}.dzi'
        files_dir = output_dir / f'{name}_files'
        files_dir.mkdir(exist_ok=True)
        
        # Create empty tile template for current tile_size
        empty_tile_path = None
        if self.fill_empty:
            empty_tile_path = files_dir / '_empty.jpeg'
            black_img = Image.fromarray(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
            black_img.save(empty_tile_path, 'JPEG', quality=self.jpeg_quality)
        
        # Export max level (original patches from HDF5)
        level_dir = files_dir / str(max_level)
        level_dir.mkdir(exist_ok=True)
        
        tq = _progress(range(rows))
        for row in tq:
            tq.set_description(f'Exporting level {max_level}: row {row+1}/{rows}')
            for col in range(cols):
                tile_path = level_dir / f'{col}_{row}.jpeg'
                if (col, row) in coord_to_idx:
                    idx = coord_to_idx[(col, row)]
                    patch = patches[idx]
                    img = Image.fromarray(patch)
                    img.save(tile_path, 'JPEG', quality=self.jpeg_quality)
                elif self.fill_empty:
                    shutil.copyfile(empty_tile_path, tile_path)
        
        # Generate lower levels by downsampling
        for level in range(max_level - 1, -1, -1):
            if _config.verbose:
                print(f'Generating level {level}...')
            self._generate_zoom_level_down(
                files_dir, level, max_level, original_width, original_height,
                tile_size, empty_tile_path
            )
        
        # Generate DZI XML
        self._generate_dzi_xml(dzi_path, original_width, original_height, tile_size)
        
        if _config.verbose:
            print(f'DZI export complete: {dzi_path}')
        
        return {
            'dzi_path': str(dzi_path),
            'max_level': max_level,
            'tile_size': tile_size,
            'grid': f'{cols}x{rows}'
        }
    
    def _generate_zoom_level_down(self,
                                   files_dir: Path,
                                   curr_level: int,
                                   max_level: int,
                                   original_width: int,
                                   original_height: int,
                                   tile_size: int,
                                   empty_tile_path: Path | None):
        """Generate a zoom level by downsampling from the higher level"""
        import math
        import shutil
        from pathlib import Path
        
        src_level = curr_level + 1
        src_dir = files_dir / str(src_level)
        curr_dir = files_dir / str(curr_level)
        curr_dir.mkdir(exist_ok=True)
        
        # Calculate dimensions at each level
        curr_scale = 2 ** (max_level - curr_level)
        curr_width = math.ceil(original_width / curr_scale)
        curr_height = math.ceil(original_height / curr_scale)
        curr_cols = math.ceil(curr_width / tile_size)
        curr_rows = math.ceil(curr_height / tile_size)
        
        src_scale = 2 ** (max_level - src_level)
        src_width = math.ceil(original_width / src_scale)
        src_height = math.ceil(original_height / src_scale)
        src_cols = math.ceil(src_width / tile_size)
        src_rows = math.ceil(src_height / tile_size)
        
        tq = _progress(range(curr_rows))
        for row in tq:
            for col in range(curr_cols):
                # Combine 4 tiles from source level
                combined = np.zeros((tile_size * 2, tile_size * 2, 3), dtype=np.uint8)
                has_any_tile = False
                
                for dy in range(2):
                    for dx in range(2):
                        src_col = col * 2 + dx
                        src_row = row * 2 + dy
                        
                        if src_col < src_cols and src_row < src_rows:
                            src_path = src_dir / f'{src_col}_{src_row}.jpeg'
                            if src_path.exists():
                                src_img = Image.open(src_path)
                                src_array = np.array(src_img)
                                h, w = src_array.shape[:2]
                                combined[dy*tile_size:dy*tile_size+h,
                                        dx*tile_size:dx*tile_size+w] = src_array
                                has_any_tile = True
                
                tile_path = curr_dir / f'{col}_{row}.jpeg'
                if has_any_tile:
                    combined_img = Image.fromarray(combined)
                    downsampled = combined_img.resize((tile_size, tile_size), Image.LANCZOS)
                    downsampled.save(tile_path, 'JPEG', quality=self.jpeg_quality)
                elif self.fill_empty and empty_tile_path:
                    shutil.copyfile(empty_tile_path, tile_path)
            
            tq.set_description(f'Generating level {curr_level}: row {row+1}/{curr_rows}')
    
    def _generate_dzi_xml(self, dzi_path: Path, width: int, height: int, tile_size: int):
        """Generate DZI XML file"""
        dzi_content = f'''<?xml version="1.0" encoding="utf-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpeg"
       Overlap="0"
       TileSize="{tile_size}">
    <Size Width="{width}" Height="{height}"/>
</Image>
'''
        with open(dzi_path, 'w', encoding='utf-8') as f:
            f.write(dzi_content)
