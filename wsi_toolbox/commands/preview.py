"""
Preview generation commands using Template Method Pattern
"""

import h5py
import numpy as np
from PIL import Image, ImageFont
from matplotlib import pyplot as plt, colors as mcolors

from ..utils import create_frame, get_platform_font
from . import _get, _progress


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
