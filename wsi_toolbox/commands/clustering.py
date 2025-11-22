"""
Clustering command for WSI features
"""

import h5py
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from ..utils.analysis import leiden_cluster
from ..utils.hdf5_paths import build_cluster_path, build_namespace
from . import _get, get_config


class ClusteringResult(BaseModel):
    """Result of clustering operation"""

    cluster_count: int
    feature_count: int
    target_path: str
    skipped: bool = False


class ClusteringCommand:
    """
    Perform clustering on extracted features with hierarchical namespace/filter support

    New HDF5 structure:
        <model>/
        ├── default/                    # Single file default namespace
        │   ├── clusters
        │   ├── umap_coordinates
        │   └── filter/                 # Filtered clustering
        │       └── 1+2+3/
        │           ├── clusters
        │           └── umap_coordinates
        └── 001+002/                    # Multi-file namespace
            ├── clusters
            └── filter/
                └── 0+1/
                    └── clusters

    Usage:
        # Basic clustering (single file)
        cmd = ClusteringCommand(resolution=1.0)
        result = cmd('data.h5')  # → uni/default/clusters

        # Multi-file clustering
        cmd = ClusteringCommand()
        result = cmd(['001.h5', '002.h5'])  # → uni/001+002/clusters

        # Filtered clustering (from existing clusters)
        cmd = ClusteringCommand(parent_filters=[[1,2,3]])
        result = cmd('data.h5')  # → uni/default/filter/1+2+3/clusters

        # Nested filtering
        cmd = ClusteringCommand(parent_filters=[[1,2,3], [0,1]])
        result = cmd('data.h5')  # → uni/default/filter/1+2+3/filter/0+1/clusters
    """

    def __init__(
        self,
        resolution: float = 1.0,
        namespace: str | None = None,
        parent_filters: list[list[int]] | None = None,
        use_umap: bool = False,
        overwrite: bool = False,
        model_name: str | None = None,
    ):
        """
        Initialize clustering command

        Args:
            resolution: Leiden clustering resolution
            namespace: Explicit namespace (None = auto-generate from input paths)
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [0,1]]
                          - [[1,2,3]] → filter on clusters 1,2,3
                          - [[1,2,3], [0,1]] → filter 1,2,3, then filter 0,1 from result
            use_umap: Whether to use UMAP embeddings for clustering
            overwrite: Whether to overwrite existing clusters
            model_name: Model name (None to use global default)
        """
        self.resolution = resolution
        self.namespace = namespace  # None = auto-generate
        self.parent_filters = parent_filters or []
        self.use_umap = use_umap
        self.overwrite = overwrite
        self.model_name = _get("model_name", model_name)

        # Validate model
        if self.model_name not in ["uni", "gigapath", "virchow2"]:
            raise ValueError(f"Invalid model: {self.model_name}")

        # Internal state (initialized in __call__)
        self.hdf5_paths = []
        self.masks = []
        self.features = None
        self.total_clusters = None
        self.umap_embeddings = None

    def __call__(self, hdf5_paths: str | list[str]) -> ClusteringResult:
        """
        Execute clustering

        Args:
            hdf5_paths: Single HDF5 path or list of paths

        Returns:
            ClusteringResult: Result metadata (cluster_count, etc.)
        """
        # Normalize to list
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]

        self.hdf5_paths = hdf5_paths

        # Determine namespace (auto-generate if not specified)
        if self.namespace is None:
            self.namespace = build_namespace(hdf5_paths)

        # Build parent cluster path (to load existing clusters for filtering)
        # If no filters, this is also the target path
        parent_path = build_cluster_path(
            self.model_name,
            self.namespace,
            filters=self.parent_filters[:-1] if len(self.parent_filters) > 0 else None,
            dataset="clusters"
        )

        # Load features (and apply parent filters if specified)
        self._load_features(parent_path)

        # Check if already exists (only for non-filtered clustering)
        is_filtering = len(self.parent_filters) > 0
        if not is_filtering and hasattr(self, "has_clusters") and self.has_clusters and not self.overwrite:
            if get_config().verbose:
                print("Skip clustering (already exists)")
            return ClusteringResult(
                cluster_count=len(np.unique(self.total_clusters)),
                feature_count=len(self.features),
                target_path=parent_path,
                skipped=True,
            )

        # Perform clustering
        self.total_clusters = leiden_cluster(
            self.features,
            umap_emb_func=self.get_umap_embeddings if self.use_umap else None,
            resolution=self.resolution,
            progress=get_config().progress,
        )

        # Build target paths for new clustering results
        target_cluster_path = build_cluster_path(
            self.model_name,
            self.namespace,
            filters=self.parent_filters,
            dataset="clusters"
        )
        target_umap_path = build_cluster_path(
            self.model_name,
            self.namespace,
            filters=self.parent_filters,
            dataset="umap_coordinates"
        )

        if get_config().verbose:
            print(f"Writing to {target_cluster_path}")
            if self.umap_embeddings is not None:
                print(f"Writing UMAP coordinates to {target_umap_path}")

        # Write results to each file
        self._write_results(target_cluster_path, target_umap_path)

        cluster_count = len(np.unique(self.total_clusters))

        return ClusteringResult(
            cluster_count=cluster_count,
            feature_count=len(self.features),
            target_path=target_cluster_path
        )

    def _load_features(self, parent_clusters_path: str):
        """
        Load features from HDF5 files

        Args:
            parent_clusters_path: Path to parent clusters (for filtering)
        """
        featuress = []
        clusterss = []
        self.masks = []

        # Get the last filter (most recent) to apply
        current_filter = self.parent_filters[-1] if len(self.parent_filters) > 0 else None

        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                patch_count = f["metadata/patch_count"][()]

                # Check existing clusters (for filtering)
                if current_filter is not None:
                    if parent_clusters_path not in f:
                        raise RuntimeError(f"Filtering requires pre-computed clusters at {parent_clusters_path}")
                    clusters = f[parent_clusters_path][:]
                    mask = np.isin(clusters, current_filter)
                else:
                    clusters = f[parent_clusters_path][:] if parent_clusters_path in f else None
                    mask = np.ones(patch_count, dtype=bool)

                self.masks.append(mask)

                # Load features
                feature_path = f"{self.model_name}/features"
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
            raise RuntimeError(f"Cluster count mismatch: {len(clusterss)} vs {len(self.hdf5_paths)}")

    def _write_results(self, clusters_path: str, umap_path: str):
        """
        Write clustering results to HDF5 files

        Args:
            clusters_path: Target path for clusters
            umap_path: Target path for UMAP coordinates
        """
        cursor = 0
        for hdf5_path, mask in zip(self.hdf5_paths, self.masks):
            count = np.sum(mask)
            clusters = self.total_clusters[cursor : cursor + count]

            with h5py.File(hdf5_path, "a") as f:
                # Ensure parent groups exist
                self._ensure_groups(f, clusters_path)

                # Write clusters
                if clusters_path in f:
                    del f[clusters_path]

                # Fill with -1 for filtered patches
                full_clusters = np.full(len(mask), -1, dtype=clusters.dtype)
                full_clusters[mask] = clusters
                f.create_dataset(clusters_path, data=full_clusters)

                # Save UMAP coordinates if computed
                if self.umap_embeddings is not None:
                    self._ensure_groups(f, umap_path)

                    if umap_path in f:
                        del f[umap_path]

                    # Get UMAP coordinates for this file's patches
                    umap_coords = self.umap_embeddings[cursor : cursor + count]

                    # Fill with NaN for filtered patches
                    full_umap = np.full((len(mask), 2), np.nan, dtype=umap_coords.dtype)
                    full_umap[mask] = umap_coords
                    f.create_dataset(umap_path, data=full_umap)

            cursor += count

    def _ensure_groups(self, h5file: h5py.File, path: str):
        """
        Ensure all parent groups exist for given path

        Args:
            h5file: HDF5 file handle
            path: Full path to dataset (e.g., "uni/default/filter/1+2+3/clusters")
        """
        parts = path.split('/')
        # All but the last part (which is the dataset name)
        group_parts = parts[:-1]

        current = ""
        for part in group_parts:
            current = f"{current}/{part}" if current else part
            if current not in h5file:
                h5file.create_group(current)

    def get_umap_embeddings(self):
        import umap

        """Get UMAP embeddings (lazy evaluation)"""
        if self.umap_embeddings is not None:
            return self.umap_embeddings

        reducer = umap.UMAP(n_components=2)
        self.umap_embeddings = reducer.fit_transform(self.features)
        return self.umap_embeddings
