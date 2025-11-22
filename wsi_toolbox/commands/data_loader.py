"""
Common data loading utilities for clustering and UMAP commands
"""

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from ..utils.hdf5_paths import build_cluster_path, build_namespace


class DataLoader:
    """
    Load data from HDF5 with namespace + filters

    Handles the common pattern of:
    1. Loading existing clusters at each filter level
    2. Building cumulative mask
    3. Loading features/UMAP coordinates with the mask
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        model_name: str,
        namespace: str,
        parent_filters: list[list[int]],
    ):
        """
        Args:
            hdf5_paths: List of HDF5 file paths
            model_name: Model name (e.g., "uni")
            namespace: Namespace (e.g., "default", "001+002")
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [4,5]]
        """
        self.hdf5_paths = hdf5_paths
        self.model_name = model_name
        self.namespace = namespace
        self.parent_filters = parent_filters

    def load_features(self, source: str = "features") -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Load features or UMAP coordinates with filtering

        Args:
            source: "features" or "umap"

        Returns:
            (data, masks):
                - data: Concatenated and normalized features/UMAP coordinates
                - masks: List of boolean masks for each file
        """
        data_list = []
        masks = []

        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                patch_count = f["metadata/patch_count"][()]

                # Build cumulative mask from filters
                mask = self._build_mask(f, patch_count)
                masks.append(mask)

                # Load data based on source
                if source == "umap":
                    # Load UMAP coordinates
                    umap_path = build_cluster_path(
                        self.model_name,
                        self.namespace,
                        filters=self.parent_filters if self.parent_filters else None,
                        dataset="umap_coordinates"
                    )
                    if umap_path not in f:
                        raise RuntimeError(
                            f"UMAP coordinates not found at {umap_path}. "
                            f"Run 'wsi-toolbox umap' first."
                        )
                    data = f[umap_path][mask]
                    if np.any(np.isnan(data)):
                        raise RuntimeError(f"NaN values in UMAP coordinates at {umap_path}")
                else:
                    # Load features
                    feature_path = f"{self.model_name}/features"
                    if feature_path not in f:
                        raise RuntimeError(f"Features not found at {feature_path} in {hdf5_path}")
                    data = f[feature_path][mask]

                data_list.append(data)

        # Concatenate and normalize
        data = np.concatenate(data_list)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        return data, masks

    def _build_mask(self, f: h5py.File, patch_count: int) -> np.ndarray:
        """
        Build cumulative mask from hierarchical filters

        Strategy: Only read the deepest cluster level
        - If filters = [[1,2,3], [4,5]], only read clusters at filter/1+2+3/filter/4+5
        - Those clusters are already filtered by [1,2,3], so we only need to filter by [4,5]
        """
        if not self.parent_filters:
            # No filtering
            return np.ones(patch_count, dtype=bool)

        # Get the deepest cluster path (parent of where we'll write new clusters)
        # If filters = [[1,2,3], [4,5]], we need clusters at filter/1+2+3/
        parent_cluster_path = build_cluster_path(
            self.model_name,
            self.namespace,
            filters=self.parent_filters[:-1] if len(self.parent_filters) > 1 else None,
            dataset="clusters"
        )

        if parent_cluster_path not in f:
            raise RuntimeError(
                f"Parent clusters not found at {parent_cluster_path}. "
                f"Run clustering at parent level first."
            )

        clusters = f[parent_cluster_path][:]

        # Filter by the last filter only (because previous filters are already applied)
        last_filter = self.parent_filters[-1]
        mask = np.isin(clusters, last_filter)

        return mask

    def get_parent_cluster_info(self, hdf5_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get parent clusters and mask for a single file

        Returns:
            (clusters, mask): Parent cluster values and boolean mask
        """
        with h5py.File(hdf5_path, "r") as f:
            patch_count = f["metadata/patch_count"][()]
            mask = self._build_mask(f, patch_count)

            if self.parent_filters:
                parent_cluster_path = build_cluster_path(
                    self.model_name,
                    self.namespace,
                    filters=self.parent_filters[:-1] if len(self.parent_filters) > 1 else None,
                    dataset="clusters"
                )
                clusters = f[parent_cluster_path][:]
            else:
                clusters = None

            return clusters, mask
