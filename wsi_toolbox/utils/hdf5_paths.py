"""
HDF5 path utilities for consistent namespace and filter handling
"""

import os
from pathlib import Path


def normalize_filename(path: str) -> str:
    """
    Normalize filename for use in namespace

    Args:
        path: File path

    Returns:
        Normalized name (stem only, forbidden chars replaced)
    """
    name = Path(path).stem
    # Replace forbidden characters
    name = name.replace('+', '_')  # + is reserved for separator
    name = name.replace('/', '_')  # path separator
    return name


def build_namespace(input_paths: list[str]) -> str:
    """
    Build namespace from input file paths

    Args:
        input_paths: List of HDF5 file paths

    Returns:
        Namespace string
        - Single file: "default"
        - Multiple files: "file1+file2+..." (sorted, normalized)
    """
    if len(input_paths) == 1:
        return "default"

    # Normalize and sort filenames
    names = sorted([normalize_filename(p) for p in input_paths])
    return '+'.join(names)


def build_cluster_path(
    model_name: str,
    namespace: str = "default",
    filters: list[list[int]] | None = None,
    dataset: str = "clusters"
) -> str:
    """
    Build HDF5 path for clustering data

    Args:
        model_name: Model name (e.g., "uni", "gigapath")
        namespace: Namespace (e.g., "default", "001+002")
        filters: Nested list of cluster filters, e.g., [[1,2,3], [0,1]]
        dataset: Dataset name ("clusters" or "umap_coordinates")

    Returns:
        Full HDF5 path

    Examples:
        >>> build_cluster_path("uni", "default")
        'uni/default/clusters'

        >>> build_cluster_path("uni", "default", [[1,2,3]])
        'uni/default/filter/1+2+3/clusters'

        >>> build_cluster_path("uni", "default", [[1,2,3], [0,1]])
        'uni/default/filter/1+2+3/filter/0+1/clusters'

        >>> build_cluster_path("uni", "001+002", [[5]])
        'uni/001+002/filter/5/clusters'
    """
    path = f"{model_name}/{namespace}"

    if filters:
        for filter_ids in filters:
            filter_str = '+'.join(map(str, sorted(filter_ids)))
            path += f"/filter/{filter_str}"

    path += f"/{dataset}"
    return path


def parse_cluster_path(path: str) -> dict:
    """
    Parse cluster path into components

    Args:
        path: HDF5 path (e.g., "uni/default/filter/1+2+3/clusters")

    Returns:
        Dict with keys: model_name, namespace, filters, dataset

    Examples:
        >>> parse_cluster_path("uni/default/clusters")
        {'model_name': 'uni', 'namespace': 'default', 'filters': [], 'dataset': 'clusters'}

        >>> parse_cluster_path("uni/default/filter/1+2+3/clusters")
        {'model_name': 'uni', 'namespace': 'default', 'filters': [[1,2,3]], 'dataset': 'clusters'}
    """
    parts = path.split('/')

    result = {
        'model_name': parts[0],
        'namespace': parts[1],
        'filters': [],
        'dataset': parts[-1]
    }

    # Parse filter hierarchy
    i = 2
    while i < len(parts) - 1:
        if parts[i] == 'filter':
            filter_str = parts[i + 1]
            filter_ids = [int(x) for x in filter_str.split('+')]
            result['filters'].append(filter_ids)
            i += 2
        else:
            i += 1

    return result


def list_namespaces(h5_file, model_name: str) -> list[str]:
    """
    List all namespaces in HDF5 file for given model

    Args:
        h5_file: h5py.File object (opened)
        model_name: Model name

    Returns:
        List of namespace strings
    """
    import h5py

    if model_name not in h5_file:
        return []

    namespaces = []
    for key in h5_file[model_name].keys():
        if isinstance(h5_file[f"{model_name}/{key}"], h5py.Group):
            # Check if it contains 'clusters' dataset
            if "clusters" in h5_file[f"{model_name}/{key}"]:
                namespaces.append(key)

    return namespaces


def list_filters(h5_file, model_name: str, namespace: str) -> list[str]:
    """
    List all filter paths under a namespace

    Args:
        h5_file: h5py.File object (opened)
        model_name: Model name
        namespace: Namespace

    Returns:
        List of filter strings (e.g., ["1+2+3", "5"])
    """
    import h5py

    base_path = f"{model_name}/{namespace}/filter"
    if base_path not in h5_file:
        return []

    filters = []

    def visit_filters(name, obj):
        if isinstance(obj, h5py.Group) and "clusters" in obj:
            # Extract filter string from full path
            rel_path = name.replace(base_path + '/', '')
            # Remove '/filter/' segments to get just the IDs
            filter_str = rel_path.replace('/filter/', '/')
            filters.append(filter_str)

    h5_file[base_path].visititems(visit_filters)

    return filters
