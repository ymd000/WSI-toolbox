"""
Command-based processors for WSI analysis pipeline.

Design pattern: __init__ for configuration, __call__ for execution
"""

# Import configuration from common module
from ..common import (
    Config,
    get_config,
    set_default_progress,
    set_default_model,
    set_default_model_preset,
    set_default_device,
    set_verbose,
    _get,
    _progress,
)


# Import and export all commands
from .wsi import Wsi2HDF5Command
from .patch_embedding import PatchEmbeddingCommand
from .clustering import ClusteringCommand
from .preview import (
    BasePreviewCommand,
    PreviewClustersCommand,
    PreviewScoresCommand,
    PreviewLatentPCACommand,
    PreviewLatentClusterCommand,
)
from .dzi_export import DziExportCommand

__all__ = [
    # Config
    'Config',
    'get_config',
    # Config setters
    'set_default_progress',
    'set_default_model',
    'set_default_model_preset',
    'set_default_device',
    'set_verbose',
    # Helper functions
    '_get',
    '_progress',
    # Commands
    'Wsi2HDF5Command',
    'PatchEmbeddingCommand',
    'ClusteringCommand',
    'BasePreviewCommand',
    'PreviewClustersCommand',
    'PreviewScoresCommand',
    'PreviewLatentPCACommand',
    'PreviewLatentClusterCommand',
    'DziExportCommand',
]
