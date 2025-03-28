# src/pgs_net/__init__.py
"""PGS-Net Module Initialization"""

# Import core components for easier access
from .config import DEFAULT_PGS_FFN_CONFIG, load_config, merge_configs
from .pgs_ffn import PGS_FFN
from .transformer_block import TransformerBlock

# Optionally import submodules or specific classes if needed at top level
from . import modules
from . import utils

# Setup root logger for the library
from .utils.logging_utils import get_logger

logger = get_logger("PGS_Net")

# Define package version (optional)
__version__ = "0.1.0-cp3.3"

__all__ = [
    "DEFAULT_PGS_FFN_CONFIG",
    "load_config",
    "merge_configs",
    "PGS_FFN",
    "TransformerBlock",
    "modules",
    "utils",
    "logger",
]
