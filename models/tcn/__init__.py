"""
TCN (Temporal Convolutional Network) model package.
"""

from .model import FluidTCN
from .config import TCNConfig

__all__ = ['FluidTCN', 'TCNConfig']