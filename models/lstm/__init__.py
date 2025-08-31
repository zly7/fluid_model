"""
LSTM model package for fluid dynamics time series prediction.
"""

from .model import FluidLSTM
from .config import LSTMConfig

__all__ = ['FluidLSTM', 'LSTMConfig']