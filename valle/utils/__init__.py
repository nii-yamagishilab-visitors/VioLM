# Derived from MIDI-VALLE and modified for VioLM by the VioLM authors, 2026.

import torch
import torch.nn as nn

from .runtime import AttributeDict, MetricsTracker, make_pad_mask, str2bool
from .symbol_table import SymbolTable


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)
