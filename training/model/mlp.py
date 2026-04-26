#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""

from torch import nn
from functools import partial

class MLP(nn.Sequential):
    """MLP block with Linear, LayerNorm, and SiLU layers."""

    def __init__(self, input_dim, hidden_dim, output_dim, depth,
                 activation = partial(nn.SiLU, inplace=True)):
        layers = [nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
        )]
        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation(),
            ))
        layers.append(nn.Linear(hidden_dim, output_dim))
        super().__init__(*layers)
