#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：mlp.py
@Author  ：He Xing
@Date    ：2026/4/2 14:27 
"""

from torch import nn
from functools import partial

class MLP(nn.Sequential):
    """带 LayerNorm + SiLU 的多层感知机"""

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

class PlainMLP(nn.Sequential):
    """
    Standard MLP without LayerNorm.
    Structure:
        Linear -> Activation -> Linear -> Activation -> ... -> Linear
    """

    def __init__(self, input_dim, hidden_dim, output_dim, depth,
                 activation=partial(nn.ReLU, inplace=True)):
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, output_dim))

        super().__init__(*layers)