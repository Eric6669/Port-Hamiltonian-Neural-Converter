#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""
import torch
from torch import nn
from model.mlp import MLP

class ConverterNODE(nn.Module):

    def __init__(
            self,
            state_dim: int = 5,
            u_ext_dim: int = 5,
            sw_dim: int = 6,
            hidden_dim: int = 128,
            depth: int = 4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.u_ext_dim = u_ext_dim
        self.sw_dim = sw_dim

        input_dim = state_dim + u_ext_dim + sw_dim
        output_dim = state_dim

        self.net = MLP(input_dim, hidden_dim, output_dim, depth)

    def forward(self, z, u_ext, u_sw):
        """
        Estimate the state derivative dz/dt.

        Parameters:
        z     : (batch, 5) [vcp, vcn, ia, ib, ic]
        u_ext : (batch, 5) [idcp, idcn, ea, eb, ec]
        u_sw  : (batch, 6)

        Returns:
        dz : (batch, 5) dz/dt
        """

        x = torch.cat([z, u_ext, u_sw], dim=-1)
        dz = self.net(x)

        return dz
