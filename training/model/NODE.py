#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：NODE.py
@Author  ：He Xing
@Date    ：2026/4/9 20:02 
"""

import torch
from torch import nn
from model.mlp import PlainMLP

class NODE(nn.Module):

    def __init__(
        self,
        converter_model: str = "2level",
        state_dim: int = 5,
        u_ext_dim: int = 5,
        hidden_dim: int = 128,
        depth: int = 4,
        sw_dim: int = None,
    ):
        super().__init__()

        if converter_model not in ["2level", "3level"]:
            raise ValueError(
                f"converter_model must be '2level' or '3level', "
                f"but got {converter_model}."
            )

        self.converter_model = converter_model
        self.state_dim = state_dim
        self.u_ext_dim = u_ext_dim

        if sw_dim is None:
            self.sw_dim = 6 if converter_model == "2level" else 12
        else:
            self.sw_dim = sw_dim

        input_dim = self.state_dim + self.u_ext_dim + self.sw_dim
        output_dim = self.state_dim

        self.net = PlainMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
        )

    def forward(self, z, u_ext, u_sw):
        """
        Parameters
        ----------
        z : Tensor
            Shape (B, 5), [vcp, vcn, ia, ib, ic].
        u_ext : Tensor
            Shape (B, 5), [idcp, idcn, va, vb, vc].
        u_sw : Tensor
            Shape (B, 6) for 2-level or (B, 12) for 3-level.

        Returns
        -------
        dz : Tensor
            Shape (B, 5), predicted dz/dt.
        """
        x = torch.cat([z, u_ext, u_sw], dim=-1)
        dz = self.net(x)
        return dz

    def forward_sequence(self, z_seq, u_ext_seq, u_sw_seq):
        """
        Batch sequence inference without integration.

        Parameters
        ----------
        z_seq : Tensor
            Shape (B, T, 5).
        u_ext_seq : Tensor
            Shape (B, T, 5).
        u_sw_seq : Tensor
            Shape (B, T, sw_dim).

        Returns
        -------
        dz_seq : Tensor
            Shape (B, T, 5).
        """
        B, T, _ = z_seq.shape

        x = torch.cat([z_seq, u_ext_seq, u_sw_seq], dim=-1)
        x = x.reshape(B * T, -1)

        dz = self.net(x)
        dz = dz.reshape(B, T, self.state_dim)

        return dz