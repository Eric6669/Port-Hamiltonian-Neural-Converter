#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：LSTM.py
@Author  ：He Xing
@Date    ：2026/5/23 18:42 
"""

import torch
from torch import nn


class LSTM(nn.Module):
    """
    Pure black-box LSTM baseline.

    Different from NODE/pH-NODE, this model does not predict dz/dt.
    It maps the historical sequence [z, u_ext, u_sw] to the next-step
    state sequence.

    z_{1:L} = LSTM_theta(z_{0:L-1}, u_ext_{0:L-1}, u_sw_{0:L-1})

    This is used to represent a conventional data-driven sequence model.
    """

    def __init__(
        self,
        converter_model: str = "2level",
        state_dim: int = 5,
        u_ext_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
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

        self.input_dim = self.state_dim + self.u_ext_dim + self.sw_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out = nn.Linear(hidden_dim, state_dim)

    def forward_sequence(self, z_seq, u_ext_seq, u_sw_seq, hidden=None):
        """
        Sequence-to-sequence one-step-ahead prediction.

        Inputs:
            z_seq     : (B, T, 5), states from k=0 to T-1
            u_ext_seq : (B, T, 5), external inputs from k=0 to T-1
            u_sw_seq  : (B, T, sw_dim), switching signals from k=0 to T-1

        Output:
            z_pred    : (B, T, 5), predicted states from k=1 to T
        """
        x = torch.cat([z_seq, u_ext_seq, u_sw_seq], dim=-1)

        h, hidden = self.lstm(x, hidden)
        z_pred = self.out(h)

        return z_pred, hidden

    def forward(self, z, u_ext, u_sw, hidden=None):
        """
        Supports both sequence input and single-step input.

        Case 1:
            z     : (B, T, 5)
            u_ext : (B, T, 5)
            u_sw  : (B, T, sw_dim)
            return:
                z_pred: (B, T, 5), one-step-ahead predictions

        Case 2:
            z     : (B, 5)
            u_ext : (B, 5)
            u_sw  : (B, sw_dim)
            return:
                z_pred: (B, 5)
        """
        if z.dim() == 3:
            z_pred, hidden = self.forward_sequence(z, u_ext, u_sw, hidden)
            return z_pred, hidden

        elif z.dim() == 2:
            x = torch.cat([z, u_ext, u_sw], dim=-1).unsqueeze(1)
            h, hidden = self.lstm(x, hidden)
            z_pred = self.out(h[:, -1, :])
            return z_pred, hidden

        else:
            raise ValueError(
                f"z must have shape (B, T, 5) or (B, 5), "
                f"but got {tuple(z.shape)}."
            )