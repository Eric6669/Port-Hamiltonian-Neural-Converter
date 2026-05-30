#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：pHNODE.py
@Author  ：He Xing
@Date    ：2026/4/2 09:34 
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from model.kan import KAN
from model.mlp import MLP

def _build_backbone(arch, input_dim, hidden_dim, output_dim, depth):
    if arch == "mlp":
        return MLP(input_dim, hidden_dim, output_dim, depth)
    elif arch == "kan":
        return KAN([input_dim] + [hidden_dim] * depth + [output_dim])
    else:
        raise ValueError(f"Unknown backbone: {arch}")

# =====================================================================
#  J_2level
#
#  u_sw = [Sa_up, Sa_dn, Sb_up, Sb_dn, Sc_up, Sc_dn]
#
#  J(S) = |  0        0      Sa_up   Sb_up   Sc_up  |
#         |  0        0     -Sa_dn  -Sb_dn  -Sc_dn  |
#         |-Sa_up   Sa_dn     0        0        0    |
#         |-Sb_up   Sb_dn     0        0        0    |
#         |-Sc_up   Sc_dn     0        0        0    |
# =====================================================================

class J_2level(nn.Module):

    def __init__(self, learnable_gain=False):
        super().__init__()
        if learnable_gain:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("alpha", torch.ones(1))

    def _build_J(self, u_sw):

        B = u_sw.size(0)
        device = u_sw.device

        Sa_up, Sa_dn = u_sw[:, 0], u_sw[:, 1]
        Sb_up, Sb_dn = u_sw[:, 2], u_sw[:, 3]
        Sc_up, Sc_dn = u_sw[:, 4], u_sw[:, 5]

        z = torch.zeros(B, device=device)

        J = torch.stack([
            z, z, Sa_up, Sb_up, Sc_up,
            z, z, -Sa_dn, -Sb_dn, -Sc_dn,
            -Sa_up, Sa_dn, z, z, z,
            -Sb_up, Sb_dn, z, z, z,
            -Sc_up, Sc_dn, z, z, z,
        ], dim=-1).view(B, 5, 5)

        return J * self.alpha

    def forward(self, u_sw, grad_H):
        """ J(S) @ grad_H, shape (batch, 5)"""
        J = self._build_J(u_sw)
        return (J @ grad_H.unsqueeze(-1)).squeeze(-1)

# =====================================================================
#  J_3level
#
#  u_sw = [
#      Sa1, Sa2, Sa3, Sa4,
#      Sb5, Sb6, Sb7, Sb8,
#      Sc9, Sc10, Sc11, Sc12
#  ]
#
#  M(S) = |  Sa1 Sa2      Sb5 Sb6      Sc9 Sc10   |
#         | -Sa3 Sa4     -Sb7 Sb8     -Sc11 Sc12  |
#
#  J(S) = |   0      M(S) |
#         | -M(S)^T   0   |
# =====================================================================

class J_3level(nn.Module):

    def __init__(self):
        super().__init__()

    def _build_J(self, u_sw):

        B = u_sw.size(0)
        device = u_sw.device

        Sa1, Sa2, Sa3, Sa4 = u_sw[:, 0], u_sw[:, 1], u_sw[:, 2], u_sw[:, 3]
        Sb5, Sb6, Sb7, Sb8 = u_sw[:, 4], u_sw[:, 5], u_sw[:, 6], u_sw[:, 7]
        Sc9, Sc10, Sc11, Sc12 = u_sw[:, 8], u_sw[:, 9], u_sw[:, 10], u_sw[:, 11]

        ma_p = Sa1 * Sa2
        mb_p = Sb5 * Sb6
        mc_p = Sc9 * Sc10

        ma_n = Sa3 * Sa4
        mb_n = Sb7 * Sb8
        mc_n = Sc11 * Sc12

        z = torch.zeros(B, device=device)

        J = torch.stack([
            z,      z,      ma_p,   mb_p,   mc_p,
            z,      z,     -ma_n,  -mb_n,  -mc_n,
           -ma_p,   ma_n,   z,      z,      z,
           -mb_p,   mb_n,   z,      z,      z,
           -mc_p,   mc_n,   z,      z,      z,
        ], dim=-1).view(B, 5, 5)

        return J

    def forward(self, u_sw, grad_H):
        J = self._build_J(u_sw)
        return (J @ grad_H.unsqueeze(-1)).squeeze(-1)

# =====================================================================
#  R_2level
#  R_2level  = |  Gpp      0      0        0       0  |
#              |  0        Gpn    0        0       0  |
#              |  0        0      Ra       0       0  |
#              |  0        0      0        Rb      0  |
#              |  0        0      0        0       Rc |
# =====================================================================

class R_2level(nn.Module):

    def __init__(self, state_dim, sw_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        self.state_dim = state_dim
        self.sw_dim = sw_dim
        self.net = _build_backbone(arch, state_dim + sw_dim, hidden_dim, state_dim, depth)

    def forward(self, z, u_sw, grad_H):
        x = torch.cat([z, u_sw], dim=-1)
        r_full = F.softplus(self.net(x)) # softplus keep >=0
        return grad_H * r_full

    def get_matrix(self, z, u_sw):
        x = torch.cat([z, u_sw], dim=-1)
        r_full = F.softplus(self.net(x))  # softplus keep >=0
        return torch.diag_embed(r_full)

# =====================================================================
#  R_3level
#
#  R_theta(z, s) = diag(Softplus(MLP([z, s])))
#  Input dimension changes from 5+6 to 5+12.
# =====================================================================

class R_3level(nn.Module):
    def __init__(self, state_dim, sw_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        self.state_dim = state_dim
        self.sw_dim = sw_dim
        self.net = _build_backbone(arch, state_dim + sw_dim, hidden_dim, state_dim, depth)

    def forward(self, z, u_sw, grad_H):
        x = torch.cat([z, u_sw], dim=-1)
        r = F.softplus(self.net(x))
        return grad_H * r

    def get_matrix(self, z, u_sw):
        x = torch.cat([z, u_sw], dim=-1)
        r = F.softplus(self.net(x))
        return torch.diag_embed(r)

# =====================================================================
#  P⁻¹
#  P⁻¹ = diag(1/Cp, 1/Cn, 1/L_a, 1/L_b, 1/L_c)
# =====================================================================

class Pinv(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        known_physics = torch.tensor([1/3.3e-3, 1/3.3e-3, 1/0.015, 1/0.015, 1/0.015])
        self.register_buffer("base_scale", known_physics)

        # Softplus(0.54) ≈ 1.0
        self.p_raw = nn.Parameter(torch.ones(state_dim) * 0.54)

    def get_matrix(self, z):
        p = F.softplus(self.p_raw) * self.base_scale
        return torch.diag(p).unsqueeze(0).expand(z.size(0), -1, -1)


# =====================================================================
#  pH-NODE
# =====================================================================

class pHNODE(nn.Module):
    """
        x = [q₁, q₂, φ_a, φ_b, φ_c]^T
        z = [v_c1, v_c2, i_a, i_b, i_c]^T
        P = diag(C₁, C₂, L_a, L_b, L_c)
        z = P⁻¹x
        dz/dt = P⁻¹ · [ (J(S) - R) · z + u ]

        Parameters:
        state_dim : int     5
        sw_dim : int        6
        R : str             "prior_diag" | "linear" | "default"
        Pinv : str          "prior_diag" | "linear" | "default"
        arch : str          "mlp" | "kan"
    """

    def __init__(
            self,
            converter_model: str = "2level",
            state_dim: int = 5,
            hidden_dim: int = 64,
            depth: int = 3,
            known_physics=None,
    ):
        super().__init__()
        if converter_model not in ["2level", "3level"]:
            raise ValueError(
                f"converter_model must be '2level' or '3level', "
                f"but got {converter_model}."
            )

        self.converter_model = converter_model
        self.state_dim = state_dim

        if converter_model == "2level":
            self.sw_dim = 6
            self.J_module = J_2level()
            self.R_module = R_2level(
                state_dim=state_dim,
                sw_dim=self.sw_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            )
            self.Pinv_module = Pinv(
                state_dim=state_dim
            )

        elif converter_model == "3level":
            self.sw_dim = 12
            self.J_module = J_3level()
            self.R_module = R_3level(
                state_dim=state_dim,
                sw_dim=self.sw_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            )
            self.Pinv_module = Pinv(
                state_dim=state_dim
            )

    def forward(self, z, u_ext, u_sw):
        """
        Parameters:
        z     : (batch, 5) [vcp, vcn, ia, ib, ic]
        u_ext : (batch, 5) [idcp, idcn, ea, eb, ec]
        u_sw  : (batch, 6/12)

        Returns:
        dz : (batch, 5)
        """

        grad_H = z
        J_gradH = self.J_module(u_sw, grad_H)
        R_gradH = self.R_module(z, u_sw, grad_H)

        # (J - R) @ grad_H
        rhs = J_gradH - R_gradH + u_ext

        # dz/dt = P⁻¹ @ rhs
        Pinv = self.Pinv_module.get_matrix(z)
        dz = (Pinv @ rhs.unsqueeze(-1)).squeeze(-1)

        return dz

    def get_matrices(self, z, u_sw):

        J = self.J_module._build_J(u_sw)
        R = self.R_module.get_matrix(z, u_sw)
        Pinv = self.Pinv_module.get_matrix(z)

        return J, R, Pinv




