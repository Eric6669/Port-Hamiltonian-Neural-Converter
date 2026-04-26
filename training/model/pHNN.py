#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from model.mlp import MLP

def _build_backbone(arch, input_dim, hidden_dim, output_dim, depth):
    if arch == "mlp":
        return MLP(input_dim, hidden_dim, output_dim, depth)
    elif arch == "kan":
        raise ValueError("KAN backbone is not included in this release; use arch=mlp.")
    else:
        raise ValueError(f"Unknown backbone: {arch}")

# =====================================================================
#  J u_sw
#  J(S) = |  0        0      Sa_up   Sb_up   Sc_up  |
#         |  0        0     -Sa_dn  -Sb_dn  -Sc_dn  |
#         |-Sa_up  Sa_dn     0        0        0    |
#         |-Sb_up  Sb_dn     0        0        0    |
#         |-Sc_up  Sc_dn     0        0        0    |
# =====================================================================

class JPrior(nn.Module):

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
#  R
#  R    = |  Gpp      0      0        0       0  |
#         |  0        Gpn    0        0       0  |
#         |  0        0      Ra       0       0  |
#         |  0        0      0        Rb      0  |
#         |  0        0      0        0       Rc |
# =====================================================================

class RPriorDiag(nn.Module):
    """Learn a non-negative diagonal dissipation matrix."""

    def __init__(self):
        super().__init__()
        self.r_raw = nn.Parameter(torch.zeros(5))

    def forward(self, z, u_sw, grad_H):
        r_full = F.softplus(self.r_raw)  # (5,) softplus keep non-negative
        return grad_H * r_full.unsqueeze(0)

    def get_matrix(self, z, u_sw):
        r_full = F.softplus(self.r_raw)
        return torch.diag(r_full).unsqueeze(0).expand(z.size(0), -1, -1)

class RLinear(nn.Module):
    """
    Learnable positive semi-definite R via  L L^T / sqrt(n)
    """

    def __init__(self, state_dim):
        super().__init__()
        self.L = nn.Parameter(torch.randn(state_dim, state_dim))
        nn.init.kaiming_normal_(self.L)

    def forward(self, z, u_sw, grad_H):
        return F.linear(F.linear(grad_H, self.L), self.L.T) / math.sqrt(self.L.size(0))

    def get_matrix(self, z, u_sw):
        R = self.L @ self.L.T / math.sqrt(self.L.size(0))
        return R.unsqueeze(0).expand(z.size(0), -1, -1)

class RDefault(nn.Module):
    """
    MLP/KAN  R = L L^T / sqrt(n)
    """

    def __init__(self, state_dim, sw_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        self.state_dim = state_dim
        self.sw_dim = sw_dim
        self.net = _build_backbone(arch, state_dim + sw_dim, hidden_dim, state_dim ** 2, depth)

    def forward(self, z, u_sw, grad_H):
        x = torch.cat([z, u_sw], dim=-1)
        L = self.net(x).view(-1, self.state_dim, self.state_dim)
        return (L @ L.permute(0, 2, 1) @ grad_H.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.state_dim)

    def get_matrix(self, z, u_sw):
        x = torch.cat([z, u_sw], dim=-1)
        L = self.net(x).view(-1, self.state_dim, self.state_dim)
        return L @ L.permute(0, 2, 1) / math.sqrt(self.state_dim)

class RNonlinearDiag(nn.Module):
    """
    MLP/KAN for R = diag(Softplus(net(z, u_sw)))
    """

    def __init__(self, state_dim, sw_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        self.state_dim = state_dim
        self.sw_dim = sw_dim
        self.net = _build_backbone(arch, state_dim + sw_dim, hidden_dim, state_dim-2, depth)

    def forward(self, z, u_sw, grad_H):
        B = z.size(0)
        x = torch.cat([z, u_sw], dim=-1)
        r_ac = F.softplus(self.net(x)) # softplus keep >=0
        zeros_dc = torch.zeros(B, 2, dtype=r_ac.dtype, device=r_ac.device)
        r_full = torch.cat([zeros_dc, r_ac], dim=-1)
        return grad_H * r_full

    def get_matrix(self, z, u_sw):
        B = z.size(0)
        x = torch.cat([z, u_sw], dim=-1)
        r_ac = F.softplus(self.net(x))  # softplus keep >=0
        zeros_dc = torch.zeros(B, 2, dtype=r_ac.dtype, device=r_ac.device)
        r_full = torch.cat([zeros_dc, r_ac], dim=-1)
        return torch.diag_embed(r_full)

class RNonlinearDiag2(nn.Module):
    """
    MLP/KAN for R = diag(Softplus(net(z, u_sw)))
    """

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
#  P_inv
#  P_inv = diag(1/Cp, 1/Cn, 1/L_a, 1/L_b, 1/L_c)
# =====================================================================

class PinvPriorDiag(nn.Module):
    """
    directly learn 1/Cp, 1/Cn, 1/L_a, 1/L_b, 1/L_c
    """

    def __init__(self, state_dim):
        super().__init__()
        self.p_raw = nn.Parameter(torch.zeros(state_dim))

    def get_matrix(self, z):
        p = F.softplus(self.p_raw)
        return torch.diag(p).unsqueeze(0).expand(z.size(0), -1, -1)

class PinvLinear(nn.Module):
    """
    P_inv = L L^T / sqrt(n)
    """

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.L = nn.Parameter(torch.eye(state_dim) * 0.1)

    def get_matrix(self, z):
        Pinv = self.L @ self.L.T / math.sqrt(self.state_dim)
        return Pinv.unsqueeze(0).expand(z.size(0), -1, -1)


class PinvDefault(nn.Module):
    """
    MLP backbone for P_inv
    """

    def __init__(self, state_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        self.state_dim = state_dim
        self.net = _build_backbone(arch, state_dim, hidden_dim, state_dim ** 2, depth)

    def get_matrix(self, z):
        """z: (batch, state_dim) -> P_inv: (batch, state_dim, state_dim)"""
        L = self.net(z).view(-1, self.state_dim, self.state_dim)
        return L @ L.permute(0, 2, 1) / math.sqrt(self.state_dim)

class PinvNominal(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        known_physics = torch.tensor([1/3.3e-3, 1/3.3e-3, 1/0.015, 1/0.015, 1/0.015])
        self.register_buffer("base_scale", known_physics)

        # Softplus(0.54) is approximately 1.0.
        self.p_raw = nn.Parameter(torch.ones(state_dim) * 0.54)

    def get_matrix(self, z):
        p = F.softplus(self.p_raw) * self.base_scale
        return torch.diag(p).unsqueeze(0).expand(z.size(0), -1, -1)


# =====================================================================
#  pHNN
# =====================================================================

class ConverterPHNN(nn.Module):
    """
        x = [q_p, q_n, phi_a, phi_b, phi_c]^T
        z = [v_c1, v_c2, i_a, i_b, i_c]^T
        P = diag(C_p, C_n, L_a, L_b, L_c)
        z = P_inv x
        dz/dt = P_inv @ ((J(S) - R) @ z + u)

        Parameters:
        state_dim : int     5
        sw_dim : int        6
        R : str             "prior_diag" | "linear" | "default"
        Pinv : str          "prior_diag" | "linear" | "default"
        arch : str          "mlp" | "kan"
    """

    def __init__(
            self,
            state_dim: int = 5,
            sw_dim: int = 6,
            hidden_dim: int = 64,
            depth: int = 3,
            R: str = "linear",
            Pinv: str = "linear",
            arch: str = "mlp",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.sw_dim = sw_dim

        # J
        self.J_module = JPrior(learnable_gain=False)

        # R
        if R == "prior_diag":
            self.R_module = RPriorDiag()
        elif R == "linear":
            self.R_module = RLinear(state_dim)
        elif R == "default":
            self.R_module = RDefault(state_dim, sw_dim, hidden_dim, depth, arch)
        elif R == "NonlinearDiag":
            self.R_module = RNonlinearDiag(state_dim, sw_dim, hidden_dim, depth, arch)
        elif R == "NonlinearDiag2":
            self.R_module = RNonlinearDiag2(state_dim, sw_dim, hidden_dim, depth, arch)
        else:
            raise ValueError(f"Unknown Rtype: {R}")

        # P_inv
        if Pinv == "prior_diag":
            self.Pinv_module = PinvPriorDiag(state_dim)
        elif Pinv == "linear":
            self.Pinv_module = PinvLinear(state_dim)
        elif Pinv == "default":
            self.Pinv_module = PinvDefault(state_dim, hidden_dim, depth, arch)
        elif Pinv == "nominal":
            self.Pinv_module = PinvNominal(state_dim)
        else:
            raise ValueError(f"Unknown Pinvtype: {Pinv}")

    def forward(self, z, u_ext, u_sw):
        """
        Parameters:
        z     : (batch, 5) [vcp, vcn, ia, ib, ic]
        u_ext : (batch, 5) [idcp, idcn, ea, eb, ec]
        u_sw  : (batch, 6)

        Returns:
        dz : (batch, 5)
        """

        grad_H = z
        J_gradH = self.J_module(u_sw, grad_H)
        R_gradH = self.R_module(z, u_sw, grad_H)

        # (J - R) @ grad_H
        rhs = J_gradH - R_gradH + u_ext

        # dz/dt = P_inv @ rhs
        Pinv = self.Pinv_module.get_matrix(z)
        dz = (Pinv @ rhs.unsqueeze(-1)).squeeze(-1)

        return dz

    def get_matrices(self, z, u_sw):

        J = self.J_module._build_J(u_sw)
        R = self.R_module.get_matrix(z, u_sw)
        Pinv = self.Pinv_module.get_matrix(z)

        return J, R, Pinv




