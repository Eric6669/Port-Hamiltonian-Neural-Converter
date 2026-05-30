#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：PINODE.py
@Author  ：He Xing
@Date    ：2026/5/23 18:44 
"""

import torch
import torch.nn.functional as F

from model.NODE import NODE


class PINODE(NODE):
    """
    PI-NODE baseline:
        - forward dynamics: black-box NODE
        - physics constraint: soft current-balance loss

    State order:
        z = [vcp, vcn, ia, ib, ic]
    """

    def __init__(
        self,
        converter_model: str = "2level",
        state_dim: int = 5,
        u_ext_dim: int = 5,
        hidden_dim: int = 128,
        depth: int = 4,
        sw_dim: int = None,
        physics_weight: float = 0.5,
    ):
        super().__init__(
            converter_model=converter_model,
            state_dim=state_dim,
            u_ext_dim=u_ext_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            sw_dim=sw_dim,
        )

        self.physics_weight = physics_weight

    @staticmethod
    def current_balance_loss(z_pred):
        """
        Physical soft constraint:
            ia + ib + ic = 0

        Parameters
        ----------
        z_pred : Tensor
            Shape (B, T, 5) or (B, 5).

        Returns
        -------
        loss_phy : Tensor
            Scalar physics loss.
        """
        if z_pred.dim() == 3:
            ia = z_pred[:, :, 2]
            ib = z_pred[:, :, 3]
            ic = z_pred[:, :, 4]
        elif z_pred.dim() == 2:
            ia = z_pred[:, 2]
            ib = z_pred[:, 3]
            ic = z_pred[:, 4]
        else:
            raise ValueError(
                f"z_pred must have shape (B, 5) or (B, T, 5), "
                f"but got {tuple(z_pred.shape)}."
            )

        return torch.mean((ia + ib + ic) ** 2)

    def total_loss(
        self,
        z_pred,
        z_target,
        loss_weight_v: float = 1.0,
        loss_weight_i: float = 1.0,
        physics_weight: float = None,
    ):
        """
        Weighted data loss + soft physical constraint.

        This matches your current pH-NODE loss structure:
            loss_v = MSE(z_pred[:, :, 0:2], z_target[:, :, 0:2])
            loss_i = MSE(z_pred[:, :, 2:5], z_target[:, :, 2:5])
            loss = w_v * loss_v + w_i * loss_i

        Then add:
            physics_weight * mean((ia + ib + ic)^2)

        Parameters
        ----------
        z_pred : Tensor
            Shape (B, T, 5) or (B, 5).
        z_target : Tensor
            Same shape as z_pred.
        loss_weight_v : float
            Weight for dc-voltage states.
        loss_weight_i : float
            Weight for ac-current states.
        physics_weight : float or None
            Weight for current-balance physical constraint.

        Returns
        -------
        loss : Tensor
            Total scalar loss.
        loss_dict : dict
            Detached loss components for logging.
        """
        if physics_weight is None:
            physics_weight = self.physics_weight

        if z_pred.dim() == 3:
            loss_v = F.mse_loss(z_pred[:, :, 0:2], z_target[:, :, 0:2])
            loss_i = F.mse_loss(z_pred[:, :, 2:5], z_target[:, :, 2:5])
        elif z_pred.dim() == 2:
            loss_v = F.mse_loss(z_pred[:, 0:2], z_target[:, 0:2])
            loss_i = F.mse_loss(z_pred[:, 2:5], z_target[:, 2:5])
        else:
            raise ValueError(
                f"z_pred must have shape (B, 5) or (B, T, 5), "
                f"but got {tuple(z_pred.shape)}."
            )

        loss_data = loss_weight_v * loss_v + loss_weight_i * loss_i
        loss_phy = self.current_balance_loss(z_pred)

        loss = loss_data + physics_weight * loss_phy

        loss_dict = {
            "loss_total": loss.detach(),
            "loss_data": loss_data.detach(),
            "loss_v": loss_v.detach(),
            "loss_i": loss_i.detach(),
            "loss_phy": loss_phy.detach(),
        }

        return loss, loss_dict