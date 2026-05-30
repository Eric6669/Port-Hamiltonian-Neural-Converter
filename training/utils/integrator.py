#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：integrator.py
@Author  ：He Xing
@Date    ：2026/4/5 21:16 
"""
import torch


def rk4_integrate(f_theta, z0, u_seq, sw_seq, dt, steps):
    """
    Parameters
    -------
    f_theta : callable
        f(z, u, sw) -> dz, ConverterPHNN.forward
    z0 : (batch, state_dim)
    u_seq : (batch, L+1, u_dim)
    sw_seq : (batch, L+1, sw_dim)
    dt : float
    steps : int

    Returns
    -------
    z_pred : (batch, L, state_dim)
        [z_1, z_2, ..., z_L] (except z_0)
    """
    z = z0
    trajectory = []

    for k in range(steps):
        u_k = u_seq[:, k, :]
        sw_k = sw_seq[:, k, :]

        # RK4: u, sw -  Zero-Order Hold
        k1 = f_theta(z, u_k, sw_k)
        k2 = f_theta(z + 0.5 * dt * k1, u_k, sw_k)
        k3 = f_theta(z + 0.5 * dt * k2, u_k, sw_k)
        k4 = f_theta(z + dt * k3, u_k, sw_k)

        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(z)

    return torch.stack(trajectory, dim=1)  # (batch, L, state_dim)


def euler_integrate(f_theta, z0, u_seq, sw_seq, dt, steps):
    """
    Parameters
    -------
    f_theta : callable
        f(z, u, sw) -> dz, ConverterPHNN.forward
    z0 : (batch, state_dim)
    u_seq : (batch, L+1, u_dim)
    sw_seq : (batch, L+1, sw_dim)
    dt : float
    steps : int

    Returns
    -------
    z_pred : (batch, L, state_dim)
        [z_1, z_2, ..., z_L] (except z_0)
    """
    z = z0
    trajectory = []

    for k in range(steps):
        u_k = u_seq[:, k, :]
        sw_k = sw_seq[:, k, :]

        # z_{k+1} = z_k + dt * f(z_k, u_k, sw_k)
        dz = f_theta(z, u_k, sw_k)
        z = z + dt * dz

        trajectory.append(z)

    return torch.stack(trajectory, dim=1)  # (batch, L, state_dim)
