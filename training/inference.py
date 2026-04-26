#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""

import os
import argparse
import time
import numpy as np
import scipy.io as sio
import torch
import matplotlib.font_manager as fm
import matplotlib
import matplotlib.pyplot as plt
from model.NODE import ConverterNODE
from model.pHNN import ConverterPHNN
from utils.integrator import rk4_integrate, euler_integrate

font_path = os.path.join(os.getcwd(), 'times.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
    print(f"Success: Loaded font from {font_path}")
else:
    print(f"Error: Font file not found at {font_path}. Using default.")
    plt.rcParams['font.family'] = 'serif'

def load_raw_trajectory(mat_path, converter_model="Switching", t_start_idx=0):

    offset = 0 if converter_model == "Switching" else 21
    mat_data = sio.loadmat(mat_path)
    data = mat_data["clean_data"]

    z = data[t_start_idx:, offset + 0: offset + 5]
    z[:, 2:5] = -z[:, 2:5]
    u_ext = data[t_start_idx:, offset + 5: offset + 10]
    u_sw = data[t_start_idx:, offset + 13: offset + 19]

    return z, u_ext, u_sw


@torch.no_grad()
def rollout(model, z0, u_seq, sw_seq, dt, chunk_size=10, integrate_fn=rk4_integrate):

    T = u_seq.shape[1]
    device = next(model.parameters()).device
    z = z0.to(device)
    u_seq = u_seq.to(device)
    sw_seq = sw_seq.to(device)

    all_preds = []
    pos = 0

    while pos < T - 1:
        L = min(chunk_size, T - 1 - pos)

        u_chunk = u_seq[:, pos:pos + L + 1, :]
        sw_chunk = sw_seq[:, pos:pos + L + 1, :]

        z_chunk = integrate_fn(model, z, u_chunk, sw_chunk, dt, L)  # (1, L, 5)
        all_preds.append(z_chunk.squeeze(0).cpu())

        z = z_chunk[:, -1, :]
        pos += L

    return torch.cat(all_preds, dim=0).numpy()  # (T-1, 5)


def plot_trajectories(t, z_true, z_pred, save_path, dt):
    channel_names = [
        r'$v_{cp}$ (V)', r'$v_{cn}$ (V)',
        r'$i_a$ (A)', r'$i_b$ (A)', r'$i_c$ (A)'
    ]

    color_true = '#1f77b4'
    color_pred = '#ff7f0e'
    color_err_fill = '#d0a9a9'
    color_err_axis = '#8b4513'

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    T_pred = z_pred.shape[0]
    t_pred = t[1:T_pred + 1]

    z_true_aligned = z_true[1:T_pred + 1]

    for i, ax in enumerate(axes):
        ax.plot(t[:T_pred + 1], z_true[:T_pred + 1, i],
                color=color_true, linewidth=1.5, alpha=0.9, label='Ground Truth')
        ax.plot(t_pred, z_pred[:, i],
                color=color_pred, linewidth=1.2, alpha=0.8, linestyle='--', label='NODE')

        ax.set_ylabel(channel_names[i], fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=16)

        ax2 = ax.twinx()
        abs_err = np.abs(z_true_aligned[:, i] - z_pred[:, i])
        ax2.fill_between(t_pred, 0, abs_err, color=color_err_fill, alpha=0.5, edgecolor='none')

        unit = '(V)' if i < 2 else '(A)'
        ax2.set_ylabel(f'AbsError {unit}', fontsize=16, color=color_err_axis)
        ax2.tick_params(axis='y', labelcolor=color_err_axis, labelsize=16)

        ax2.spines['right'].set_color(color_err_axis)
        ax2.spines['left'].set_color('black')

        max_err = np.max(abs_err)
        if max_err < 1e-6:
            max_err = 1e-6
        ax2.set_ylim(0, max_err * 3)

    axes[-1].set_xlabel('Time (s)', fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.close()
    print(f"[SAVE] {save_path}")


def compute_errors(z_true, z_pred):

    T = z_pred.shape[0]
    z_t = z_true[1:T + 1]

    channel_names = ['vcp', 'vcn', 'ia', 'ib', 'ic']

    print(f"\n{'Channel':<8s} {'MAE':>12s} {'RMSE':>12s} {'MAPE(%)':>12s}")
    print("-" * 48)

    for i, name in enumerate(channel_names):
        err = np.abs(z_t[:, i] - z_pred[:, i])
        mae = err.mean()
        rmse = np.sqrt((err ** 2).mean())

        denom = np.abs(z_t[:, i]).clip(min=1e-6)
        mape = (err / denom).mean() * 100

        print(f"{name:<8s} {mae:>12.4f} {rmse:>12.4f} {mape:>12.2f}")

    total_mae = np.abs(z_t - z_pred).mean()
    total_rmse = np.sqrt(((z_t - z_pred) ** 2).mean())
    print(f"{'Overall':<8s} {total_mae:>12.4f} {total_rmse:>12.4f}")

class Inference:

    def __init__(self, args):

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = os.path.join('datasets', 'raw')
        self.save_dir = os.path.join('results', self.args.converter_model)
        self.dt = args.Ts

    def run(self):

        # ---------------------------------------------------
        # 1. Load Raw Trajectory
        # ---------------------------------------------------
        mat_path = os.path.join(self.data_dir, self.args.infer_file)
        z_raw, u_raw, sw_raw = load_raw_trajectory(mat_path, self.args.converter_model)
        T = z_raw.shape[0]
        t = np.arange(T) * self.dt

        meta_path = os.path.join('datasets', 'processed', f'{self.args.converter_model}_meta.pt')
        meta = torch.load(meta_path, weights_only=False)
        scalers = meta.get("scalers", None)

        if scalers is not None:
            scaler_z = scalers["z"]
            scaler_u = scalers["u_ext"]
            z_norm = scaler_z.transform(z_raw)
            u_norm = scaler_u.transform(u_raw)
        else:
            z_norm, u_norm = z_raw, u_raw

        # ---------------------------------------------------
        # 2. Load Model
        # ---------------------------------------------------
        if self.args.converter_neural_model == "ConverterPHNN":
            model = ConverterPHNN(
                hidden_dim=self.args.hidden_dim,
                depth=self.args.layers,
                R=self.args.R_type,
                Pinv=self.args.Pinv_type,
                arch=self.args.arch,
            ).to(self.device)
        elif self.args.converter_neural_model == "ConverterNODE":
            model = ConverterNODE(
                hidden_dim=self.args.hidden_dim,
                depth=self.args.layers
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model: {self.args.converter_neural_model}")

        ckpt_path = os.path.join(
            'checkpoints',
            f"{self.args.converter_model}_{self.args.converter_neural_model}_R{self.args.R_type}_Pinv{self.args.Pinv_type}_arch{self.args.arch}.pt"
        )

        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        model.to(self.device)
        model.eval()

        # ---------------------------------------------------
        # 3. Rollout Inference
        # ---------------------------------------------------
        if self.args.integrate_method == "rk4":
            integrate_fn = rk4_integrate
        elif self.args.integrate_method == "euler":
            integrate_fn = euler_integrate
        else:
            raise ValueError(f"Unknown integrate_method: {self.args.integrate_method}")

        z_tensor = torch.from_numpy(z_norm.astype(np.float32)).unsqueeze(0)  # (1, T, 5)
        u_tensor = torch.from_numpy(u_norm.astype(np.float32)).unsqueeze(0)  # (1, T, 5)
        sw_tensor = torch.from_numpy(sw_raw.astype(np.float32)).unsqueeze(0)  # (1, T, 6)
        z0 = z_tensor[:, 0, :]

        t0 = time.time()
        z_pred_norm = rollout(model, z0, u_tensor, sw_tensor, self.dt,
                              chunk_size=self.args.chunk_size,
                              integrate_fn=integrate_fn)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({T / elapsed:.0f} steps/s)")

        if scalers is not None:
            z_pred_raw = scaler_z.inverse_transform(z_pred_norm)
        else:
            z_pred_raw = z_pred_norm

        compute_errors(z_raw, z_pred_raw)

        # ---------------------------------------------------
        # 4. Plot trajectories
        # ---------------------------------------------------

        os.makedirs(self.save_dir, exist_ok=True)
        mat_name = os.path.splitext(self.args.infer_file)[0]
        save_path = os.path.join(self.save_dir, f"{mat_name}_R{self.args.R_type}_Pinv{self.args.Pinv_type}.svg")
        plot_trajectories(t, z_raw, z_pred_raw, save_path, self.dt)
