#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：inference.py
@Author  ：He Xing
@Date    ：2026/4/6 13:36 
"""

import os
import argparse
import time
import numpy as np
import scipy.io as sio
import torch
import matplotlib.font_manager as fm
import matplotlib
import matplotlib.pyplot as plt

from model.pHNODE import pHNODE
from model.NODE import NODE
from model.PINODE import PINODE
from model.LSTM import LSTM

from utils.integrator import rk4_integrate, euler_integrate

font_path = os.path.join(os.getcwd(), 'times.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
    print(f"Success: Loaded font from {font_path}")
else:
    print(f"Error: Font file not found at {font_path}. Using default.")
    plt.rcParams['font.family'] = 'serif'

COLUMN_LAYOUT = {
    "2level": {
        "offset": 21,
        "z": (0, 5),
        "u_ext": (5, 10),
        "u_sw": (13, 19),
    },
    "3level": {
        "offset": 0,
        "z": (0, 5),
        "u_ext": (5, 10),
        "u_sw": (13, 25),
    },
}

ODE_MODEL_NAMES = ["pHNODE", "NODE", "PINODE"]
SEQ_MODEL_NAMES = ["LSTM"]


def build_model(args):
    if args.converter_neural_model == "pHNODE":
        return pHNODE(
            converter_model=args.converter_model,
            hidden_dim=args.hidden_dim,
            depth=args.layers,
        )

    elif args.converter_neural_model == "NODE":
        return NODE(
            converter_model=args.converter_model,
            hidden_dim=args.hidden_dim,
            depth=args.layers,
        )

    elif args.converter_neural_model == "PINODE":
        return PINODE(
            converter_model=args.converter_model,
            hidden_dim=args.hidden_dim,
            depth=args.layers,
            physics_weight=getattr(args, "physics_weight", 0.5),
        )

    elif args.converter_neural_model == "LSTM":
        return LSTM(
            converter_model=args.converter_model,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            dropout=getattr(args, "lstm_dropout", 0.0),
        )

    else:
        raise ValueError(f"Unknown model: {args.converter_neural_model}")

def get_data_tag(args):
    if getattr(args, "normalization", 0) == 1:
        return f"{args.converter_model}_norm"
    return args.converter_model

def load_raw_trajectory(mat_path, converter_model, t_start_idx=0):

    if converter_model not in COLUMN_LAYOUT:
        raise ValueError(f"Unknown converter_model: {converter_model}")

    layout = COLUMN_LAYOUT[converter_model]
    off = layout["offset"]

    mat_data = sio.loadmat(mat_path)
    data = mat_data["clean_data"]

    z_start, z_end = layout["z"]
    u_start, u_end = layout["u_ext"]
    sw_start, sw_end = layout["u_sw"]

    z = data[t_start_idx:, off + z_start: off + z_end].copy()
    z[:, 2:5] = -z[:, 2:5]

    u_ext = data[t_start_idx:, off + u_start: off + u_end]
    u_sw = data[t_start_idx:, off + sw_start: off + sw_end]

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


@torch.no_grad()
def rollout_lstm(model, z0, u_seq, sw_seq):
    """
    Autoregressive LSTM rollout:
        z_{k+1} = LSTM(z_k, u_k, s_k, h_k)

    Inputs:
        z0     : (1, 5)
        u_seq  : (1, T, 5)
        sw_seq : (1, T, sw_dim)

    Returns:
        z_pred : (T-1, 5), predicted z_1 ... z_{T-1}
    """
    device = next(model.parameters()).device

    z = z0.to(device)
    u_seq = u_seq.to(device)
    sw_seq = sw_seq.to(device)

    hidden = None
    preds = []

    T = u_seq.shape[1]

    for k in range(T - 1):
        u_k = u_seq[:, k, :]
        sw_k = sw_seq[:, k, :]

        z_next, hidden = model(z, u_k, sw_k, hidden)
        preds.append(z_next.squeeze(0).cpu())

        z = z_next

    return torch.stack(preds, dim=0).numpy()

def plot_trajectories(t, z_true, z_pred, save_path, dt, start_offset=1):
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
    t_pred = t[start_offset:start_offset + T_pred]

    z_true_aligned = z_true[start_offset:start_offset + T_pred]

    for i, ax in enumerate(axes):
        ax.plot(t_pred, z_true_aligned[:, i],
                color=color_true, linewidth=1.5, alpha=0.9, label='Ground Truth')
        ax.plot(t_pred, z_pred[:, i],
                color=color_pred, linewidth=1.2, alpha=0.8, linestyle='--', label='Prediction')

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


def compute_errors(z_true, z_pred, start_offset=1):

    T = z_pred.shape[0]
    z_t = z_true[start_offset:start_offset + T]

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
        self.data_dir = os.path.join("datasets", self.args.converter_model, "raw")
        self.save_dir = os.path.join("results", self.args.converter_model)
        self.dt = args.Ts

    def run(self):

        # ---------------------------------------------------
        # 1. Load Raw Trajectory
        # ---------------------------------------------------
        mat_path = os.path.join(self.data_dir, self.args.infer_file)
        z_raw, u_raw, sw_raw = load_raw_trajectory(mat_path, self.args.converter_model)
        T = z_raw.shape[0]
        t = np.arange(T) * self.dt

        data_tag = get_data_tag(self.args)

        meta_path = os.path.join(
            "datasets",
            self.args.converter_model,
            "processed",
            f"{data_tag}_meta.pt"
        )

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
        model = build_model(self.args).to(self.device)

        ckpt_path = os.path.join(
            "checkpoints",
            f"{self.args.converter_model}_{self.args.converter_neural_model}.pt"
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
        sw_tensor = torch.from_numpy(sw_raw.astype(np.float32)).unsqueeze(0)  # (1, T, 6/12)

        t0 = time.time()

        if self.args.converter_neural_model in ODE_MODEL_NAMES:
            z0 = z_tensor[:, 0, :]

            z_pred_norm = rollout(
                model, z0, u_tensor, sw_tensor, self.dt,
                chunk_size=self.args.chunk_size,
                integrate_fn=integrate_fn
            )
            start_offset = 1


        elif self.args.converter_neural_model in SEQ_MODEL_NAMES:
            z0 = z_tensor[:, 0, :]
            z_pred_norm = rollout_lstm(
                model, z0, u_tensor, sw_tensor
            )
            start_offset = 1

        else:
            raise ValueError(f"Unknown model: {self.args.converter_neural_model}")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({T / elapsed:.0f} steps/s)")

        if scalers is not None:
            z_pred_raw = scaler_z.inverse_transform(z_pred_norm)
        else:
            z_pred_raw = z_pred_norm

        compute_errors(z_raw, z_pred_raw, start_offset=start_offset)

        # ---------------------------------------------------
        # 4. Plot trajectories
        # ---------------------------------------------------

        os.makedirs(self.save_dir, exist_ok=True)
        mat_name = os.path.splitext(self.args.infer_file)[0]
        save_path = os.path.join(
            self.save_dir,
            f"{mat_name}_{self.args.converter_neural_model}.svg"
        )
        plot_trajectories(
            t, z_raw, z_pred_raw, save_path, self.dt,
            start_offset=start_offset
        )
