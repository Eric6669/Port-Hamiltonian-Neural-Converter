#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Plot real-time simulation comparison results."""

import os
import numpy as np
import scipy.io as sio
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

font_path = os.path.join(os.getcwd(), 'times.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    print(f"Success: Loaded font from {font_path}")
else:
    print(f"Error: Font file not found at {font_path}. Using default.")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

VAR_INFO = {
    'vcp': {'col': 0, 'label': r'$v_{dc,p}$', 'unit': 'V'},
    'vcn': {'col': 1, 'label': r'$v_{dc,n}$', 'unit': 'V'},
    'ia': {'col': 2, 'label': r'$i_{sa}$', 'unit': 'A'},
    'ib': {'col': 3, 'label': r'$i_{sb}$', 'unit': 'A'},
    'ic': {'col': 4, 'label': r'$i_{sc}$', 'unit': 'A'},
    'p': {'col': 5, 'label': r'$P$', 'unit': 'p.u.'},
    'q': {'col': 6, 'label': r'$Q$', 'unit': 'p.u.'},
    'vdc': {'col': 7, 'label': r'$v_{dc}$', 'unit': 'V'}
}


def calculate_nrmse(y_true, y_pred):
    """Calculate range-normalized RMSE in percent."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y_range = np.max(y_true) - np.min(y_true)
    if y_range == 0:
        return 0.0
    return (rmse / y_range) * 100.0


def plot_three_curves(t, y_true_all, y_ai_all, y_swf_all, var_key, save_path, zoom_x_range, inset_pos):
    """Plot IGBT, neural converter, and switching-function traces."""
    info = VAR_INFO[var_key]
    col = info['col']

    y_true = y_true_all[:, col]
    y_ai = y_ai_all[:, col]
    y_swf = y_swf_all[:, col]

    color_true = '#1f77b4'
    color_ai = '#ff7f0e'
    color_swf = '#2ca02c'
    color_err_axis = '#8b4513'

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(t, y_true, color=color_true, linewidth=1.5, alpha=0.9, label='DUB')
    ax.plot(t, y_ai, color=color_ai, linewidth=1.2, alpha=0.9, linestyle='--', label='pH-NODE')
    ax.plot(t, y_swf, color=color_swf, linewidth=1.2, alpha=0.8, linestyle='-.', label='SWF')

    ax.set_ylabel(f"{info['label']} ({info['unit']})", fontsize=24, labelpad=5)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=20, framealpha=0.9)
    ax.set_xlim(t[0], t[-1])

    ax2 = ax.twinx()
    err_ai = np.abs(y_true - y_ai)
    err_swf = np.abs(y_true - y_swf)
    ax2.fill_between(t, 0, err_ai, color=color_ai, alpha=0.3, edgecolor='none')
    ax2.fill_between(t, 0, err_swf, color=color_swf, alpha=0.2, edgecolor='none')
    ax2.set_ylabel(f"AbsError ({info['unit']})", fontsize=24, color=color_err_axis, labelpad=5)
    ax2.tick_params(axis='y', labelcolor=color_err_axis, labelsize=22)
    ax2.spines['right'].set_color(color_err_axis)
    ax2.spines['left'].set_color('black')

    max_err = max(np.max(err_ai), np.max(err_swf), 1e-6)
    ax2.set_ylim(0, max_err * 3.5)

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    if zoom_x_range is not None:
        axins = ax.inset_axes(inset_pos)
        axins.set_facecolor('white')
        axins.patch.set_alpha(0.95)

        axins.plot(t, y_true, color=color_true, linewidth=1.5, alpha=0.9)
        axins.plot(t, y_ai, color=color_ai, linewidth=1.2, alpha=0.9, linestyle='--')
        axins.plot(t, y_swf, color=color_swf, linewidth=1.2, alpha=0.8, linestyle='-.')

        z_x1, z_x2 = zoom_x_range
        idx1 = np.searchsorted(t, z_x1)
        idx2 = np.searchsorted(t, z_x2)

        if idx2 > idx1:
            window_true = y_true[idx1:idx2]
            window_ai = y_ai[idx1:idx2]
            window_swf = y_swf[idx1:idx2]

            z_y1 = min(np.min(window_true), np.min(window_ai), np.min(window_swf))
            z_y2 = max(np.max(window_true), np.max(window_ai), np.max(window_swf))

            margin = (z_y2 - z_y1) * 0.1 if z_y2 != z_y1 else 1e-3
            y_lower = z_y1 - margin
            y_upper = z_y2 + margin
            axins.set_xlim(z_x1, z_x2)
            axins.set_ylim(y_lower, y_upper)

            axins.tick_params(axis='both', which='major', labelsize=14)
            axins.grid(True, alpha=0.3)

            ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Time (s)', fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.close()
    print(f"[SAVE]: {save_path}")


if __name__ == '__main__':
    data_folder = 'RT_1.0s_dc_3000_3200_2.0s_p_0.8_0.6'

    print("Load .mat...")
    path_true = os.path.join(data_folder, 'Y_IGBT_40.mat')
    path_pred = os.path.join(data_folder, 'Y_ai_40.mat')
    path_swf = os.path.join(data_folder, 'Y_swf_40.mat')

    Y_true = sio.loadmat(path_true)['Y_truncated']
    Y_pred = sio.loadmat(path_pred)['Y_truncated']
    Y_swf = sio.loadmat(path_swf)['Y_truncated']

    min_len = min(Y_true.shape[0], Y_pred.shape[0], Y_swf.shape[0])
    Y_true = Y_true[:min_len, :]
    Y_pred = Y_pred[:min_len, :]
    Y_swf = Y_swf[:min_len, :]

    if '_40' in path_true:
        dt = 40e-6
    elif '_20' in path_true:
        dt = 20e-6
    else:
        raise ValueError('Cannot infer time step from file name.')

    t = np.arange(min_len) * dt
    skip_time = 0.5
    start_idx = int(skip_time / dt)

    t = t[start_idx:]
    Y_true = Y_true[start_idx:, :]
    Y_pred = Y_pred[start_idx:, :]
    Y_swf = Y_swf[start_idx:, :]
    t_plot = t - skip_time

    target_vars = ['p', 'q', 'vdc']

    print("\n" + "=" * 50)
    print(" NRMSE Analysis (after 0.5s)")
    print("=" * 50)

    for var in target_vars:
        col = VAR_INFO[var]['col']
        nrmse_ai = calculate_nrmse(Y_true[:, col], Y_pred[:, col])
        nrmse_swf = calculate_nrmse(Y_true[:, col], Y_swf[:, col])
        print(f"[{var.upper()}]")
        print(f"  - Neural Converter NRMSE: {nrmse_ai:.4f} %")
        print(f"  - Switching Function NRMSE: {nrmse_swf:.4f} %")
        print("-" * 50)

    print("\nGenerating plots...")
    for var in target_vars:
        save_name = os.path.join(data_folder, f'Result_HIL_{var}_Comparison.svg')
        plot_three_curves(t_plot, Y_true, Y_pred, Y_swf, var_key=var, save_path=save_name,
                          zoom_x_range=[1.45, 1.55],
                          inset_pos=[0.30, 0.45, 0.4, 0.45])
    print("All tasks completed.")
