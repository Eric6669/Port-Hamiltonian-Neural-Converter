#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：offline_plot_result.py
@Author  ：He Xing
@Date    ：2026/4/23 10:29 
"""

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
    plt.rcParams['mathtext.rm'] = 'Times New Roman'  # 正体
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'  # 粗体

    print(f"Success: Loaded font from {font_path} and forced MathText to use it.")
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

def plot_selected_variables(
        t, Y_true, Y_pred, vars_to_plot, save_path,
        true_label='DUB',
        pred_label='pH-NODE',
        zoom_x_range=None,
        inset_pos=(0.55, 0.48, 0.38, 0.38)
):

    num_vars = len(vars_to_plot)
    if num_vars == 0:
        print("No chosen variables！")
        return

    color_true = '#1f77b4'
    color_pred = '#ff7f0e'
    color_err_fill = '#d0a9a9'
    color_err_axis = '#8b4513'

    fig, axes = plt.subplots(num_vars, 1, figsize=(14, 3.5 * num_vars), sharex=True)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        var_key = vars_to_plot[i]
        info = VAR_INFO[var_key]
        col = info['col']

        y_true_val = Y_true[:, col]
        y_pred_val = Y_pred[:, col]

        ax.plot(t, y_true_val, color=color_true, linewidth=1.5, alpha=0.9, label=true_label)
        ax.plot(t, y_pred_val, color=color_pred, linewidth=1.2, alpha=0.9, linestyle='--', label=pred_label)

        y_label_str = f"{info['label']} ({info['unit']})"
        ax.set_ylabel(y_label_str, fontsize=24, labelpad=0)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=22)
        ax.set_xlim(t[0], t[-1])

        ax2 = ax.twinx()
        abs_err = np.abs(y_true_val - y_pred_val)
        ax2.fill_between(t, 0, abs_err, color=color_err_fill, alpha=0.5, edgecolor='none')

        ax2.set_ylabel(f"AbsError ({info['unit']})", fontsize=24, color=color_err_axis, labelpad=5)
        ax2.tick_params(axis='y', labelcolor=color_err_axis, labelsize=22)

        ax2.spines['right'].set_color(color_err_axis)
        ax2.spines['left'].set_color('black')

        max_err = np.max(abs_err)
        if max_err < 1e-6:
            max_err = 1e-6
        ax2.set_ylim(0, max_err * 3)

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        # Inset zoom, only enabled when zoom_x_range is provided
        if zoom_x_range is not None:
            axins = ax.inset_axes(inset_pos)
            axins.set_facecolor('white')
            axins.patch.set_alpha(0.95)

            # Re-plot the two waveforms in the inset
            axins.plot(
                t, y_true_val,
                color=color_true,
                linewidth=1.5,
                alpha=0.9
            )
            axins.plot(
                t, y_pred_val,
                color=color_pred,
                linewidth=1.2,
                alpha=0.9,
                linestyle='--'
            )

            # Automatically determine the y-axis range in the zoomed window
            z_x1, z_x2 = zoom_x_range
            idx1 = np.searchsorted(t, z_x1)
            idx2 = np.searchsorted(t, z_x2)

            if idx2 > idx1:
                window_true = y_true_val[idx1:idx2]
                window_pred = y_pred_val[idx1:idx2]

                z_y1 = min(np.min(window_true), np.min(window_pred))
                z_y2 = max(np.max(window_true), np.max(window_pred))

                margin = (z_y2 - z_y1) * 0.1 if z_y2 != z_y1 else 1e-3

                axins.set_xlim(z_x1, z_x2)
                axins.set_ylim(z_y1 - margin, z_y2 + margin)

                axins.tick_params(axis='both', which='major', labelsize=14)
                axins.grid(True, alpha=0.3)

                ax.indicate_inset_zoom(
                    axins,
                    edgecolor="black",
                    alpha=0.8,
                    linewidth=1.5
                )

    axes[-1].set_xlabel('Time (s)', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.close()
    print(f"[SAVE]: {save_path}")

def crop_to_same_length(Y_true, Y_pred):
    min_len = min(Y_true.shape[0], Y_pred.shape[0])
    return Y_true[:min_len], Y_pred[:min_len]

def run_plot_case(data_folder, true_file, true_key,
                  pred_file='Y_pred.mat', pred_key='Y_pred',
                  true_label='DUB',
                  dt=20e-6,
                  skip_time=1.5,
                  vars_to_plot=None,
                  zoom_x_range=None,
                  inset_pos=(0.55, 0.48, 0.38, 0.38)):

    if vars_to_plot is None:
        vars_to_plot = ['vcp', 'vcn', 'p', 'q', 'vdc', 'ia', 'ib', 'ic']

    print("\n" + "=" * 60)
    print(f"[CASE]: {data_folder}")
    print("=" * 60)

    path_true = os.path.join(data_folder, true_file)
    path_pred = os.path.join(data_folder, pred_file)

    print(f"[LOAD]: {path_true}")
    print(f"[LOAD]: {path_pred}")

    Y_true = sio.loadmat(path_true)[true_key]
    Y_pred = sio.loadmat(path_pred)[pred_key]

    Y_true, Y_pred = crop_to_same_length(Y_true, Y_pred)

    N = Y_true.shape[0]
    t = np.arange(N) * dt

    start_idx = int(skip_time / dt)

    t = t[start_idx:]
    Y_true = Y_true[start_idx:]
    Y_pred = Y_pred[start_idx:]

    os.makedirs(data_folder, exist_ok=True)

    for var in vars_to_plot:
        save_name = os.path.join(data_folder, f"Result_{data_folder}_{var}.svg")

        plot_selected_variables(
            t=t,
            Y_true=Y_true,
            Y_pred=Y_pred,
            vars_to_plot=[var],
            save_path=save_name,
            true_label=true_label,
            pred_label='pH-NODE',
            zoom_x_range=zoom_x_range,
            inset_pos=inset_pos
        )


if __name__ == '__main__':
    single_vars = ['vcp', 'vcn', 'p', 'q', 'vdc', 'ia', 'ib', 'ic']

    # =========================================================
    # 1. close_loop_2level
    # Compare:
    #   Y_IGBT.mat vs Y_pred.mat
    # =========================================================
    run_plot_case(
        data_folder='close_loop_2level',
        true_file='Y_IGBT.mat',
        true_key='Y_IGBT',
        pred_file='Y_pred.mat',
        pred_key='Y_pred',
        true_label='DUB',
        dt=20e-6,
        skip_time=1.5,
        vars_to_plot=single_vars,
        zoom_x_range=(2.50, 2.65),
        # inset_pos=(0.50, 0.12, 0.27, 0.32) # p
        # inset_pos=(0.15, 0.12, 0.27, 0.32) # q
        inset_pos=(0.14, 0.10, 0.27, 0.32) # vdc
    )

    # =========================================================
    # 2. closed_loop_3level
    # Compare:
    #   Y_NPC.mat vs Y_pred.mat
    # =========================================================
    run_plot_case(
        data_folder='closed_loop_3level',
        true_file='Y_NPC.mat',
        true_key='Y_NPC',
        pred_file='Y_pred.mat',
        pred_key='Y_pred',
        true_label='NPC',
        dt=20e-6,
        skip_time=1.5,
        vars_to_plot=single_vars,
        zoom_x_range=(2.50, 2.65),
        # inset_pos=(0.47, 0.12, 0.27, 0.32) # p
        # inset_pos=(0.30, 0.12, 0.27, 0.32)  # q
        inset_pos=(0.30, 0.12, 0.27, 0.32)  # vdc
    )

    # =========================================================
    # 3. open_loop_2level
    # Compare:
    #   Y_IGBT.mat vs Y_pred.mat
    # =========================================================
    run_plot_case(
        data_folder='open_loop_2level',
        true_file='Y_IGBT.mat',
        true_key='Y_IGBT',
        pred_file='Y_pred.mat',
        pred_key='Y_pred',
        true_label='DUB',
        dt=20e-6,
        skip_time=1.5,
        vars_to_plot=single_vars,
        zoom_x_range=(2.50, 2.65),
        # inset_pos=(0.47, 0.12, 0.27, 0.32) # p
        inset_pos=(0.15, 0.12, 0.27, 0.32)  # vdc
    )


