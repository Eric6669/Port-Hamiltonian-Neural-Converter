#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Plot offline Simulink comparison results."""

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


def plot_selected_variables(t, Y_true, Y_comp, vars_to_plot, save_path, comp_label, zoom_x_range, inset_pos):
    """Plot selected channels and their absolute-error envelopes."""
    num_vars = len(vars_to_plot)
    if num_vars == 0:
        print("No variables were selected.")
        return

    color_true = '#1f77b4'
    color_pred = '#ff7f0e'
    color_err_fill = '#d0a9a9'
    color_err_axis = '#8b4513'

    fig, axes = plt.subplots(num_vars, 1, figsize=(14, 3 * num_vars), sharex=True)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        var_key = vars_to_plot[i]
        info = VAR_INFO[var_key]
        col = info['col']

        y_true_val = Y_true[:, col]
        y_comp_val = Y_comp[:, col]

        ax.plot(t, y_true_val, color=color_true, linewidth=1.5, alpha=0.9, label='DUB')
        ax.plot(t, y_comp_val, color=color_pred, linewidth=1.2, alpha=0.8, linestyle='--', label=comp_label)

        ax.set_ylabel(f"{info['label']} ({info['unit']})", fontsize=24, labelpad=0)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=22)
        ax.set_xlim(t[0], t[-1])

        ax2 = ax.twinx()
        abs_err = np.abs(y_true_val - y_comp_val)
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
        if zoom_x_range is not None:

            axins = ax.inset_axes(inset_pos)
            axins.set_facecolor('white')
            axins.patch.set_alpha(0.95)

            axins.plot(t, y_true_val, color=color_true, linewidth=1.5, alpha=0.9)
            axins.plot(t, y_comp_val, color=color_pred, linewidth=1.2, alpha=0.8, linestyle='--')

            z_x1, z_x2 = zoom_x_range
            idx1 = np.searchsorted(t, z_x1)
            idx2 = np.searchsorted(t, z_x2)

            if idx2 > idx1:
                window_true = y_true_val[idx1:idx2]
                window_comp = y_comp_val[idx1:idx2]
                z_y1 = min(np.min(window_true), np.min(window_comp))
                z_y2 = max(np.max(window_true), np.max(window_comp))

                margin = (z_y2 - z_y1) * 0.1 if z_y2 != z_y1 else 1e-3
                axins.set_xlim(z_x1, z_x2)
                axins.set_ylim(z_y1 - margin, z_y2 + margin)

                axins.tick_params(axis='both', which='major', labelsize=14)
                axins.grid(True, alpha=0.3)

                ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8, linewidth=1.5)

    axes[-1].set_xlabel('Time (s)', fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.close()
    print(f"[SAVE]: {save_path}")


if __name__ == '__main__':
    # data_folder = 'open_1.0s_ac_1.0_0.8_2.0s_dc_3000_2800'
    data_folder = 'close_1.0s_p_0.8_0.6_2.0s_dc_3000_2800'

    print("Load .mat...")
    path_true = os.path.join(data_folder, 'Y_IGBT.mat')
    path_pred = os.path.join(data_folder, 'Y_pred.mat')
    path_swf = os.path.join(data_folder, 'Y_SWF.mat')

    Y_true = sio.loadmat(path_true)['Y_IGBT']
    Y_pred = sio.loadmat(path_pred)['Y_pred']
    Y_SWF = sio.loadmat(path_swf)['Y_SWF']

    dt = 20e-6
    N = Y_true.shape[0]
    t = np.arange(N) * dt

    skip_time = 0.5
    start_idx = int(skip_time / dt)

    t = t[start_idx:]
    Y_true = Y_true[start_idx:]
    Y_pred = Y_pred[start_idx:]
    Y_SWF = Y_SWF[start_idx:]

    single_vars = ['vcp', 'vcn', 'p', 'q', 'vdc', 'ia', 'ib', 'ic']
    for var in single_vars:
        save_name = os.path.join(data_folder, f'Result_{var}_IGBTvPred.svg')
        plot_selected_variables(t - 0.5, Y_true, Y_pred, [var], save_name, comp_label='pH-NODE',
                                zoom_x_range=[1.45, 1.55],  # [1.45, 1.55]
                                inset_pos=[0.20, 0.15, 0.4, 0.45])  # [0.25, 0.15, 0.4, 0.45]
                                # [left bound, low bound, width, height] percentage
