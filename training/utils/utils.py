#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：utils.py
@Author  ：He Xing
@Date    ：2026/4/2 22:28 
"""

import os
import numpy as np
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
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

def plot_loss_curves(train_losses, val_losses, val_epochs, save_dir, filename="loss_curve.png"):
    # ---------------- Data ----------------
    train_epochs = np.arange(1, len(train_losses) + 1)

    color_train = '#1f77b4'
    color_val = '#ff7f0e'

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(
        train_epochs,
        train_losses,
        color=color_train,
        linewidth=2.2,
        alpha=0.95,
        label='Training Loss'
    )

    if val_losses is not None and len(val_losses) > 0 and val_epochs is not None and len(val_epochs) > 0:
        ax.plot(
            val_epochs,
            val_losses,
            color=color_val,
            linewidth=2.2,
            alpha=0.95,
            linestyle='--',
            marker='o',
            markersize=4.5,
            markerfacecolor='white',
            markeredgewidth=1.2,
            label='Validation Loss'
        )

    # ---------------- Axis style ----------------
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=22)
    ax.set_ylabel('Loss', fontsize=22)

    ax.tick_params(axis='both', which='major', labelsize=20, direction='in', length=5, width=1.0)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)

    ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.35)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.25)

    ax.legend(
        loc='upper right',
        fontsize=18,
        framealpha=0.95,
        edgecolor='black'
    )

    # Make borders clearer
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # Reasonable x range
    ax.set_xlim(train_epochs[0], train_epochs[-1])

    # ---------------- Save ----------------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.tight_layout()

    # SVG for paper; PNG fallback can be obtained by changing filename
    if filename.lower().endswith(".svg"):
        plt.savefig(save_path, bbox_inches='tight', format='svg')
    else:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

    plt.close()
    print(f"[SAVE]: {save_path}")