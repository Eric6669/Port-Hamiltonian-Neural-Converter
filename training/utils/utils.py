#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""

import os
import matplotlib.pyplot as plt

def normalized_mse(x, target):
    """Compute normalized mean squared error."""
    std = target.std(dim=0).clamp(min=1e-8)  # (n_channels,)
    return ((x - target) ** 2 / std ** 2).mean()

def normalized_mae(x, target):
    """Compute normalized mean absolute error."""
    std = target.std(dim=0).clamp(min=1e-8)  # (n_channels,)
    return ((x - target).abs() / std).mean()


def plot_loss_curves(train_losses, val_losses, val_epochs, save_dir, filename="loss_curve.png"):

    plt.figure(figsize=(10, 6))

    train_epochs = range(1, len(train_losses) + 1)

    plt.plot(train_epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    if val_losses and val_epochs:
        plt.plot(val_epochs, val_losses, label='Val Loss', color='#ff7f0e', linewidth=2)

    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()