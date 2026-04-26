#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""

import os
import time
import torch
import argparse
import numpy as np
import optuna
from torch.utils.data import DataLoader

from model.pHNN import ConverterPHNN
from utils.integrator import rk4_integrate, euler_integrate
from datasets.preprocess import TrajectorySliceDataset
from utils.utils import normalized_mse

# ==========================================
# 1. Define hyper-parameter search space
# ==========================================
class Config:
    def __init__(self, trial=None, fixed_args=None):

        if fixed_args:
            args_dict = vars(fixed_args) if isinstance(fixed_args, argparse.Namespace) else fixed_args
            self.__dict__.update(args_dict)

        if trial:
            self.seq_len = trial.suggest_categorical("seq_len", [800, 1000, 2000])

            # self.hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32])
            # self.layers = trial.suggest_int("layers", 1, 3)
            combo = trial.suggest_categorical("architecture_combo", ["8_3", "16_2", "16_3", "32_2"])
            hidden_dim_str, layers_str = combo.split("_")
            self.hidden_dim = int(hidden_dim_str)
            self.layers = int(layers_str)

            self.lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
            self.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            self.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

            self.warmup_steps = trial.suggest_int("warmup_steps", 5, 10)
            self.patience = trial.suggest_int("patience", 10, 20)

            self.loss_weight_v = trial.suggest_float("loss_weight_v", 0.1, 100.0, log=True)
            self.loss_weight_i = trial.suggest_float("loss_weight_i", 0.1, 100.0, log=True)

# ==========================================
# 2. Objective Function
# ==========================================
def objective(trial, base_args):

    args = Config(trial, base_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------
    # Data Loading
    # ---------------------------------------------------
    project_dir = os.getcwd()
    data_path = os.path.join(project_dir, 'datasets', 'processed', f'{args.converter_model}_trajectories.pt')
    meta_path = os.path.join(project_dir, 'datasets', 'processed', f'{args.converter_model}_meta.pt')

    if not os.path.exists(data_path):
        raise RuntimeError(f"Dataset not found at {data_path}. Please run preprocess.py first.")

    data = torch.load(data_path, weights_only=False)
    meta = torch.load(meta_path, weights_only=False)

    z_all = data["z"]  # (N_traj, T, 5)
    u_all = data["u_ext"]  # (N_traj, T, 5)
    sw_all = data["u_sw"]  # (N_traj, T, 6)
    dt = meta["dt"]
    N_traj = z_all.shape[0]

    n_train = int(N_traj * args.dataset_split)
    perm = torch.randperm(N_traj, generator=torch.Generator().manual_seed(args.seed))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = TrajectorySliceDataset(
        z_all[train_idx], u_all[train_idx], sw_all[train_idx],
        seq_len=args.seq_len,
        samples_per_epoch=args.samples_per_epoch,
    )
    val_ds = TrajectorySliceDataset(
        z_all[val_idx], u_all[val_idx], sw_all[val_idx],
        seq_len=args.seq_len,
        samples_per_epoch=args.samples_per_epoch // 5,
    )

    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ---------------------------------------------------
    # Model Initialization
    # ---------------------------------------------------
    model = ConverterPHNN(
        hidden_dim=args.hidden_dim,
        depth=args.layers,
        R=args.R_type,
        Pinv=args.Pinv_type,
        arch=args.arch,
    ).to(device)

    # ---------------------------------------------------
    # Optimizer & Scheduler
    # ---------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience, min_lr=1e-6
    )

    if args.loss_fn == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss_fn == "normalized_mse":
        criterion = normalized_mse
    else:
        raise ValueError(f"Unknown loss: {args.loss_fn}")

    global_step = 0

    # ==========================================
    # Training Loop
    # ==========================================

    if args.integrate_method == "rk4":
        integrate_fn = rk4_integrate
    elif args.integrate_method == "euler":
        integrate_fn = euler_integrate
    else:
        raise ValueError(f"Unknown integrate_method: {args.integrate_method}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for z_seg, u_seg, sw_seg in train_loader:
            z_seg = z_seg.to(device)
            u_seg = u_seg.to(device)
            sw_seg = sw_seg.to(device)

            z0 = z_seg[:, 0, :]
            z_target = z_seg[:, 1:, :]

            optimizer.zero_grad(set_to_none=True)

            z_pred = integrate_fn(model, z0, u_seg, sw_seg, dt, args.seq_len)

            loss_v = criterion(z_pred[:, :, 0:2], z_target[:, :, 0:2])
            loss_i = criterion(z_pred[:, :, 2:5], z_target[:, :, 2:5])

            loss = args.loss_weight_v * loss_v + args.loss_weight_i * loss_i

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step <= args.warmup_steps:
                lr_scale = global_step / args.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * lr_scale

        # ==========================================
        # Validation Loop
        # ==========================================
        if epoch % args.val_frequency == 0 or epoch == args.epochs:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for z_seg, u_seg, sw_seg in val_loader:
                    z_seg = z_seg.to(device)
                    u_seg = u_seg.to(device)
                    sw_seg = sw_seg.to(device)

                    z0 = z_seg[:, 0, :]
                    z_target = z_seg[:, 1:, :]

                    z_pred = integrate_fn(model, z0, u_seg, sw_seg, dt, args.seq_len)
                    val_loss_v = criterion(z_pred[:, :, 0:2], z_target[:, :, 0:2])
                    val_loss_i = criterion(z_pred[:, :, 2:5], z_target[:, :, 2:5])

                    val_loss_total = args.loss_weight_v * val_loss_v + args.loss_weight_i * val_loss_i

                    val_loss_sum += val_loss_total.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(1, val_batches)

            if global_step > args.warmup_steps:
                scheduler.step(avg_val_loss)

            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return avg_val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- Optuna ---
    parser.add_argument('--study_name', type=str, default='pH_NODE_Optimization')
    parser.add_argument('--n_trials', type=int, default=100, help='Search')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna.db')

    # --- Fixed ---
    parser.add_argument("--converter_model", type=str, default="IGBT")
    parser.add_argument("--dataset_split", type=float, default=0.8)
    parser.add_argument('--R_type', type=str, default="NonlinearDiag2")
    parser.add_argument('--Pinv_type', type=str, default="nominal")
    parser.add_argument('--arch', type=str, default="mlp")

    parser.add_argument("--samples_per_epoch", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--val_frequency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--integrate_method", type=str, default="euler", help="rk4/euler")

    opt_args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        storage=opt_args.storage,
        study_name=opt_args.study_name,
        load_if_exists=True
    )

    print(f"Start Optimization for {opt_args.study_name}...")

    try:
        study.optimize(lambda trial: objective(trial, opt_args), n_trials=opt_args.n_trials)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # Print results
    print("\n" + "=" * 30)
    print("Optimization Finished!")
    print("=" * 30)

    if len(study.trials) > 0:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (Val Loss Norm): {trial.value:.6f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        print("\nTo train with these params, update your train.py arguments.")
    else:
        print("No trials completed.")