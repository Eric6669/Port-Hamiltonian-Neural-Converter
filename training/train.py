#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：train.py
@Author  ：He Xing
@Date    ：2026/4/2 21:18 
"""

import os
import math
import time
import torch
import numpy as np

from model.pHNODE import pHNODE
from model.NODE import NODE
from model.PINODE import PINODE
from model.LSTM import LSTM

from utils.integrator import rk4_integrate, euler_integrate
from datasets.preprocess import TrajectorySliceDataset
from utils.utils import plot_loss_curves
from torch.utils.data import DataLoader, random_split


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
            physics_weight=args.physics_weight,
        )

    elif args.converter_neural_model == "LSTM":
        return LSTM(
            converter_model=args.converter_model,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            dropout=args.lstm_dropout,
        )

    else:
        raise ValueError(f"Unknown model: {args.converter_neural_model}")


def get_data_tag(args):
    if getattr(args, "normalization", 0) == 1:
        return f"{args.converter_model}_norm"
    return args.converter_model


def compute_state_loss(model, z_pred, z_target, args, criterion):
    if args.converter_neural_model == "PINODE":
        loss, loss_dict = model.total_loss(
            z_pred=z_pred,
            z_target=z_target,
            loss_weight_v=args.loss_weight_v,
            loss_weight_i=args.loss_weight_i,
            physics_weight=args.physics_weight,
        )
        return loss

    loss_v = criterion(z_pred[:, :, 0:2], z_target[:, :, 0:2])
    loss_i = criterion(z_pred[:, :, 2:5], z_target[:, :, 2:5])
    loss = args.loss_weight_v * loss_v + args.loss_weight_i * loss_i
    return loss

def train(args):
    # ---------------------------------------------------
    # 1. Environment & Config
    # ---------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------
    project_dir = os.getcwd()
    data_tag = get_data_tag(args)

    data_path = os.path.join(
        project_dir, 'datasets', f'{args.converter_model}', 'processed',
        f'{data_tag}_trajectories.pt'
    )
    meta_path = os.path.join(
        project_dir, 'datasets', f'{args.converter_model}', 'processed',
        f'{data_tag}_meta.pt'
    )

    print(f"Loading: {data_path}")
    data = torch.load(data_path, weights_only=False)
    meta = torch.load(meta_path, weights_only=False)

    z_all = data["z"]  # (N_traj, T, 5)
    u_all = data["u_ext"]  # (N_traj, T, 5)
    sw_all = data["u_sw"]  # (N_traj, T, 6)

    dt = meta["dt"]
    N_traj, T, _ = z_all.shape
    print(f"  {N_traj} trajectories × {T} steps, dt={dt:.1e}s")

    n_train = int(N_traj * args.dataset_split)
    n_val = N_traj - n_train

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

    # num_workers = min([os.cpu_count(), 8])
    num_workers = 0

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------
    model = build_model(args).to(device)

    # ---------------------------------------------------
    # 4. Optimizer & Scheduler
    # ---------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    warmup_steps = args.warmup_steps
    global_step = 0

    def get_warmup_factor():
        if global_step < warmup_steps:
            return global_step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=args.patience, min_lr=1e-6
    )

    # ---------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------

    if args.integrate_method == "rk4":
        integrate_fn = rk4_integrate
    elif args.integrate_method == "euler":
        integrate_fn = euler_integrate
    else:
        raise ValueError(f"Unknown integrate_method: {args.integrate_method}")

    if args.loss_fn == "mse":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss_fn}")

    save_dir = os.path.join(project_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"{args.converter_model}_{args.converter_neural_model}.pt"
    )

    best_val_loss = float('inf')
    t_start = time.time()

    train_loss_history = []
    val_loss_history = []
    val_epochs = []

    global_step = 0

    print(f"\nTraining {args.epochs} epochs | L={args.seq_len} | dt={dt:.1e}s")
    print("-" * 100)

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for z_seg, u_seg, sw_seg in train_loader:
            z_seg = z_seg.to(device)  # (B, L+1, 5)
            u_seg = u_seg.to(device)  # (B, L+1, 5)
            sw_seg = sw_seg.to(device)  # (B, L+1, 6)

            optimizer.zero_grad(set_to_none=True)

            if args.converter_neural_model in ODE_MODEL_NAMES:
                z0 = z_seg[:, 0, :]
                z_target = z_seg[:, 1:, :]

                z_pred = integrate_fn(
                    f_theta=model,
                    z0=z0,
                    u_seq=u_seg,
                    sw_seq=sw_seg,
                    dt=dt,
                    steps=args.seq_len,
                )

            elif args.converter_neural_model in SEQ_MODEL_NAMES:
                # LSTM teacher forcing:
                # [z, u_ext, u_sw]_{0:L-1} -> z_{1:L}
                z_input = z_seg[:, :-1, :]
                u_input = u_seg[:, :-1, :]
                sw_input = sw_seg[:, :-1, :]
                z_target = z_seg[:, 1:, :]

                z_pred, _ = model(z_input, u_input, sw_input)

            else:
                raise ValueError(f"Unknown model: {args.converter_neural_model}")

            loss = compute_state_loss(model, z_pred, z_target, args, criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            global_step += 1
            if global_step <= args.warmup_steps:
                lr_scale = global_step / args.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * lr_scale

            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / max(1, n_batches)
        train_loss_history.append(avg_train)

        # ---- validate ----
        if epoch % args.val_frequency == 0 or epoch == args.epochs:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for z_seg, u_seg, sw_seg in val_loader:
                    z_seg = z_seg.to(device)
                    u_seg = u_seg.to(device)
                    sw_seg = sw_seg.to(device)

                    if args.converter_neural_model in ODE_MODEL_NAMES:
                        z0 = z_seg[:, 0, :]
                        z_target = z_seg[:, 1:, :]

                        z_pred = integrate_fn(
                            model, z0, u_seg, sw_seg, dt, args.seq_len
                        )

                    elif args.converter_neural_model in SEQ_MODEL_NAMES:
                        z_input = z_seg[:, :-1, :]
                        u_input = u_seg[:, :-1, :]
                        sw_input = sw_seg[:, :-1, :]
                        z_target = z_seg[:, 1:, :]

                        z_pred, _ = model(z_input, u_input, sw_input)

                    else:
                        raise ValueError(f"Unknown model: {args.converter_neural_model}")

                    val_loss_total = compute_state_loss(
                        model, z_pred, z_target, args, criterion
                    )

                    val_loss_sum += val_loss_total.item()
                    val_batches += 1

            avg_val = val_loss_sum / max(1, val_batches)

            if global_step > args.warmup_steps:
                scheduler.step(avg_val)

            val_loss_history.append(avg_val)
            val_epochs.append(epoch)

            elapsed = time.time() - t_start
            lr_now = optimizer.param_groups[0]['lr']

            log = (f"Epoch {epoch:>3d}/{args.epochs} | "
                   f"Train loss: {avg_train:.4e} | Val loss: {avg_val:.4e} | "
                   f"LR: {lr_now:.2e} | Time: {elapsed:.0f}s")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), save_path)
                log += "  [★ Best]"

            print(log)

    # ---------------------------------------------------
    # 6. Load Best & Return
    # ---------------------------------------------------
    print("-" * 100)
    if best_val_loss < float('inf'):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"Loaded best model (Val MSE = {best_val_loss:.4e}) from {save_path}")

    plot_loss_curves(
        train_losses=train_loss_history,
        val_losses=val_loss_history,
        val_epochs=val_epochs,
        save_dir=save_dir,
        filename=f"{args.converter_model}_{args.converter_neural_model}_loss.png"
    )

    model.cpu()
    return model













