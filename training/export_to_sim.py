#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：export_to_sim.py
@Author  ：He Xing
@Date    ：2026/4/10 21:28 
"""

import os
import argparse
import torch
import scipy.io as sio
from model.pHNODE import pHNODE
from model.NODE import NODE
from model.PINODE import PINODE
from model.LSTM import LSTM

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

def export_model_to_mat(args):

    model = build_model(args)

    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"{args.converter_model}_{args.converter_neural_model}.pt"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    weights_dict = {}

    for name, param in model.named_parameters():
        weights_dict[name.replace(".", "_")] = param.detach().cpu().numpy()

    for name, buffer in model.named_buffers():
        weights_dict[name.replace(".", "_")] = buffer.detach().cpu().numpy()

    os.makedirs(args.save_dir, exist_ok=True)

    save_path = f"{args.converter_model}_{args.converter_neural_model}_weights.mat"

    sio.savemat(save_path, weights_dict)

    print(f"[SAVE] {save_path}")
    print("[INFO] Exported parameters:")
    for key, value in weights_dict.items():
        print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--converter_model",
        type=str,
        default="2level",
        choices=["2level", "3level"],
        help="Converter topology: 2level or 3level.",
    )

    parser.add_argument(
        "--converter_neural_model",
        type=str,
        default="LSTM",
        choices=["pHNODE", "NODE", "PINODE", "LSTM"],
        help="Neural model name used in checkpoint filename.",
    )

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--physics_weight", type=float, default=0.5)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Root directory of checkpoints.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="sim_weights",
        help="Directory to save exported .mat file.",
    )

    args = parser.parse_args()

    export_model_to_mat(args)
