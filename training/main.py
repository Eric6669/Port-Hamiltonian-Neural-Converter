#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""

import torch
import argparse
import numpy as np
from inference import Inference
from pyinstrument import Profiler
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- data ---
    parser.add_argument("--converter_model", type=str, default="IGBT", help="Switching, IGBT")
    parser.add_argument("--dataset_split", type=float, default=0.8)

    # --- model ---
    parser.add_argument("--converter_neural_model", type=str, default="ConverterPHNN", help="ConverterPHNN, ConverterNODE")
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--R_type', type=str, default="NonlinearDiag2")
    parser.add_argument('--Pinv_type', type=str, default="nominal")
    parser.add_argument('--arch', type=str, default="mlp")

    # --- ODE ---
    parser.add_argument("--integrate_method", type=str, default="euler", help="rk4/euler")
    parser.add_argument("--seq_len", type=int, default=1000, help="integrated steps L")
    parser.add_argument("--samples_per_epoch", type=int, default=10000)

    # --- training ---
    parser.add_argument('--lr', type=float, default=0.003329440205897641)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.0009831502750510465)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--warmup_steps", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_weight_v", type=float, default=2.0411662248723124)
    parser.add_argument("--loss_weight_i", type=float, default=1.0076270164917709)
    parser.add_argument("--loss_fn", type=str, default="mse", help="mse | normalized_mse")
    parser.add_argument("--val_frequency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=77)

    # --- inference ---
    parser.add_argument("--infer_file", type=str, default="sim_record_001.mat")
    parser.add_argument("--Ts", type=float, default=20e-6)
    parser.add_argument("--chunk_size", type=int, default=1000)

    arguments = parser.parse_args()

    model = train(arguments)
    #
    # dummy_z = torch.zeros(1, 5)
    # dummy_u_sw = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    #
    # with torch.no_grad():
    #     J_tensor, R_tensor, Pinv_tensor = model.get_matrices(dummy_z, dummy_u_sw)
    #
    #     J_matrix = J_tensor.squeeze(0).numpy()
    #     R_matrix = R_tensor.squeeze(0).numpy()
    #     Pinv_matrix = Pinv_tensor.squeeze(0).numpy()
    #
    # np.set_printoptions(precision=4, suppress=True, linewidth=120)
    #
    # print(J_matrix)
    # print(R_matrix)
    # print(Pinv_matrix)

    profiler = Profiler()
    profiler.start()

    inferencer = Inference(arguments)
    inferencer.run()

    profiler.stop()
    profiler.print()