#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：main.py
@Author  ：He Xing
@Date    ：2026/4/2 10:06 
"""

import torch
import argparse
import numpy as np
from inference import Inference
from pyinstrument import Profiler
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ————————————————————————
    #      2level pHNODE
    # ————————————————————————
    # # --- data ---
    # parser.add_argument("--converter_model", type=str, default="2level", help="2level, 3level")
    # parser.add_argument("--dataset_split", type=float, default=0.8)
    # parser.add_argument("--normalization", type=int, default=0, help="0: use raw dataset; 1: use normalized dataset with *_norm_*.pt")
    #
    # # --- model ---
    # parser.add_argument("--converter_neural_model", type=str, default="pHNODE", choices=["pHNODE", "NODE", "PINODE", "LSTM"])
    # parser.add_argument('--hidden_dim', type=int, default=8)
    # parser.add_argument('--layers', type=int, default=3)
    #
    # # --- ODE ---
    # parser.add_argument("--integrate_method", type=str, default="euler", help="rk4/euler")
    # parser.add_argument("--seq_len", type=int, default=1000, help="integrated steps L")
    # parser.add_argument("--samples_per_epoch", type=int, default=10000)
    #
    # # --- training ---
    # parser.add_argument('--lr', type=float, default=0.003329440205897641)
    # parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--weight_decay", type=float, default=0.0009831502750510465)
    # parser.add_argument("--patience", type=int, default=14)
    # parser.add_argument("--warmup_steps", type=int, default=9)
    # parser.add_argument("--batch_size", type=int, default=512)
    # parser.add_argument("--grad_clip", type=float, default=1.0)
    # parser.add_argument("--loss_weight_v", type=float, default=2.0411662248723124)
    # parser.add_argument("--loss_weight_i", type=float, default=1.0076270164917709)
    # parser.add_argument("--loss_fn", type=str, default="mse", help="mse | normalized_mse")
    # parser.add_argument("--val_frequency", type=int, default=1)
    # parser.add_argument("--seed", type=int, default=77)
    #
    # # --- inference ---
    # parser.add_argument("--infer_file", type=str, default="sim_record_001.mat")
    # parser.add_argument("--Ts", type=float, default=20e-6)
    # parser.add_argument("--chunk_size", type=int, default=1000)

    # ————————————————————————
    #      3level pHNODE
    # ————————————————————————
    # # --- data ---
    # parser.add_argument("--converter_model", type=str, default="3level", help="2level, 3level")
    # parser.add_argument("--dataset_split", type=float, default=0.8)
    # parser.add_argument("--normalization", type=int, default=0, help="0: use raw dataset; 1: use normalized dataset with *_norm_*.pt")
    #
    # # --- model ---
    # parser.add_argument("--converter_neural_model", type=str, default="pHNODE", choices=["pHNODE", "NODE", "PINODE", "LSTM"])
    # parser.add_argument('--hidden_dim', type=int, default=16)
    # parser.add_argument('--layers', type=int, default=2)
    #
    # # --- ODE ---
    # parser.add_argument("--integrate_method", type=str, default="euler", help="rk4/euler")
    # parser.add_argument("--seq_len", type=int, default=1000, help="integrated steps L")
    # parser.add_argument("--samples_per_epoch", type=int, default=10000)
    #
    # # --- training ---
    # parser.add_argument('--lr', type=float, default=0.003628537550438353)
    # parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--weight_decay", type=float, default=1.0299300313519892e-05)
    # parser.add_argument("--patience", type=int, default=14)
    # parser.add_argument("--warmup_steps", type=int, default=7)
    # parser.add_argument("--batch_size", type=int, default=512)
    # parser.add_argument("--grad_clip", type=float, default=1.0)
    # parser.add_argument("--loss_weight_v", type=float, default=0.9786656828093553)
    # parser.add_argument("--loss_weight_i", type=float, default=8.465940924223974)
    # parser.add_argument("--loss_fn", type=str, default="mse", help="mse | normalized_mse")
    # parser.add_argument("--val_frequency", type=int, default=1)
    # parser.add_argument("--seed", type=int, default=77)
    #
    # # --- inference ---
    # parser.add_argument("--infer_file", type=str, default="sim_record_001.mat")
    # parser.add_argument("--Ts", type=float, default=20e-6)
    # parser.add_argument("--chunk_size", type=int, default=1000)

    # ————————————————————————
    #      2level NODE
    # ————————————————————————
    # --- data ---
    parser.add_argument("--converter_model", type=str, default="2level", help="2level, 3level")
    parser.add_argument("--dataset_split", type=float, default=0.8)
    parser.add_argument("--normalization", type=int, default=1,
                        help="0: use raw dataset; 1: use normalized dataset with *_norm_*.pt")

    # --- model ---
    parser.add_argument("--converter_neural_model", type=str, default="NODE",
                        choices=["pHNODE", "NODE", "PINODE", "LSTM"])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)

    # --- ODE ---
    parser.add_argument("--integrate_method", type=str, default="euler", help="rk4/euler")
    parser.add_argument("--seq_len", type=int, default=1000, help="integrated steps L")
    parser.add_argument("--samples_per_epoch", type=int, default=10000)

    # --- training ---
    parser.add_argument('--lr', type=float, default=0.003302785504854589)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1.5463444007916514e-05)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--warmup_steps", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_weight_v", type=float, default=0.21809510907666285)
    parser.add_argument("--loss_weight_i", type=float, default=0.19267785057596976)
    parser.add_argument("--loss_fn", type=str, default="mse", help="mse | normalized_mse")
    parser.add_argument("--val_frequency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=77)

    # --- inference ---
    parser.add_argument("--infer_file", type=str, default="sim_record_001.mat")
    parser.add_argument("--Ts", type=float, default=20e-6)
    parser.add_argument("--chunk_size", type=int, default=1000)

    # ————————————————————————
    #      2level PINODE
    # ————————————————————————
    # # --- data ---
    # parser.add_argument("--converter_model", type=str, default="2level", help="2level, 3level")
    # parser.add_argument("--dataset_split", type=float, default=0.8)
    # parser.add_argument("--normalization", type=int, default=1,
    #                     help="0: use raw dataset; 1: use normalized dataset with *_norm_*.pt")
    #
    # # --- model ---
    # parser.add_argument("--converter_neural_model", type=str, default="PINODE",
    #                     choices=["pHNODE", "NODE", "PINODE", "LSTM"])
    # parser.add_argument('--hidden_dim', type=int, default=512)
    # parser.add_argument('--layers', type=int, default=3)
    #
    # # --- ODE ---
    # parser.add_argument("--integrate_method", type=str, default="euler", help="rk4/euler")
    # parser.add_argument("--seq_len", type=int, default=800, help="integrated steps L")
    # parser.add_argument("--samples_per_epoch", type=int, default=10000)
    #
    # # --- training ---
    # parser.add_argument('--lr', type=float, default=0.001657108056630809)
    # parser.add_argument("--epochs", type=int, default=400)
    # parser.add_argument("--weight_decay", type=float, default=8.54100027016024e-05)
    # parser.add_argument("--patience", type=int, default=16)
    # parser.add_argument("--warmup_steps", type=int, default=7)
    # parser.add_argument("--batch_size", type=int, default=1024)
    # parser.add_argument("--grad_clip", type=float, default=1.0)
    # parser.add_argument("--loss_weight_v", type=float, default=1.851169311286726)
    # parser.add_argument("--loss_weight_i", type=float, default=0.12338122396498605)
    # parser.add_argument("--loss_fn", type=str, default="mse", help="mse | normalized_mse")
    # parser.add_argument("--val_frequency", type=int, default=1)
    # parser.add_argument("--seed", type=int, default=77)
    #
    # # --- inference ---
    # parser.add_argument("--infer_file", type=str, default="sim_record_001.mat")
    # parser.add_argument("--Ts", type=float, default=20e-6)
    # parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--physics_weight", type=float, default=0.5)

    # ————————————————————————
    #      2level LSTM
    # ————————————————————————
    # # --- data ---
    # parser.add_argument("--converter_model", type=str, default="2level", help="2level, 3level")
    # parser.add_argument("--dataset_split", type=float, default=0.8)
    # parser.add_argument("--normalization", type=int, default=1,
    #                     help="0: use raw dataset; 1: use normalized dataset with *_norm_*.pt")
    #
    # # --- model ---
    # parser.add_argument("--converter_neural_model", type=str, default="LSTM",
    #                     choices=["pHNODE", "NODE", "PINODE", "LSTM"])
    # parser.add_argument('--hidden_dim', type=int, default=128)
    # parser.add_argument('--layers', type=int, default=2)
    #
    # # --- ODE ---
    # parser.add_argument("--integrate_method", type=str, default="euler", help="rk4/euler")
    # parser.add_argument("--seq_len", type=int, default=2000, help="integrated steps L")
    # parser.add_argument("--samples_per_epoch", type=int, default=10000)
    #
    # # --- training ---
    # parser.add_argument('--lr', type=float, default=0.001105632873276508)
    # parser.add_argument("--epochs", type=int, default=10000)
    # parser.add_argument("--weight_decay", type=float, default=0.0001315949317811959)
    # parser.add_argument("--patience", type=int, default=13)
    # parser.add_argument("--warmup_steps", type=int, default=9)
    # parser.add_argument("--batch_size", type=int, default=512)
    # parser.add_argument("--grad_clip", type=float, default=1.0)
    # parser.add_argument("--loss_weight_v", type=float, default=3.2121167002795112)
    # parser.add_argument("--loss_weight_i", type=float, default=0.12874207481438385)
    # parser.add_argument("--loss_fn", type=str, default="mse", help="mse | normalized_mse")
    # parser.add_argument("--val_frequency", type=int, default=1)
    # parser.add_argument("--seed", type=int, default=77)
    #
    # # --- inference ---
    # parser.add_argument("--infer_file", type=str, default="sim_record_001.mat")
    # parser.add_argument("--Ts", type=float, default=20e-6)
    # parser.add_argument("--chunk_size", type=int, default=1000)
    # parser.add_argument("--lstm_dropout", type=float, default=0.0)

    arguments = parser.parse_args()

    model = train(arguments)
    #
    # dummy_z = torch.zeros(1, 5)
    # dummy_u_sw_6 = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    # dummy_u_sw_12 = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    #                                1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    #
    # with torch.no_grad():
    #     J_tensor, R_tensor, Pinv_tensor = model.get_matrices(dummy_z, dummy_u_sw_12)
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

    # profiler = Profiler()
    # profiler.start()
    #
    # inferencer = Inference(arguments)
    # inferencer.run()
    #
    # profiler.stop()
    # profiler.print()