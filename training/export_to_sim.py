#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Utilities for the port-Hamiltonian neural converter project."""

import torch
import scipy.io as sio
from model.pHNN import ConverterPHNN

model = ConverterPHNN(state_dim=5, sw_dim=6, hidden_dim=8, depth=3, R="NonlinearDiag2", Pinv="nominal", arch="mlp")
model.load_state_dict(torch.load('checkpoints/IGBT_ConverterPHNN_RNonlinearDiag2_Pinvnominal_archmlp.pt', map_location='cpu'))
model.eval()

weights_dict = {}
for name, param in model.named_parameters():
    weights_dict[name.replace('.', '_')] = param.detach().numpy()

sio.savemat('phnode_weights.mat', weights_dict)
print("Saved phnode_weights.mat")