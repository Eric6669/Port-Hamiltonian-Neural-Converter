# Port-Hamiltonian Neural Converter

<div align="center">

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![MATLAB](https://img.shields.io/badge/MATLAB-R2018b-red.svg)
![Python](https://img.shields.io/badge/Python-3.10.19-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900.svg)

</div>

## Overview

<p align="center">
  <img src="https://img.shields.io/badge/EMT_Simulation-✓-blue" alt="EMT Simulation">
  <img src="https://img.shields.io/badge/Port_Hamiltonian_Modeling-✓-orange" alt="Port-Hamiltonian Modeling">
  <img src="https://img.shields.io/badge/Neural_Converter-✓-critical" alt="Neural Converter">
</p>

<p align="center">
  <img src="docs/framework.png" alt="Physics-priori neural converter framework" width="500">
</p>

Power-electronic-converter dominated grids require electromagnetic transient (EMT) models that are simultaneously fast, accurate, and stable over long rollouts. Conventional detailed IGBT/Diode models provide high fidelity but are expensive for real-time simulation, while switching-function models improve speed at the cost of neglected nonlinear and dissipative dynamics. Pure black-box neural surrogates can learn from data, but they do not preserve converter structure and may accumulate non-physical errors.

This repository provides the implementation of a **physics-priori neural converter** based on **port-Hamiltonian neural ordinary differential equations (pH-NODEs)**. The model preserves the known switching-induced interconnection matrix and learns only uncertain dissipation and parameter mismatch terms. This design improves physical interpretability, enforces positive dissipation through structural parameterization, and supports high-fidelity offline and real-time EMT simulation.

**Our Contribution:**

- **A physics-priori Port-Hamiltonian neural converter**.
- **A passivity-based mechanism** supports bounded long-horizon EMT rollout.
- **Offline and online validation** against detailed IGBT/Diode and switching-function models.
- **Reproducible code and models** for converter models:

| Label | Model | Role                  |
|-------|-------|-----------------------|
| **M1** | IGBT/Diode detailed model | Ground truth          |
| **M2** | Switching-function model | Real-time comparision |
| **M3** | pH-NODE neural converter (ours) | Proposed              |

## Environment

### Software Environment

| Component | Version / Configuration |
|-----------|------------------------|
| MATLAB/Simulink | R2018b |
| Python | 3.10.19 |
| PyTorch | 2.9.1 |
| CUDA | 12.8 |
| Training GPU | NVIDIA GeForce RTX 5090 |
| Real-time simulator | OPAL-RT OP4610XG (AMD Ryzen 3.8 GHz) |

## Repository Layout

```text
├── simulink/
│   ├── data_generation/        # Parallel Simulink data-generation model and script
│   ├── offline/                # Offline comparison models
│   └── realtime/               # RT real-time models for OP4610XG
│
├── training/
│   ├── datasets/
│   │   ├── raw/                # Raw .mat files generated from Simulink (large, not fully uploaded)
│   │   ├── processed/          # Preprocessed tensors and metadata
│   │   └── preprocess.py       # Raw-to-tensor preprocessing script
│   ├── model/                  # pH-NODE and black-box-NODE model definitions
│   ├── utils/                  # ODE integrators, losses, and plotting helpers
│   ├── checkpoints/            # Trained PyTorch weights
│   ├── plots/                  # Paper figures and exported result data
│   ├── results/                # Inference output directory
│   ├── main.py                 # Single entry point for training and inference
│   ├── train.py                # Supervised rollout training loop
│   ├── inference.py            # Long-horizon rollout test and error reporting
│   └── export_to_sim.py        # Export PyTorch weights to MATLAB .mat format
│
├── requirements.txt
└── README.md
```

## Workflow

The full pipeline from data generation to real-time validation is illustrated below. Each step is detailed in the sections that follow.

```text
 ┌──────────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
 │  Data Generation │─────│ Preprocessing │─────│   Training   │─────│  Inference  │
 │    (Simulink)    │     │   (Python)    │     │   (Python)   │     │   (Python)  │
 └──────────────────┘     └───────────────┘     └──────────────┘     └──────┬──────┘
                                                                            │
                                              ┌─────────────────────────────┘
                                              ▼
                                   ┌────────────────────┐
                                   │ Export to Simulink │
                                   │  (export_to_sim)   │
                                   └─────────┬──────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                   ┌────────────────────┐         ┌───────────────────┐
                   │ Offline Validation │         │  RT / Real-Time  │
                   │    (Simulink)      │         │    (OPAL-RT)      │
                   └────────────────────┘         └───────────────────┘
```

### Step 1 — Data Generation (Simulink)

Generate broadband transient trajectories by injecting random sinusoidal perturbations into operating-condition references (active power, reactive power, dc voltage, ac voltage). The dataset used in the paper consists of 200 independent 3-second simulations at a 20 μs step size.

**Files:**
- `simulink/data_generation/Parim_for_ai.mdl` — Simulink model for parallel trajectory generation.
- `simulink/data_generation/Parsim_for_ai.m` — MATLAB script that randomizes conditions, runs parallel simulations, and saves raw `.mat` records.

Run the `Parsim_for_ai.m` and place the generated raw data under `training/datasets/raw/`.

### Step 2 — Preprocessing

Convert raw `.mat` trajectories into normalized PyTorch tensors:

```bash
cd training
python datasets/preprocess.py --converter_model IGBT --normalization 0
```

The released processed data are based on the **IGBT/Diode detailed model (M1)**. Use `--converter_model Switching` only if you intentionally want to build a switching-function surrogate model.

### Step 3 — Train and Inference

Install dependencies and launch training:

```bash
pip install -r requirements.txt

cd training
python main.py converter_model IGBT converter_neural_model ConverterPHNN \
    --R_type NonlinearDiag2 --Pinv_type nominal --arch mlp --integrate_method euler
```

Key hyperparameters (from the paper):

| Hyperparameter | Value |
|----------------|-------|
| MLP architecture | [11, 8, 8, 8, 5] |
| ODE solver | Euler |
| Optimizer | AdamW |
| Loss | MSE|
| Batch size | 512 |
| Learning rate | 3.33e-3 |
| LR decay factor | 0.5 |
| Weight decay | 9.83e-4 |
| Warmup steps | 9 |
| Epochs | 30 |

The trained checkpoint is saved to:

```text
training/checkpoints/IGBT_ConverterPHNN_RNonlinearDiag2_Pinvnominal_archmlp.pt
```

### Step 4 — Export to Simulink

Convert the trained PyTorch weights into a MATLAB `.mat` file for use in Simulink:

```bash
cd training
python export_to_sim.py
```

Generated pre-exported weight files are already included:

```text
simulink/offline/phnode_weights_3_8.mat
simulink/realtime/phnode_weights_3_8.mat
```

### Step 5 — Offline Validation (Simulink)

Compare M1, M2, and M3 under open-loop and closed-loop scenarios inside Simulink.

**Files:**
- `simulink/offline/Compare_AI_kk.mdl` — comparison model for M1, M2, and M3.
- `simulink/offline/Net_improve_init.m` — converter and simulation parameter initialization.
- `simulink/offline/Export_result_plot.m` — exports Simulink variables to `.mat` for plotting.
- `simulink/offline/Compare_error.m` — computes RMSE, MAE, NRMSE, and relative RMSE.

**Procedure:**

1. Open `Compare_AI_kk.mdl` and run the desired scenario.
2. Run `Export_result_plot.m` to save `Y_IGBT.mat`, `Y_pred.mat`, `Y_SWF.mat`.
3. Run `Compare_error.m` to compute error metrics against M1.

### Step 6 — Real-Time Validation (OPAL-RT)

Deploy M1, M2, and M3 on the OP4610XG real-time simulator for hardware-in-the-loop testing.

**Files:**
- `simulink/realtime/RT_IGBT.slx` — M1 detailed IGBT/Diode model.
- `simulink/realtime/RT_SWF.slx` — M2 switching-function model.
- `simulink/realtime/RT_AI.slx` — M3 pH-NODE neural converter.
- `simulink/realtime/Net_improve_init.m` and `phnode_weights_3_8.mat` — initialization and weights.

## Performance Validation Results

The repository includes exported result data and figures for the paper's three validation cases:

| Case | Directory |
|------|-----------|
| Open-loop (ac sag + dc step) | `training/plots/open_1.0s_ac_1.0_0.8_2.0s_dc_3000_2800/` |
| Closed-loop (power step + dc step) | `training/plots/close_1.0s_p_0.8_0.6_2.0s_dc_3000_2800/` |
| RT (dc step + power step) | `training/plots/RT_1.0s_dc_3000_3200_2.0s_p_0.8_0.6/` |

Each directory contains the exported `.mat` files and SVG comparison figures. The RT directory additionally includes raw OPAL-RT exports and `from_RT_to_py.m` for format conversion.

## Citation

If you find this work useful, please cite:

```bibtex
@article{
  title   = {Physics-Priori Neural Converters Modelling for Intelligent EMT Simulation},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
