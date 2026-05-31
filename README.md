# Port-Hamiltonian Neural Converter

<div align="center">

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![MATLAB](https://img.shields.io/badge/MATLAB-R2018b-red.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c.svg)

</div>

## Overview

<p align="center">
  <img src="docs/framework.png" alt="Physics-priori neural converter framework" width="560">
</p>

This repository contains the code, trained models, Simulink models, and validation data for a physics-priori neural converter based on port-Hamiltonian neural ordinary differential equations (pH-NODEs). The goal is to provide fast and physically meaningful electromagnetic transient (EMT) models for power-electronic-converter dominated systems.

The project combines:

- detailed Simulink EMT data generation for two-level and three-level converters;
- preprocessing utilities that convert raw `.mat` trajectories into PyTorch tensors;
- neural converter models including `pHNODE`, `NODE`, `PINODE`, and `LSTM`;
- trained checkpoints and MATLAB-exported weights for Simulink deployment;
- offline Simulink validation and OPAL-RT real-time validation assets.

Compared with pure black-box neural surrogates, the pH-NODE model keeps the known converter structure in the port-Hamiltonian formulation and learns the uncertain nonlinear terms from data. This design improves interpretability and helps maintain stable long-horizon rollouts.

## Repository Layout

```text
.
|-- docs/
|   `-- framework.png
|-- simulink/
|   |-- data_generation/
|   |   |-- level2/        # Two-level converter data-generation model and scripts
|   |   `-- level3/        # Three-level converter data-generation model and scripts
|   |-- offline/
|   |   |-- 2level/        # Offline comparison models and exported pH-NODE weights
|   |   `-- 3level/
|   `-- realtime/
|       |-- 2level/        # OPAL-RT real-time models and weights
|       `-- 3level/
|-- training/
|   |-- datasets/
|   |   |-- 2level/
|   |   |   |-- raw/       # Raw .mat data placeholder and Zenodo link
|   |   |   `-- processed/ # Processed tensor placeholder and Zenodo link
|   |   `-- 3level/
|   |       |-- raw/
|   |       `-- processed/
|   |-- model/             # pHNODE, NODE, PINODE, LSTM, MLP, and KAN modules
|   |-- utils/             # Integrators, plotting, and helper utilities
|   |-- checkpoints/       # Released PyTorch checkpoints and loss curves
|   |-- plots/             # Offline, online, and real-time plotting data/scripts
|   |-- results/           # Inference output directory
|   |-- main.py            # Main training entry point
|   |-- train.py           # Training loop
|   |-- train_with_optuna.py
|   |-- inference.py       # Long-horizon rollout and error reporting
|   `-- export_to_sim.py   # Export PyTorch checkpoints to MATLAB .mat weights
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Environment

The project is organized for MATLAB/Simulink plus Python training.

| Component | Notes |
|-----------|-------|
| MATLAB/Simulink | Developed with MATLAB/Simulink R2018b assets |
| Python | Python 3.10 recommended |
| PyTorch | Required for training and inference |
| OPAL-RT | Real-time validation models target OP4610XG workflows |

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` includes:

```text
numpy
scipy
torch
scikit-learn
matplotlib
pyinstrument
```

## Data

Large raw and processed datasets are not stored directly in Git. The dataset placeholder files point to Zenodo records:

| Dataset | Location in repository | Zenodo record |
|---------|------------------------|---------------|
| Raw `.mat` trajectories | `training/datasets/2level/raw/`, `training/datasets/3level/raw/` | https://zenodo.org/records/20422710 |
| Processed PyTorch tensors | `training/datasets/2level/processed/`, `training/datasets/3level/processed/` | https://zenodo.org/records/20463176 |

Download the data and place the files under the matching topology directory before training or inference. For example:

```text
training/datasets/2level/raw/sim_record_001.mat
training/datasets/2level/processed/2level_trajectories.pt
training/datasets/2level/processed/2level_meta.pt
```

## Workflow

### 1. Generate Raw Data in Simulink

The Simulink data-generation assets are separated by converter topology:

```text
simulink/data_generation/level2/
simulink/data_generation/level3/
```

Use the corresponding MATLAB script and model:

| Topology | MATLAB script | Simulink model |
|----------|---------------|----------------|
| Two-level converter | `Parsim_for_ai_2level.m` | `Parsim_for_2level.mdl` |
| Three-level converter | `Parsim_for_ai_3level.m` | `Parsim_for_3level.mdl` |

Generated raw trajectories should be saved as `.mat` files under:

```text
training/datasets/2level/raw/
training/datasets/3level/raw/
```

### 2. Preprocess Data

From the `training/` directory, convert raw MATLAB trajectories into PyTorch tensors:

```bash
cd training
python datasets/preprocess.py --converter_model 2level --normalization 1
```

For the three-level converter:

```bash
python datasets/preprocess.py --converter_model 3level --normalization 1
```

Useful options:

| Option | Meaning |
|--------|---------|
| `--converter_model` | `2level` or `3level` |
| `--normalization` | `1` to save normalized tensors, `0` to save raw-scale tensors |
| `--downsample` | Downsampling factor |
| `--file_name` | `all` or a single `.mat` file name |
| `--data_dir` | Optional custom raw data directory |
| `--save_dir` | Optional custom processed data directory |

### 3. Train Neural Converter Models

The main training entry point is:

```bash
cd training
python main.py
```

`main.py` exposes command-line options for topology, model family, network size, ODE integration, loss weighting, and training settings. The supported neural model names are:

```text
pHNODE
NODE
PINODE
LSTM
```

Example pH-NODE training command:

```bash
python main.py --converter_model 2level --converter_neural_model pHNODE \
  --hidden_dim 8 --layers 3 --integrate_method euler \
  --normalization 1 --epochs 50 --batch_size 512
```

Example black-box NODE command:

```bash
python main.py --converter_model 2level --converter_neural_model NODE \
  --hidden_dim 128 --layers 4 --integrate_method euler \
  --normalization 1 --epochs 200 --batch_size 2048
```

Training saves checkpoints and loss curves to:

```text
training/checkpoints/
```

Checkpoint names follow:

```text
<converter_model>_<converter_neural_model>.pt
```

The repository includes released checkpoints for:

```text
2level_pHNODE.pt
2level_NODE.pt
2level_PINODE.pt
2level_LSTM.pt
3level_pHNODE.pt
```

### 4. Run Inference

Inference is implemented in `training/inference.py` and is invoked by `main.py` after training in the current workflow. It loads a raw `.mat` trajectory, the matching processed metadata, and a checkpoint from `training/checkpoints/`, then writes rollout plots under:

```text
training/results/<converter_model>/
```

The default inference file is:

```text
sim_record_001.mat
```

Use `--infer_file`, `--Ts`, and `--chunk_size` to change the rollout configuration.

### 5. Export Weights to Simulink

Use `export_to_sim.py` to convert a PyTorch checkpoint into MATLAB `.mat` weights:

```bash
cd training
python export_to_sim.py --converter_model 2level --converter_neural_model pHNODE
```

The script looks for:

```text
training/checkpoints/<converter_model>_<converter_neural_model>.pt
```

and exports MATLAB weight files that can be used by the Simulink models. Pre-exported weights are already included in the Simulink folders:

```text
simulink/offline/2level/2level_pHNODE_weights.mat
simulink/offline/3level/3level_pHNODE_weights.mat
simulink/realtime/2level/2level_pHNODE_weights.mat
simulink/realtime/3level/3level_pHNODE_weights.mat
simulink/realtime/scale/2level_pHNODE_weights.mat
```

### 6. Offline Validation in Simulink

Offline Simulink comparison models are stored under:

```text
simulink/offline/2level/
simulink/offline/3level/
```

Key files include:

| File | Purpose |
|------|---------|
| `Compare_2level.mdl`, `AI_Compare_2level.mdl` | Two-level offline comparison models |
| `Compare_3level.mdl` | Three-level offline comparison model |
| `Net_improve_init.m` | Converter and simulation initialization |
| `Export_result_plot.m` | Export Simulink results for Python/MATLAB plotting |
| `Compare_error.m` | Compute error metrics |

### 7. Real-Time Validation

Real-time Simulink models and OPAL-RT-oriented assets are organized by topology:

```text
simulink/realtime/2level/
simulink/realtime/3level/
```

The two-level real-time folder includes detailed IGBT/Diode, switching-function, and AI model variants:

```text
RT_IGBT.slx
RT_SWF.slx
RT_AI.slx
```

The scalability folder contains additional real-time models for larger-scale comparison workflows.

## Results and Plotting

Validation data and plotting scripts are under `training/plots/`:

```text
training/plots/open_loop_2level/
training/plots/close_loop_2level/
training/plots/closed_loop_3level/
training/plots/real_time_2level/
```

Plotting utilities include:

| Script | Purpose |
|--------|---------|
| `offline_plot_result.py` | Offline validation figures |
| `online_plot_result.py` | Online/real-time validation figures |
| `testai_plot_result.py` | AI rollout plotting |
| `computa_cost_result.py` | Computational-cost result analysis |

## Model Summary

| Model | File | Role |
|-------|------|------|
| `pHNODE` | `training/model/pHNODE.py` | Proposed port-Hamiltonian neural ODE converter |
| `NODE` | `training/model/NODE.py` | Black-box neural ODE baseline |
| `PINODE` | `training/model/PINODE.py` | Physics-informed NODE baseline |
| `LSTM` | `training/model/LSTM.py` | Sequence-model baseline |
| `pHNN` | `training/model/pHNN.py` | Port-Hamiltonian neural network component |

## Notes

- Keep large generated raw data outside Git or download it from Zenodo into the dataset folders.
- Run Python commands from `training/` unless a script explicitly documents another working directory.
- Ensure that the selected `--converter_model` matches the dataset, checkpoint, and Simulink weight file being used.
- For normalized training or inference, the matching `*_norm_meta.pt` file must be present in the processed dataset directory.

## Citation

If you find this repository useful, please cite the related paper:

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
