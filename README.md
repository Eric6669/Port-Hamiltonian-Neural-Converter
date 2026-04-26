# Port-Hamiltonian Neural Converter

This repository contains the code and Simulink models for the paper **"Physics-Priori Neural Converters Modelling for Intelligent EMT Simulation"**. The project builds a physics-priori neural converter based on port-Hamiltonian neural ordinary differential equations (pH-NODEs) for electromagnetic transient (EMT) simulation of a two-level voltage source converter.

The model preserves the known switching-induced interconnection matrix and learns only the uncertain dissipation and parameter mismatch terms. This keeps the converter structure physically interpretable, enforces positive dissipation through Softplus parameterizations, and supports stable long-horizon rollout for offline and real-time simulation.

## Repository Layout

```text
simulink/
  data_generation/      Parallel Simulink data-generation script
  offline/              Offline comparison model, initialization, export, and error analysis
  realtime/             Real-time simulation models for M1, M2, and M3
training/
  datasets/             Dataset preprocessing code and placeholders for raw/processed data
  model/                pH-NODE and NODE model definitions
  utils/                ODE integrators, losses, and plotting helpers
  plots/                Offline and HIL plotting/post-processing scripts
  checkpoints/          Small example PyTorch checkpoint
  main.py               Train and run inference from one entry point
  train.py              Supervised rollout training
  train_with_optuna.py  Bayesian hyper-parameter search
  inference.py          Long-horizon rollout inference and error reporting
  export_to_sim.py      Export trained PyTorch weights to MATLAB .mat format
```

## Model Definitions

The paper compares three converter models:

- **M1: IGBT/Diode detailed model**. This is the Simscape Universal Bridge reference model and is treated as ground truth.
- **M2: switching-function model**. This uses the ideal switching-function approximation for faster EMT simulation.
- **M3: pH-NODE neural converter**. This keeps the port-Hamiltonian interconnection structure and learns the dissipation matrix and inverse parameter matrix.

The offline Simulink model `simulink/offline/Compare_AI_kk.mdl` contains M1, M2, and M3 for open-loop and closed-loop comparison. The real-time models are separated as `RT_IGBT.slx`, `RT_SWF.slx`, and `RT_AI.slx` under `simulink/realtime/`.

## Data Generation

`simulink/data_generation/Parsim_for_ai.m` generates broadband transient trajectories with randomized sinusoidal perturbations applied to active-power reference, reactive-power reference, dc-side voltage, and ac-side voltage. The script uses parallel Simulink simulations and saves cleaned `.mat` records to a `raw/` folder.

Large generated datasets are intentionally not tracked by Git. Place generated records under:

```text
training/datasets/raw/
```

Then preprocess them with:

```bash
cd training
python datasets/preprocess.py --data_dir datasets/raw --save_dir datasets/processed --converter_model IGBT --normalization 0
```

Use `--converter_model Switching` to preprocess the switching-function channels instead.

## Training

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Train the pH-NODE model:

```bash
cd training
python main.py --converter_model IGBT --converter_neural_model ConverterPHNN --hidden_dim 8 --layers 3 --R_type NonlinearDiag2 --Pinv_type nominal --arch mlp --integrate_method euler
```

The architecture used in the paper is an MLP with input dimension 11, three hidden layers of width 8, and output dimension 5 for the learned dissipation term. Training uses supervised rollout sequences with Euler integration, AdamW, MSE loss, batch size 512, learning rate `3.33e-3`, and loss weights `2.04` for dc capacitor voltages and `1.01` for phase currents.

Hyper-parameter search can be repeated with:

```bash
cd training
python train_with_optuna.py --n_trials 100
```

## Export to Simulink

After training, export the learned PyTorch weights for the Simulink pH-NODE implementation:

```bash
cd training
python export_to_sim.py
```

The repository also includes the trained MATLAB weight file used by the provided offline Simulink model:

```text
simulink/offline/phnode_weights_3_8.mat
```

## Offline Simulation Workflow

1. Open MATLAB/Simulink and add `simulink/offline/` to the MATLAB path.
2. Run `Net_improve_init.m` to initialize converter parameters and simulation settings.
3. Open `Compare_AI_kk.mdl` and run the desired open-loop or closed-loop case.
4. Run `export_result_plot.m` to save `Y_IGBT.mat`, `Y_pred.mat`, and `Y_SWF.mat` from the Simulink `out` object.
5. Run `Compare_error.m` to compute RMSE, MAE, NRMSE, and relative RMSE for M3 and M2 against M1.

Plot scripts are provided in `training/plots/` for offline and HIL figures.

## Real-Time Simulation Workflow

The real-time models are organized as:

- `simulink/realtime/RT_IGBT.slx`: M1 detailed IGBT/Diode model.
- `simulink/realtime/RT_SWF.slx`: M2 switching-function model.
- `simulink/realtime/RT_AI.slx`: M3 pH-NODE neural converter.

These models correspond to the OPAL-RT/HIL validation in the paper. The reported tests compare 20 us and 40 us real-time steps under dc-voltage and active-power reference changes.

## Notes

- `model/kan.py` from the original training folder is not included because the released experiments use the MLP backbone.
- Large `.mat` trajectory files and processed `.pt` tensors are ignored to keep the GitHub repository lightweight.
- Simulink cache/build folders such as `slprj/` are ignored, while the selected `.slxc` file required by the offline model is retained.
