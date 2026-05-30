#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：testai_plot_result.py
@Author  ：He Xing
@Date    ：2026/5/25 22:18 
"""

import os
import time
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference import (
    build_model,
    get_data_tag,
    load_raw_trajectory,
    rollout,
    rollout_lstm,
    ODE_MODEL_NAMES,
    SEQ_MODEL_NAMES,
)

from utils.integrator import euler_integrate, rk4_integrate


# =========================================================
# Font setting
# =========================================================
font_path = os.path.join(PROJECT_ROOT, "plots", "times.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    print(f"Success: Loaded font from {font_path}.")
else:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    print("Font file not found. Using default font.")


# =========================================================
# Variable information
# =========================================================
VAR_INFO = {
    "vcp": {"col": 0, "label": r"$v_{dc,p}$", "unit": "V"},
    "vcn": {"col": 1, "label": r"$v_{dc,n}$", "unit": "V"},
    "ia": {"col": 2, "label": r"$i_{sa}$", "unit": "A"},
    "ib": {"col": 3, "label": r"$i_{sb}$", "unit": "A"},
    "ic": {"col": 4, "label": r"$i_{sc}$", "unit": "A"},
}


# =========================================================
# Model configuration
# 按你的实际训练配置修改这里
# =========================================================
MODEL_CONFIGS = {
    "pHNODE": {
        "plot_label": "pH-NODE",
        "hidden_dim": 8,
        "layers": 3,
        "normalization": 0,
        "ckpt": os.path.join(PROJECT_ROOT, "checkpoints", "2level_pHNODE.pt"),
    },
    "LSTM": {
        "plot_label": "LSTM",
        "hidden_dim": 128,
        "layers": 2,
        "normalization": 1,
        "ckpt": os.path.join(PROJECT_ROOT, "checkpoints", "2level_LSTM.pt"),
        "lstm_dropout": 0.0,
    },
    "NODE": {
        "plot_label": "NODE",
        "hidden_dim": 128,
        "layers": 4,
        "normalization": 1,
        "ckpt": os.path.join(PROJECT_ROOT, "checkpoints", "2level_NODE.pt"),
    },
    "PINODE": {
        "plot_label": "PI-NODE",
        "hidden_dim": 512,
        "layers": 3,
        "normalization": 1,
        "ckpt": os.path.join(PROJECT_ROOT, "checkpoints", "2level_PINODE.pt"),
        "physics_weight": 0.5,
    },
}


PLOT_STYLE = {
    "DUB": {
        "color": "#1f77b4",
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 0.95,
    },
    "pHNODE": {
        "color": "#ff7f0e",
        "linestyle": "--",
        "linewidth": 1.4,
        "alpha": 0.95,
    },
    "LSTM": {
        "color": "#2ca02c",
        "linestyle": "-.",
        "linewidth": 1.2,
        "alpha": 0.90,
    },
    "NODE": {
        "color": "#d62728",
        "linestyle": ":",
        "linewidth": 1.5,
        "alpha": 0.90,
    },
    "PINODE": {
        "color": "#9467bd",
        "linestyle": (0, (5, 1)),
        "linewidth": 1.2,
        "alpha": 0.90,
    },
}


def load_scalers(converter_model, normalization):
    """
    Load scalers from *_meta.pt according to normalization setting.
    """
    dummy_args = SimpleNamespace(
        converter_model=converter_model,
        normalization=normalization,
    )

    data_tag = get_data_tag(dummy_args)

    meta_path = os.path.join(
        PROJECT_ROOT,
        "datasets",
        converter_model,
        "processed",
        f"{data_tag}_meta.pt",
    )

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    meta = torch.load(meta_path, weights_only=False)
    return meta.get("scalers", None)


def normalize_inputs(z_raw, u_raw, scalers):
    """
    Normalize z and u_ext if scalers exist.
    """
    if scalers is None:
        return z_raw, u_raw

    scaler_z = scalers["z"]
    scaler_u = scalers["u_ext"]

    z_norm = scaler_z.transform(z_raw)
    u_norm = scaler_u.transform(u_raw)

    return z_norm, u_norm


def inverse_z(z_pred_norm, scalers):
    """
    Inverse transform z prediction.
    """
    if scalers is None:
        return z_pred_norm

    scaler_z = scalers["z"]
    return scaler_z.inverse_transform(z_pred_norm)


@torch.no_grad()
def infer_single_model(
        model_name,
        cfg,
        converter_model,
        z_raw,
        u_raw,
        sw_raw,
        dt,
        integrate_method,
        chunk_size,
        device,
):
    print("\n" + "=" * 70)
    print(f"[INFER] {model_name} ({cfg['plot_label']})")
    print("=" * 70)

    scalers = load_scalers(converter_model, cfg["normalization"])
    z_in, u_in = normalize_inputs(z_raw, u_raw, scalers)

    model_args = SimpleNamespace(
        converter_model=converter_model,
        converter_neural_model=model_name,
        hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"],
        normalization=cfg["normalization"],
        physics_weight=cfg.get("physics_weight", 0.5),
        lstm_dropout=cfg.get("lstm_dropout", 0.0),
    )

    model = build_model(model_args).to(device)

    ckpt_path = cfg["ckpt"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt_path}")

    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()

    z_tensor = torch.from_numpy(z_in.astype(np.float32)).unsqueeze(0)
    u_tensor = torch.from_numpy(u_in.astype(np.float32)).unsqueeze(0)
    sw_tensor = torch.from_numpy(sw_raw.astype(np.float32)).unsqueeze(0)

    z0 = z_tensor[:, 0, :]

    if integrate_method == "euler":
        integrate_fn = euler_integrate
    elif integrate_method == "rk4":
        integrate_fn = rk4_integrate
    else:
        raise ValueError(f"Unknown integrate_method: {integrate_method}")

    t0 = time.time()

    if model_name in ODE_MODEL_NAMES:
        z_pred_norm = rollout(
            model=model,
            z0=z0,
            u_seq=u_tensor,
            sw_seq=sw_tensor,
            dt=dt,
            chunk_size=chunk_size,
            integrate_fn=integrate_fn,
        )
        start_offset = 1

    elif model_name in SEQ_MODEL_NAMES:
        z_pred_norm = rollout_lstm(
            model=model,
            z0=z0,
            u_seq=u_tensor,
            sw_seq=sw_tensor,
        )
        start_offset = 1

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    elapsed = time.time() - t0
    print(f"[DONE] {model_name}: {elapsed:.2f} s")

    z_pred_raw = inverse_z(z_pred_norm, scalers)

    return z_pred_raw, start_offset


def compute_rmse_table(Y_true, pred_dict):
    """
    Print physical-domain RMSE for quick check.
    """
    names = ["vcp", "vcn", "ia", "ib", "ic"]

    print("\n" + "=" * 70)
    print("[RMSE]")
    print("=" * 70)
    print(f"{'Model':<10s} " + " ".join([f"{n:>12s}" for n in names]))
    print("-" * 70)

    for model_name, Y_pred in pred_dict.items():
        min_len = min(Y_true.shape[0], Y_pred.shape[0])
        err = Y_pred[:min_len] - Y_true[:min_len]
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        print(
            f"{model_name:<10s} "
            + " ".join([f"{v:>12.4f}" for v in rmse])
        )


def plot_one_variable(
        t,
        Y_true,
        pred_dict,
        var_key,
        save_path,
        zoom_x_range=None,
        inset_pos=(0.55, 0.45, 0.38, 0.38),
):
    info = VAR_INFO[var_key]
    col = info["col"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.2))

    # =========================================================
    # 1) Main trajectories on left axis
    # =========================================================
    y_true = Y_true[:, col]

    dub_style = PLOT_STYLE["DUB"]

    ax.plot(
        t,
        y_true,
        label="DUB",
        color=dub_style["color"],
        linestyle=dub_style["linestyle"],
        linewidth=dub_style["linewidth"],
        alpha=dub_style["alpha"],
        zorder=3,
    )

    plot_order = ["pHNODE", "LSTM", "NODE", "PINODE"]

    for model_name in plot_order:
        if model_name not in pred_dict:
            continue

        cfg = MODEL_CONFIGS[model_name]
        style = PLOT_STYLE[model_name]
        y_pred = pred_dict[model_name][:, col]

        ax.plot(
            t,
            y_pred,
            label=cfg["plot_label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=3,
        )

    y_label_str = f"{info['label']} ({info['unit']})"
    ax.set_ylabel(y_label_str, fontsize=24, labelpad=5)
    ax.set_xlabel("Time (s)", fontsize=24)

    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=20, framealpha=0.9)
    ax.set_xlim(t[0], t[-1])

    # =========================================================
    # 2) Right axis: absolute error area for each model
    # =========================================================
    ax2 = ax.twinx()

    color_err_axis = "#8b4513"
    max_err = 1e-6

    # 每个模型误差用对应模型的颜色，只降低透明度
    error_alpha = {
        "pHNODE": 0.30,
        "LSTM": 0.20,
        "NODE": 0.18,
        "PINODE": 0.18,
    }

    for model_name in plot_order:
        if model_name not in pred_dict:
            continue

        style = PLOT_STYLE[model_name]
        y_pred = pred_dict[model_name][:, col]
        abs_err = np.abs(y_true - y_pred)

        ax2.fill_between(
            t,
            0,
            abs_err,
            color=style["color"],
            alpha=error_alpha.get(model_name, 0.20),
            edgecolor="none",
            zorder=1,
        )

        max_err = max(max_err, float(np.max(abs_err)))

    ax2.set_ylabel(
        f"AbsError ({info['unit']})",
        fontsize=24,
        color=color_err_axis,
        labelpad=5,
    )
    ax2.tick_params(axis="y", labelcolor=color_err_axis, labelsize=22)
    ax2.spines["right"].set_color(color_err_axis)
    ax2.spines["left"].set_color("black")
    ax2.set_ylim(0, max_err * 3.5)

    # 让左轴主曲线显示在误差阴影上方
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # =========================================================
    # 3) Inset zoom: only plot trajectories, not error areas
    # =========================================================
    if zoom_x_range is not None:
        axins = ax.inset_axes(inset_pos)
        axins.set_facecolor("white")
        axins.patch.set_alpha(0.95)

        axins.plot(
            t,
            y_true,
            color=dub_style["color"],
            linestyle=dub_style["linestyle"],
            linewidth=1.5,
            alpha=dub_style["alpha"],
        )

        for model_name in plot_order:
            if model_name not in pred_dict:
                continue

            style = PLOT_STYLE[model_name]
            y_pred = pred_dict[model_name][:, col]

            axins.plot(
                t,
                y_pred,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.2,
                alpha=style["alpha"],
            )

        z_x1, z_x2 = zoom_x_range
        idx1 = np.searchsorted(t, z_x1)
        idx2 = np.searchsorted(t, z_x2)

        if idx2 > idx1:
            window_values = [y_true[idx1:idx2]]

            for model_name in plot_order:
                if model_name not in pred_dict:
                    continue
                window_values.append(pred_dict[model_name][idx1:idx2, col])

            y_window = np.concatenate(window_values)
            z_y1 = np.min(y_window)
            z_y2 = np.max(y_window)

            margin = (z_y2 - z_y1) * 0.1 if z_y2 != z_y1 else 1e-3

            axins.set_xlim(z_x1, z_x2)
            axins.set_ylim(z_y1 - margin, z_y2 + margin)

            axins.tick_params(axis="both", which="major", labelsize=14)
            axins.grid(True, alpha=0.3)

            ax.indicate_inset_zoom(
                axins,
                edgecolor="black",
                alpha=0.8,
                linewidth=1.5,
            )

    # =========================================================
    # 4) Save
    # =========================================================
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", format="svg")
    plt.close()
    print(f"[SAVE] {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--converter_model", type=str, default="2level")
    parser.add_argument("--infer_file", type=str, default="sim_record_001.mat")
    parser.add_argument("--Ts", type=float, default=20e-6)
    parser.add_argument("--integrate_method", type=str, default="euler", choices=["euler", "rk4"])
    parser.add_argument("--chunk_size", type=int, default=1000)

    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "plots", "testai_plot"))
    parser.add_argument("--vars", type=str, default="vcp,vcn,ia,ib,ic")

    parser.add_argument("--skip_time", type=float, default=0.0)
    parser.add_argument(
        "--plot_time",
        type=float,
        default=1.0,
        help="Only plot the first plot_time seconds after skip_time. Example: 1.0 means plot 0-1 s."
    )

    # 示例：--zoom 2.50,2.65；不需要局部放大就留空
    parser.add_argument("--zoom", type=str, default="")

    # 例如 cuda, cpu
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    print(f"[DEVICE] {device}")

    # =========================================================
    # 1. Load test trajectory
    # =========================================================
    raw_path = os.path.join(
        PROJECT_ROOT,
        "datasets",
        args.converter_model,
        "raw",
        args.infer_file,
    )

    print(f"[LOAD RAW] {raw_path}")
    z_raw, u_raw, sw_raw = load_raw_trajectory(raw_path, args.converter_model)

    T = z_raw.shape[0]
    t_raw = np.arange(T) * args.Ts

    # =========================================================
    # 2. Infer all models
    # =========================================================
    pred_dict_full = {}
    start_offsets = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        z_pred_raw, start_offset = infer_single_model(
            model_name=model_name,
            cfg=cfg,
            converter_model=args.converter_model,
            z_raw=z_raw,
            u_raw=u_raw,
            sw_raw=sw_raw,
            dt=args.Ts,
            integrate_method=args.integrate_method,
            chunk_size=args.chunk_size,
            device=device,
        )

        pred_dict_full[model_name] = z_pred_raw
        start_offsets[model_name] = start_offset

    common_offset = 1
    min_len = min([p.shape[0] for p in pred_dict_full.values()])
    min_len = min(min_len, z_raw.shape[0] - common_offset)

    t = t_raw[common_offset: common_offset + min_len]
    Y_true = z_raw[common_offset: common_offset + min_len]

    pred_dict = {
        model_name: pred[:min_len]
        for model_name, pred in pred_dict_full.items()
    }

    # =========================================================
    # 3. Optional plot crop
    # =========================================================
    if args.skip_time > 0 or args.plot_time is not None:
        t0 = args.skip_time
        t1 = np.inf if args.plot_time is None else args.skip_time + args.plot_time

        keep_idx = (t >= t0) & (t <= t1)

        t = t[keep_idx]
        Y_true = Y_true[keep_idx]

        for model_name in list(pred_dict.keys()):
            pred_dict[model_name] = pred_dict[model_name][keep_idx]

        # 将横坐标重新从 0 开始，方便显示为 0-1 s
        t = t - t[0]

    # =========================================================
    # 4. Print RMSE
    # =========================================================
    compute_rmse_table(Y_true, pred_dict)

    # =========================================================
    # 5. Plot figures
    # =========================================================
    vars_to_plot = [v.strip() for v in args.vars.split(",") if v.strip()]

    zoom_x_range = None
    if args.zoom.strip():
        x1, x2 = args.zoom.split(",")
        zoom_x_range = (float(x1), float(x2))

    os.makedirs(args.save_dir, exist_ok=True)

    for var_key in vars_to_plot:
        if var_key not in VAR_INFO:
            raise ValueError(f"Unknown variable: {var_key}")

        save_path = os.path.join(
            args.save_dir,
            f"TestAI_AllModels_{var_key}.svg",
        )

        plot_one_variable(
            t=t,
            Y_true=Y_true,
            pred_dict=pred_dict,
            var_key=var_key,
            save_path=save_path,
            zoom_x_range=zoom_x_range,
            inset_pos=(0.55, 0.45, 0.38, 0.38),
        )


if __name__ == "__main__":
    main()