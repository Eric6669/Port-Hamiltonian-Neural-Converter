#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation
@File    ：computa_cost_result.py
@Author  ：He Xing
@Date    ：2026/5/28
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


def plot_scalability(save_path="scalability_multiple_converters.svg"):
    # =========================================================
    # Font setting
    # =========================================================
    font_path = os.path.join(os.getcwd(), "times.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "serif"

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

    # =========================================================
    # Hard-coded data
    # =========================================================
    num_converters = np.array([3, 5, 10, 50], dtype=float)

    # 使用等间距类别横坐标，避免 3 和 5 过于拥挤，50 又太远
    x_pos = np.arange(len(num_converters))  # [0, 1, 2, 3]
    x_labels = ["3", "5", "10", "50"]

    # Avg. time/step, unit: us
    avg_time = {
        "DUB":     np.array([5.45, np.nan, np.nan, np.nan]),
        "SWF":     np.array([2.33, 4.94, 18.38, np.nan]),
        "pH-NODE": np.array([1.71, 2.30, 3.64, 14.81]),
    }

    # CPU usage, unit: %
    cpu_usage = {
        "DUB":     np.array([27.23, 100.00, np.nan, np.nan]),
        "SWF":     np.array([11.66, 24.70, 91.89, np.nan]),
        "pH-NODE": np.array([8.56, 11.50, 18.21, 74.05]),
    }

    # =========================================================
    # Style
    # =========================================================
    colors = {
        "DUB": "#1f77b4",
        "SWF": "#2ca02c",
        "pH-NODE": "#ff7f0e",
    }

    linestyles = {
        "DUB": "-",
        "SWF": "-.",
        "pH-NODE": "--",
    }

    markers = {
        "DUB": "o",
        "SWF": "s",
        "pH-NODE": "^",
    }

    fill_alpha = {
        "DUB": 0.12,
        "SWF": 0.14,
        "pH-NODE": 0.18,
    }

    # =========================================================
    # Figure
    # =========================================================
    fig, ax = plt.subplots(figsize=(14, 4))
    ax2 = ax.twinx()

    # =========================================================
    # Right axis: CPU usage area
    # =========================================================
    for model in ["DUB", "SWF", "pH-NODE"]:
        valid = ~np.isnan(cpu_usage[model])

        ax2.fill_between(
            x_pos[valid],
            0,
            cpu_usage[model][valid],
            color=colors[model],
            alpha=fill_alpha[model],
            edgecolor="none",
            zorder=1,
        )

    # =========================================================
    # Left axis: average execution time line
    # =========================================================
    line_handles = []

    for model in ["DUB", "SWF", "pH-NODE"]:
        valid = ~np.isnan(avg_time[model])

        x_valid = x_pos[valid]
        y_valid = avg_time[model][valid]

        line, = ax.plot(
            x_valid,
            y_valid,
            label=model,
            color=colors[model],
            linestyle=linestyles[model],
            marker=markers[model],
            markersize=8,
            linewidth=2.2,
            markerfacecolor="white",
            markeredgewidth=1.8,
            zorder=5,
        )
        line_handles.append(line)

        # =====================================================
        # Annotate Avg. time/step values
        # DUB and SWF: above the line
        # pH-NODE: below the line
        # =====================================================
        for x, y in zip(x_valid, y_valid):
            if model == "DUB":
                dx = -0.08
                dy = 0.70
                va = "bottom"

            elif model == "SWF":
                dx = 0.08
                dy = 0.60
                va = "bottom"

            else:  # pH-NODE
                dx = 0.00
                dy = -0.70
                va = "top"

            # 单独微调，避免局部重叠
            if model == "pH-NODE" and x == x_pos[0]:
                dx = 0.12
                dy = -0.75

            if model == "SWF" and x == x_pos[0]:
                dx = -0.12
                dy = 0.75

            if model == "DUB" and x == x_pos[0]:
                dx = -0.10
                dy = 0.75

            if model == "pH-NODE" and x == x_pos[1]:
                dx = 0.08
                dy = -0.75

            if model == "SWF" and x == x_pos[1]:
                dx = 0.08
                dy = 0.65

            ax.text(
                x + dx,
                y + dy,
                rf"{y:.2f}$\mu$s",
                color=colors[model],
                fontsize=22,
                ha="center",
                va=va,
                zorder=6,
            )

    # =========================================================
    # Axis setting
    # =========================================================
    ax.set_xlabel("Number of Converters", fontsize=26)
    ax.set_ylabel(r"Avg. Time/Step ($\mu$s)", fontsize=26)
    ax2.set_ylabel("CPU Usage (%)", fontsize=26, color="#8b4513")

    ax.tick_params(axis="both", which="major", labelsize=26)
    ax2.tick_params(axis="y", which="major", labelsize=26, colors="#8b4513")

    ax2.spines["right"].set_color("#8b4513")

    # 横坐标使用等间距类别轴
    ax.set_xlim(-0.35, len(x_pos) - 1 + 0.35)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)

    # 给 pH-NODE 下方标签留一点空间
    ax.set_ylim(-1.2, 22)
    ax2.set_ylim(0, 100)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.30)

    # Make line axis appear above CPU usage area
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # =========================================================
    # Legend
    # =========================================================
    handles1, labels1 = ax.get_legend_handles_labels()

    ax.legend(
        handles1,
        labels1,
        loc="upper left",
        fontsize=24,
        framealpha=0.95,
        edgecolor="black",
        ncol=1,
        labelspacing=0.3,
        handlelength=2.0,
        borderpad=0.4,
    )

    # =========================================================
    # Save
    # =========================================================
    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches="tight",
        format=save_path.split(".")[-1],
        dpi=600
    )
    plt.close()

    print(f"[SAVE] {save_path}")


if __name__ == "__main__":
    plot_scalability("scalability_multiple_converters.svg")