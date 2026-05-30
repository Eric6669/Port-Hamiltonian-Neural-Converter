#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PaperCode-Neural-Converters-for-AI-EMT-Simulation 
@File    ：preprocess.py
@Author  ：He Xing
@Date    ：2026/4/2 09:05 
"""

import os
import argparse
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

# ===================================================================
#  [0:5]   z   = [vcp, vcn, ia, ib, ic]
#  [5:10]  u_ext = [idcp, idcn, ea, eb, ec]
#  [10:13] vref  = [vrefa, vrefb, vrefc]
#  [13:19] u_sw  = [Sa_up, Sa_dn, Sb_up, Sb_dn, Sc_up, Sc_dn]
#  [19:21] PQ    = [P, Q]
# ===================================================================

COLUMN_LAYOUT = {
    # 2-level: SWF-offset==0 IGBT-offset==21
    "2level": {
        "offset": 21,
        "z": (0, 5),
        "u_ext": (5, 10),
        "vref": (10, 13),
        "u_sw": (13, 19),
        "pq": (19, 21),
        "sw_dim": 6,
        "sw_names": ["Sa_up", "Sa_dn", "Sb_up", "Sb_dn", "Sc_up", "Sc_dn"],
    },

    # 3-level: NPC-offset==0 T-offset==27
    "3level": {
        "offset": 0,
        "z": (0, 5),
        "u_ext": (5, 10),
        "vref": (10, 13),
        "u_sw": (13, 25),
        "pq": (25, 27),
        "sw_dim": 12,
        "sw_names": [
            "Sa_1", "Sa_2", "Sa_3", "Sa_4",
            "Sb_5", "Sb_6", "Sb_7", "Sb_8",
            "Sc_9", "Sc_10", "Sc_11", "Sc_12",
        ],
    },
}

# ===================================================================
#  Data Processor
# ===================================================================
class ConverterDataProcessor:
    def __init__(
            self,
            data_dir: str,
            ts: float,
            converter_model: str,
            normalization: int = 1,
            downsample: int = 1,
            file_name: str = "all"
    ):
        """
        :param data_dir:  .mat
        :param ts: simulation sampling step
        :param converter_model: '2level' '3level'
        :param normalization: 1 Yes
        :param downsample: 1 nodownsample
        :return: dataset
        """
        self.data_dir = data_dir
        self.Ts = ts
        self.converter_model = converter_model
        self.normalization = normalization
        self.downsample = downsample
        self.layout = COLUMN_LAYOUT[converter_model]
        self.offset = self.layout["offset"]
        self.file_name = file_name

    def _load_single_file(self, file_path: str):
        """single .mat extract z, u_ext, u_sw, compute dz/dt"""
        mat_data = sio.loadmat(file_path)
        data = mat_data["clean_data"]

        off = self.offset
        layout = self.layout

        z_start, z_end = layout["z"]
        u_start, u_end = layout["u_ext"]
        sw_start, sw_end = layout["u_sw"]

        # [vcp, vcn, ia, ib, ic]
        z = data[:, off + z_start: off + z_end]
        z[:, 2:5] = -z[:, 2:5]

        # [idcp, idcn, ea, eb, ec]
        u_ext = data[:, off + u_start: off + u_end]

        # [Sa_up, Sa_dn, Sb_up, Sb_dn, Sc_up, Sc_dn]
        u_sw = data[:, off + sw_start: off + sw_end]

        # downsample
        if self.downsample > 1:
            z = z[:: self.downsample, :]
            u_ext = u_ext[:: self.downsample, :]
            u_sw = u_sw[:: self.downsample, :]

        z = z[:-1, :]
        u_ext = u_ext[:-1, :]
        u_sw = u_sw[:-1, :]

        return z, u_ext, u_sw

    def _load_all_files(self):
        """traverse data_dir all .mat"""
        file_list = sorted(
            [f for f in os.listdir(self.data_dir) if f.endswith(".mat")]
        )

        print(f"total: {len(file_list)} mats, "
              f"converter_model: {self.converter_model}, "
              f"downsample: {self.downsample}x")

        trajectories = []
        for fname in file_list:
            try:
                z, u, sw = self._load_single_file(os.path.join(self.data_dir, fname))
                trajectories.append((z, u, sw))
            except Exception as e:
                print(f"[WARN] Skip {fname}: {e}")

        print(f"[INFO] Load {len(trajectories)} mats")
        return trajectories

    def _load_data(self):
        if self.file_name.lower() == "all":
            return self._load_all_files()
        else:
            fpath = os.path.join(self.data_dir, self.file_name)
            print(f"[INFO] Loading single file: {self.file_name}")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"File not found: {fpath}")
            z, u, sw = self._load_single_file(fpath)
            return [(z, u, sw)]

    def _fit_and_transform(self, trajectories):
        """
        except sw
        """
        if self.normalization != 1:
            return trajectories, None

        Z_cat = np.concatenate([t[0] for t in trajectories], axis=0)
        U_cat = np.concatenate([t[1] for t in trajectories], axis=0)

        scaler_z = MaxAbsScaler().fit(Z_cat)
        scaler_u = MaxAbsScaler().fit(U_cat)

        normed = []
        for z, u, sw in trajectories:
            normed.append((
                scaler_z.transform(z),
                scaler_u.transform(u),
                sw,
            ))

        scalers = {"z": scaler_z, "u_ext": scaler_u}
        return normed, scalers

    @staticmethod
    def _print_stats(Z, U_ext):
        """check raw data"""
        names_z = ["vcp", "vcn", "ia", "ib", "ic"]
        names_u = ["idcp", "idcn", "ea", "eb", "ec"]
        print(f"\n{'Channel':<12s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
        print("-" * 72)
        for i, n in enumerate(names_z):
            c = Z[:, i]
            print(f"z.{n:<9s} {c.mean():>12.2f} {c.std():>12.2f} "
                  f"{c.min():>12.2f} {c.max():>12.2f}")
        for i, n in enumerate(names_u):
            c = U_ext[:, i]
            print(f"u.{n:<9s} {c.mean():>12.2f} {c.std():>12.2f} "
                  f"{c.min():>12.2f} {c.max():>12.2f}")

    def generate_and_save(self, save_dir: str = "processed"):
        """
        Main process

        output:
            save_dir/
            ├── {model}_dataset.pt    # TensorDataset
            └── {model}_meta.pt       # scalers

        """
        os.makedirs(save_dir, exist_ok=True)

        if self.normalization == 1:
            tag = f"{self.converter_model}_norm"
        else:
            tag = self.converter_model

        trajectories = self._load_data()

        Z_cat = np.concatenate([t[0] for t in trajectories], axis=0)
        U_cat = np.concatenate([t[1] for t in trajectories], axis=0)
        self._print_stats(Z_cat, U_cat)

        normed_trajs, scalers = self._fit_and_transform(trajectories)

        N = len(normed_trajs)
        lengths = [t[0].shape[0] for t in normed_trajs]
        T = min(lengths)
        if max(lengths) != T:
            print(f"[WARN] Truncating to min length: {T}")

        z_tensor = torch.zeros(N, T, 5, dtype=torch.float32)
        u_tensor = torch.zeros(N, T, 5, dtype=torch.float32)
        sw_dim = self.layout["sw_dim"]
        sw_tensor = torch.zeros(N, T, sw_dim, dtype=torch.float32)

        for i, (z, u, sw) in enumerate(normed_trajs):
            z_tensor[i] = torch.from_numpy(z[:T].astype(np.float32))
            u_tensor[i] = torch.from_numpy(u[:T].astype(np.float32))
            sw_tensor[i] = torch.from_numpy(sw[:T].astype(np.float32))

        data_dict = {"z": z_tensor, "u_ext": u_tensor, "u_sw": sw_tensor}
        data_path = os.path.join(save_dir, f"{tag}_trajectories.pt")
        torch.save(data_dict, data_path)

        print(f"\n[SAVE] {data_path}")

        if scalers:
            print(f"  z scale factors: {scalers['z'].max_abs_}")
            print(f"  u scale factors: {scalers['u_ext'].max_abs_}")

        meta = {
            "converter_model": self.converter_model,
            "Ts": self.Ts,
            "downsample": self.downsample,
            "dt": self.Ts * self.downsample,
            "normalization": self.normalization,
            "normalization_method": "MaxAbsScaler",
            "scalers": scalers,
            "n_trajectories": N,
            "trajectory_length": T,
            "file_loaded": self.file_name,
            "feature_names": {
                "z": ["vcp", "vcn", "ia", "ib", "ic"],
                "u_ext": ["idcp", "idcn", "ea", "eb", "ec"],
                "u_sw": self.layout["sw_names"],
            },
        }
        meta_path = os.path.join(save_dir, f"{tag}_meta.pt")
        torch.save(meta, meta_path)
        print(f"[SAVE] {meta_path}")

class TrajectorySliceDataset(Dataset):
    """
    seq_len+1 trajectory slice

       data format:
            z:    (N_traj, T, 5)
            u:    (N_traj, T, 5)
            sw:   (N_traj, T, 6)

       each sample:
           z_seg:  (seq_len+1, 5)   z[t0:t0+seq_len+1]
           u_seg:  (seq_len+1, 5)   u[t0:t0+seq_len+1]
           sw_seg: (seq_len+1, 6)   sw[t0:t0+seq_len+1]
    """

    def __init__(self, z, u, sw, seq_len, samples_per_epoch=10000):
        self.z = z
        self.u = u
        self.sw = sw
        self.seq_len = seq_len
        self.N_traj = z.shape[0]
        self.T = z.shape[1]
        self.max_start = self.T - seq_len - 1
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):

        traj_idx = torch.randint(0, self.N_traj, (1,)).item()
        start = torch.randint(0, self.max_start + 1, (1,)).item()
        end = start + self.seq_len + 1

        return (
            self.z[traj_idx, start:end],    # (L+1, 5)
            self.u[traj_idx, start:end],    # (L+1, 5)
            self.sw[traj_idx, start:end],   # (L+1, 6)
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="raw data directory")
    parser.add_argument("--ts",type=float, default=20e-6, help="time step")
    parser.add_argument("--converter_model", type=str, default="2level", help="2level, 3level")
    parser.add_argument("--normalization", type=int, default=1, help="1-yes, 0-no")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--file_name", type=str, default="all") # sim_record_001.mat
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(args.converter_model, "raw")

    if args.save_dir is None:
        args.save_dir = os.path.join(args.converter_model, "processed")

    processor = ConverterDataProcessor(
        data_dir=args.data_dir,
        ts=args.ts,
        converter_model=args.converter_model,
        normalization=args.normalization,
        downsample=args.downsample,
        file_name=args.file_name
    )
    processor.generate_and_save(save_dir=args.save_dir)
