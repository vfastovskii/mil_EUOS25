from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MILTrainDataset(Dataset):
    """One item = one molecule (bag)."""

    def __init__(
        self,
        ids: List[str],
        X2d: np.ndarray,
        y_cls: np.ndarray,
        w_cls: np.ndarray,
        y_abs: np.ndarray,
        m_abs: np.ndarray,
        w_abs: np.ndarray,
        y_fluo: np.ndarray,
        m_fluo: np.ndarray,
        w_fluo: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
        id2pos: Dict[str, int],
        Xinst_sorted: np.ndarray,
        max_instances: int,
        seed: int,
    ):
        self.ids = [str(x) for x in ids]
        self.X2d = np.asarray(X2d, dtype=np.float32)

        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.w_cls = torch.tensor(w_cls, dtype=torch.float32)
        self.y_abs = torch.tensor(y_abs, dtype=torch.float32)
        self.m_abs = torch.tensor(m_abs, dtype=torch.bool)
        self.w_abs = torch.tensor(w_abs, dtype=torch.float32)
        self.y_fluo = torch.tensor(y_fluo, dtype=torch.float32)
        self.m_fluo = torch.tensor(m_fluo, dtype=torch.bool)
        self.w_fluo = torch.tensor(w_fluo, dtype=torch.float32)

        self.starts = starts
        self.counts = counts
        self.id2pos = id2pos
        self.Xinst = Xinst_sorted

        self.max_instances = int(max_instances)
        self.rng = np.random.default_rng(seed)

        if len(self.ids) != self.X2d.shape[0]:
            raise ValueError(f"MILTrainDataset: len(ids)={len(self.ids)} != X2d rows={self.X2d.shape[0]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        mol_id = self.ids[i]
        p = self.id2pos[mol_id]
        s = int(self.starts[p])
        c = int(self.counts[p])
        bag = self.Xinst[s : s + c]

        if self.max_instances > 0 and bag.shape[0] > self.max_instances:
            idx = self.rng.choice(bag.shape[0], size=self.max_instances, replace=False)
            bag = bag[idx]

        return (
            self.X2d[i],  # np float32 [F2]
            bag,  # np float32 [Ni,F3]
            self.y_cls[i],
            self.w_cls[i],  # torch [4]
            self.y_abs[i],
            self.m_abs[i],
            self.w_abs[i],  # [2]
            self.y_fluo[i],
            self.m_fluo[i],
            self.w_fluo[i],  # [4]
        )


class MILExportDataset(Dataset):
    """Leaderboard export dataset: returns mol_id, conf_ids list, plus tensors inputs."""

    def __init__(
        self,
        ids: List[str],
        X2d: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
        id2pos: Dict[str, int],
        Xinst_sorted: np.ndarray,
        conf_sorted: np.ndarray,
        max_instances: int = 0,
        seed: int = 0,
    ):
        self.ids = [str(x) for x in ids]
        self.X2d = np.asarray(X2d, dtype=np.float32)
        self.starts = starts
        self.counts = counts
        self.id2pos = id2pos
        self.Xinst = Xinst_sorted
        self.conf = conf_sorted
        self.max_instances = int(max_instances)
        self.rng = np.random.default_rng(seed)

        if len(self.ids) != self.X2d.shape[0]:
            raise ValueError(f"MILExportDataset: len(ids)={len(self.ids)} != X2d rows={self.X2d.shape[0]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[str, List[str], np.ndarray, np.ndarray]:
        mol_id = self.ids[i]
        p = self.id2pos[mol_id]
        s = int(self.starts[p])
        c = int(self.counts[p])

        bag = self.Xinst[s : s + c]
        conf = self.conf[s : s + c].tolist()

        if self.max_instances > 0 and bag.shape[0] > self.max_instances:
            idx = self.rng.choice(bag.shape[0], size=self.max_instances, replace=False)
            bag = bag[idx]
            conf = [conf[j] for j in idx.tolist()]

        return mol_id, conf, self.X2d[i], bag
