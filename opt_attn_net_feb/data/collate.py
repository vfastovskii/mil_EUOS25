from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def collate_train(batch):
    B = len(batch)
    x2d_np = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    bags_np = [b[1] for b in batch]
    lens = np.array([int(x.shape[0]) for x in bags_np], dtype=np.int64)
    max_len = int(lens.max())

    F3 = int(bags_np[0].shape[1])
    x3d_pad = np.zeros((B, max_len, F3), dtype=np.float32)
    for i, bag in enumerate(bags_np):
        x3d_pad[i, : bag.shape[0], :] = bag

    # key_padding_mask True=PAD
    kpm = np.ones((B, max_len), dtype=bool)
    for i, L in enumerate(lens):
        kpm[i, :L] = False

    x2d = torch.from_numpy(x2d_np).float()
    x3d = torch.from_numpy(x3d_pad).float()
    kpm_t = torch.from_numpy(kpm)

    y_cls = torch.stack([b[2] for b in batch], dim=0)
    w_cls = torch.stack([b[3] for b in batch], dim=0)

    y_abs = torch.stack([b[4] for b in batch], dim=0)
    m_abs = torch.stack([b[5] for b in batch], dim=0)
    w_abs = torch.stack([b[6] for b in batch], dim=0)

    y_fluo = torch.stack([b[7] for b in batch], dim=0)
    m_fluo = torch.stack([b[8] for b in batch], dim=0)
    w_fluo = torch.stack([b[9] for b in batch], dim=0)

    return x2d, x3d, kpm_t, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo


def collate_export(batch):
    # returns mol_ids(list[str]), conf_pad(object array), x2d, x3d, kpm
    B = len(batch)
    mol_ids = [b[0] for b in batch]
    conf_lists = [b[1] for b in batch]
    x2d_np = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
    bags_np = [b[3] for b in batch]

    lens = np.array([len(x) for x in conf_lists], dtype=np.int64)
    max_len = int(lens.max())
    F3 = int(bags_np[0].shape[1])

    x3d_pad = np.zeros((B, max_len, F3), dtype=np.float32)
    conf_pad = np.empty((B, max_len), dtype=object)
    conf_pad[:] = ""

    for i in range(B):
        L = int(lens[i])
        x3d_pad[i, :L, :] = bags_np[i]
        conf_pad[i, :L] = conf_lists[i]

    kpm = np.ones((B, max_len), dtype=bool)
    for i, L in enumerate(lens):
        kpm[i, :int(L)] = False

    return (
        mol_ids,
        conf_pad,
        torch.from_numpy(x2d_np).float(),
        torch.from_numpy(x3d_pad).float(),
        torch.from_numpy(kpm),
    )
