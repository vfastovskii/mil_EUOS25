from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils.constants import TASK_COLS


@torch.no_grad()
def export_leaderboard_attention(
    model: Any,
    dl_lb_export: Any,
    device: torch.device,
    out_path: Path,
) -> None:
    """Export attention weights per (ID, conf_id, task) into a Parquet/CSV file.

    Parameters
    - model: MILTaskAttnMixerWithAux (or compatible) that returns (logits, abs, fluo, attn)
             when called with return_attn=True; attn shape must be [B, 4, N].
    - dl_lb_export: DataLoader yielding (mol_ids, conf_pad, x2d, x3d, kpm)
    - device: torch device for inference
    - out_path: destination file (.parquet, .pq, or .csv)
    """
    model.eval()
    model.to(device)

    rows: List[Dict[str, Any]] = []

    for mol_ids, conf_pad, x2d, x3d, kpm in dl_lb_export:
        x2d = x2d.to(device, non_blocking=True)
        x3d = x3d.to(device, non_blocking=True)
        kpm = kpm.to(device, non_blocking=True)

        logits, abs_out, fluo_out, attn = model(x2d, x3d, kpm, return_attn=True)
        if attn is None:
            raise RuntimeError("Attention not returned; expected attn when return_attn=True")

        attn_np = attn.detach().cpu().numpy()          # [B,4,N]
        kpm_np = kpm.detach().cpu().numpy().astype(bool)
        B, T, N = attn_np.shape

        for b in range(B):
            mid = str(mol_ids[b])
            valid = ~kpm_np[b]
            L = int(valid.sum())
            if L <= 0:
                continue
            confs = [str(x) for x in conf_pad[b, :L].tolist()]

            for t in range(T):
                w = attn_np[b, t, :L].astype(np.float64)
                s = float(w.sum())
                if not np.isfinite(s) or s <= 0:
                    w[:] = 1.0 / float(L)
                else:
                    w /= s  # enforce sum-to-1 (safety)
                task_name = TASK_COLS[t]
                for i in range(L):
                    rows.append({"ID": mid, "conf_id": confs[i], "task": task_name, "attn_weight": float(w[i])})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    if out_path.suffix.lower() in [".parquet", ".pq"]:
        df_out.to_parquet(out_path, index=False)
    else:
        df_out.to_csv(out_path, index=False)

    print(f"[ATTN] wrote {len(df_out)} rows -> {out_path}")
