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
    """
    Exports attention weights for leaderboard evaluation to a specified output path.

    This function processes the attention weights from the model's predictions on a data
    loader, filters and normalizes them, and writes the resulting data to a file in either
    Parquet or CSV format. The output file includes molecule IDs, conformer IDs, tasks, and
    attention weights.

    Parameters:
        model: Any
            The PyTorch model that will generate predictions including attention weights.

        dl_lb_export: Any
            Data loader containing the input data for evaluation (e.g., molecule data,
            conformations, etc.).

        device: torch.device
            The device (CPU or GPU) to which the model and input data will be moved during
            processing.

        out_path: Path
            The file path where the attention data will be written. File format is inferred
            from the extension (.parquet or .csv).

    Raises:
        RuntimeError:
            Raised if attention weights are not returned by the model when expected.

    Notes:
        - The function operates without calculating gradients due to the use of @torch.no_grad().
        - Attention weights are normalized to ensure they sum to 1 (with safety checks for
          invalid values).
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
        try:
            df_out.to_parquet(out_path, index=False)
        except ImportError as exc:
            # Fallback for environments without optional parquet dependencies.
            fallback_csv = out_path.with_suffix(".csv")
            df_out.to_csv(fallback_csv, index=False)
            print(
                "[ATTN][WARN] parquet engine unavailable; wrote CSV instead: "
                f"{fallback_csv} ({exc})"
            )
            out_path = fallback_csv
    else:
        df_out.to_csv(out_path, index=False)

    print(f"[ATTN] wrote {len(df_out)} rows -> {out_path}")
