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

    This function processes model predictions and per-task attention weights, normalizes
    attention per task over valid conformers, and writes one row per conformer:

    - ID
    - conf_id
    - 4 endpoint predictions (probabilities from logits)
    - 4 attention weights (one per endpoint)

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

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        probs_np = torch.sigmoid(logits).detach().cpu().numpy()  # [B,4]
        attn_np = attn.detach().cpu().numpy()          # [B,4,N]
        kpm_np = kpm.detach().cpu().numpy().astype(bool)
        B, T, N = attn_np.shape
        if T != len(TASK_COLS):
            raise RuntimeError(f"Unexpected attention task count: {T}, expected {len(TASK_COLS)}")

        for b in range(B):
            mid = str(mol_ids[b])
            valid = ~kpm_np[b]
            L = int(valid.sum())
            if L <= 0:
                continue
            confs = [str(x) for x in conf_pad[b, :L].tolist()]

            # Normalize attention for each task over valid conformers.
            attn_norm = np.zeros((T, L), dtype=np.float64)
            for t in range(T):
                w = attn_np[b, t, :L].astype(np.float64)
                s = float(w.sum())
                if not np.isfinite(s) or s <= 0:
                    w[:] = 1.0 / float(L)
                else:
                    w /= s  # enforce sum-to-1 (safety)
                attn_norm[t, :] = w

            pred_cols = {
                f"pred_{TASK_COLS[t]}": float(probs_np[b, t])
                for t in range(T)
            }
            for i in range(L):
                row: Dict[str, Any] = {
                    "ID": mid,
                    "conf_id": confs[i],
                    **pred_cols,
                }
                for t in range(T):
                    row[f"attn_{TASK_COLS[t]}"] = float(attn_norm[t, i])
                rows.append(row)

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
