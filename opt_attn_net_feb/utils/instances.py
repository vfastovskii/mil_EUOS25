from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .constants import NONFEAT_3D, NONFEAT_QM
from .ops import infer_feature_cols


def load_and_merge_instances(
    geom_csv: str,
    qm_csv: str,
    allowed_ids: Optional[Set[str]],
    id_col: str = "ID",
    conf_col: str = "conf_id",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dg = pd.read_csv(geom_csv)
    dq = pd.read_csv(qm_csv)

    for d, name in [(dg, "3d_scaled"), (dq, "3d_qm_scaled")]:
        for c in [id_col, conf_col]:
            if c not in d.columns:
                raise ValueError(f"{name} missing column {c}")

    dg[id_col] = dg[id_col].astype(str)
    dg[conf_col] = dg[conf_col].astype(str)
    dq[id_col] = dq[id_col].astype(str)
    dq[conf_col] = dq[conf_col].astype(str)

    if allowed_ids is not None:
        dg = dg[dg[id_col].isin(allowed_ids)].copy()
        dq = dq[dq[id_col].isin(allowed_ids)].copy()

    if "error" in dq.columns:
        dq = dq[dq["error"].isna() | (dq["error"].astype(str).str.len() == 0)].copy()
    if "status" in dq.columns:
        s = dq["status"].astype(str).str.lower()
        ok = s.isin(["ok", "success", "0", "1", "true"]) | dq["status"].isna()
        dq = dq[ok].copy()

    # Logging: uniqueness and duplicates in 3D geometry and quantum tables
    # Geometry
    n_rows_g = len(dg)
    n_unique_ids_g = dg[id_col].nunique(dropna=False)
    n_unique_pairs_g = dg[[id_col, conf_col]].drop_duplicates().shape[0]
    grp_g = dg.groupby([id_col, conf_col], dropna=False).size()
    dup_g = grp_g[grp_g > 1].sort_values(ascending=False)
    if len(dup_g) > 0:
        total_extra_rows_g = int((dup_g - 1).sum())
        dup_ids_g = pd.Index([k[0] for k in dup_g.index]).nunique()
        ex_g = ", ".join([f"{str(k[0])}|{str(k[1])}×{int(v)}" for k, v in dup_g.head(10).items()])
        print(f"[DATA-3D-GEOM] rows={n_rows_g} unique_ids={n_unique_ids_g} unique_pairs={n_unique_pairs_g} duplicated_pairs={len(dup_g)} duplicated_ids={dup_ids_g} total_extra_rows={total_extra_rows_g}")
        if ex_g:
            print(f"[DATA-3D-GEOM] duplicate (ID,conf) examples (up to 10): {ex_g}")
    else:
        print(f"[DATA-3D-GEOM] rows={n_rows_g} unique_ids={n_unique_ids_g} unique_pairs={n_unique_pairs_g} duplicated_pairs=0")

    # Quantum
    n_rows_q = len(dq)
    n_unique_ids_q = dq[id_col].nunique(dropna=False)
    n_unique_pairs_q = dq[[id_col, conf_col]].drop_duplicates().shape[0]
    grp_q = dq.groupby([id_col, conf_col], dropna=False).size()
    dup_q = grp_q[grp_q > 1].sort_values(ascending=False)
    if len(dup_q) > 0:
        total_extra_rows_q = int((dup_q - 1).sum())
        dup_ids_q = pd.Index([k[0] for k in dup_q.index]).nunique()
        ex_q = ", ".join([f"{str(k[0])}|{str(k[1])}×{int(v)}" for k, v in dup_q.head(10).items()])
        print(f"[DATA-3D-QM] rows={n_rows_q} unique_ids={n_unique_ids_q} unique_pairs={n_unique_pairs_q} duplicated_pairs={len(dup_q)} duplicated_ids={dup_ids_q} total_extra_rows={total_extra_rows_q}")
        if ex_q:
            print(f"[DATA-3D-QM] duplicate (ID,conf) examples (up to 10): {ex_q}")
    else:
        print(f"[DATA-3D-QM] rows={n_rows_q} unique_ids={n_unique_ids_q} unique_pairs={n_unique_pairs_q} duplicated_pairs=0")

    g_cols = infer_feature_cols(dg, NONFEAT_3D)
    q_cols = infer_feature_cols(dq, NONFEAT_QM)

    # Deduplicate potential repeated (ID, conf_id) keys by averaging features
    dg_sub = dg[[id_col, conf_col] + g_cols].copy()
    dq_sub = dq[[id_col, conf_col] + q_cols].copy()

    if dg_sub.duplicated([id_col, conf_col]).any():
        dg_sub = dg_sub.groupby([id_col, conf_col], as_index=False)[g_cols].mean()
    if dq_sub.duplicated([id_col, conf_col]).any():
        dq_sub = dq_sub.groupby([id_col, conf_col], as_index=False)[q_cols].mean()

    m = dg_sub.merge(
        dq_sub,
        on=[id_col, conf_col],
        how="inner",
        validate="one_to_one",
    )

    ids_conf = m[id_col].astype(str).to_numpy()
    conf_ids = m[conf_col].astype(str).to_numpy()
    Xg = m[g_cols].to_numpy(dtype=np.float32)
    Xq = m[q_cols].to_numpy(dtype=np.float32)
    X = np.hstack([Xg, Xq]).astype(np.float32)
    return ids_conf, conf_ids, X


def build_instance_index(
    ids_conf: np.ndarray,
    conf_ids: np.ndarray,
    Xinst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], np.ndarray, np.ndarray]:
    # stable sort by ID to pack each bag contiguously
    order = np.argsort(ids_conf, kind="mergesort")
    ids_sorted = ids_conf[order]
    X_sorted = Xinst[order]
    conf_sorted = conf_ids[order]

    uniq, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    id2pos = {str(u): i for i, u in enumerate(uniq)}
    return uniq, starts.astype(np.int64), counts.astype(np.int64), id2pos, X_sorted, conf_sorted
