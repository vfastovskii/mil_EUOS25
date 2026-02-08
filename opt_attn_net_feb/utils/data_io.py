from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .constants import NONFEAT_2D
from .ops import infer_feature_cols


def load_labels(labels_csv: str, id_col: str = "ID") -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if id_col not in df.columns:
        raise ValueError(f"labels missing {id_col}")
    df[id_col] = df[id_col].astype(str)
    return df


def load_2d(feat2d_csv: str, id_col: str = "ID") -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(feat2d_csv)
    if id_col not in df.columns:
        raise ValueError(f"2d features missing {id_col}")
    df[id_col] = df[id_col].astype(str)

    # Logging duplicate IDs and basic stats
    n_rows = len(df)
    n_unique_ids = df[id_col].nunique(dropna=False)
    dups_mask = df.duplicated([id_col], keep=False)
    if dups_mask.any():
        dup_counts = df.groupby(id_col, dropna=False).size()
        dup_counts = dup_counts[dup_counts > 1].sort_values(ascending=False)
        total_extra_rows = int((dup_counts - 1).sum())
        examples = ", ".join([f"{str(i)}×{int(dup_counts.loc[i])}" for i in dup_counts.index[:10]])
        print(f"[DATA-2D] rows={n_rows} unique_ids={n_unique_ids} duplicated_ids={len(dup_counts)} total_extra_rows={total_extra_rows}")
        if examples:
            print(f"[DATA-2D] duplicate ID examples (up to 10): {examples}")
    else:
        print(f"[DATA-2D] rows={n_rows} unique_ids={n_unique_ids} duplicated_ids=0")

    feat_cols = infer_feature_cols(df, NONFEAT_2D)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    ids = df[id_col].astype(str).tolist()
    return ids, X


def align_by_id(ids_file: List[str], X: np.ndarray, ids_target: List[str]) -> np.ndarray:
    id2row = {str(i): r for r, i in enumerate(ids_file)}
    miss = [i for i in ids_target if str(i) not in id2row]
    if miss:
        raise ValueError(f"[2D] Missing {len(miss)} IDs (first 10): {miss[:10]}")
    rows = np.array([id2row[str(i)] for i in ids_target], dtype=np.int64)
    return X[rows]
