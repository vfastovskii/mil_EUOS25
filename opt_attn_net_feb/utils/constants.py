from __future__ import annotations

# Task and auxiliary target columns
TASK_COLS = [
    "Transmittance_340",
    "Transmittance_450",
    "Fluorescence_340_450",
    "Fluorescence_more_than_480",
]

AUX_ABS_COLS = [
    "Transmittance_340_quantitative",
    "Transmittance_450_quantitative",
]

AUX_FLUO_BASE_COLS = ["wl_pred_nm", "qy_pred"]

# Sample weight columns mapping per task index
WEIGHT_COLS = {
    0: "sample_weight_340",
    1: "sample_weight_450",
    2: "w_ad",
    3: "w_ad",
}

# Non-feature columns per modality
NONFEAT_2D = {"ID", "curated_SMILES", "split"}
NONFEAT_3D = {"ID", "conf_id", "smiles", "split"}
NONFEAT_QM = {"record_index", "ID", "conf_id", "status", "error", "split"}
