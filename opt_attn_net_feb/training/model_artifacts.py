from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from ..models.multimodal_mil import MILTaskAttnMixerWithAux


def save_mil_model_artifact(
    *,
    path: str | Path,
    model: MILTaskAttnMixerWithAux,
    best_params: Mapping[str, Any],
    pos_weight: torch.Tensor,
    gamma: torch.Tensor,
    lam: np.ndarray,
    train_info: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "artifact_type": "mil_multimodal_model",
        "artifact_version": 1,
        "model_class": "MILTaskAttnMixerWithAux",
        "init_kwargs": dict(model.hparams),
        "state_dict": {str(k): v.detach().cpu() for k, v in model.state_dict().items()},
        "best_params": dict(best_params),
        "pos_weight": torch.as_tensor(pos_weight, dtype=torch.float32).detach().cpu(),
        "gamma": torch.as_tensor(gamma, dtype=torch.float32).detach().cpu(),
        "lam": np.asarray(lam, dtype=np.float32),
        "train_info": dict(train_info),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, out_path)
    return out_path


def load_mil_model_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[MILTaskAttnMixerWithAux, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    model = MILTaskAttnMixerWithAux(
        **dict(payload["init_kwargs"]),
        pos_weight=torch.as_tensor(payload["pos_weight"], dtype=torch.float32),
        gamma=torch.as_tensor(payload["gamma"], dtype=torch.float32),
        lam=np.asarray(payload["lam"], dtype=np.float32),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, dict(payload)


__all__ = ["save_mil_model_artifact", "load_mil_model_artifact"]
