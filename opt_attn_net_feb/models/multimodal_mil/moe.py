from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .embedder_mlp_v3_base import build_mlp_v3_embedder


def _make_activation_module(name: str) -> nn.Module:
    n = str(name).strip().lower()
    if n == "gelu":
        return nn.GELU()
    if n in {"relu", "leakyrelu", "leaky_relu"}:
        return nn.ReLU()
    return nn.SiLU()


class TaskAwareMoEMixer(nn.Module):
    """
    Dense top-k MoE mixer used after modality fusion.

    The router is task-aware through a learned task embedding and always keeps
    a shared expert available when `use_shared_expert=True`.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        activation: str,
        num_tasks: int,
        gate_dim: int = 0,
        num_experts: int = 4,
        top_k: int = 2,
        use_shared_expert: bool = True,
        router_hidden: Optional[int] = None,
        load_balance_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        router_entropy_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tasks = int(max(1, num_tasks))
        self.gate_dim = int(max(0, gate_dim))
        self.num_experts = int(max(1, num_experts))
        self.top_k = int(max(1, min(int(top_k), self.num_experts)))
        self.use_shared_expert = bool(use_shared_expert)
        self.load_balance_weight = float(max(0.0, load_balance_weight))
        self.z_loss_weight = float(max(0.0, z_loss_weight))
        self.router_entropy_weight = float(max(0.0, router_entropy_weight))

        self.task_embed_dim = int(max(8, min(32, self.hidden_dim // 4)))
        self.task_embeddings = nn.Embedding(self.num_tasks, self.task_embed_dim)
        router_in_dim = self.input_dim + self.task_embed_dim + self.gate_dim
        router_hidden_dim = (
            int(router_hidden)
            if router_hidden is not None
            else int(max(64, min(self.input_dim, self.hidden_dim)))
        )

        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden_dim),
            _make_activation_module(str(activation)),
            nn.Dropout(float(dropout)),
            nn.Linear(router_hidden_dim, self.num_experts),
        )
        self.shared_gate = (
            nn.Sequential(
                nn.Linear(router_in_dim, router_hidden_dim),
                _make_activation_module(str(activation)),
                nn.Dropout(float(dropout)),
                nn.Linear(router_hidden_dim, 1),
            )
            if self.use_shared_expert
            else None
        )
        self.shared_expert = (
            build_mlp_v3_embedder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                layers=int(layers),
                dropout=float(dropout),
                activation=str(activation),
            )
            if self.use_shared_expert
            else None
        )
        self.experts = nn.ModuleList(
            [
                build_mlp_v3_embedder(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    layers=int(layers),
                    dropout=float(dropout),
                    activation=str(activation),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        task_index: torch.Tensor,
        router_context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        if x.ndim != 2:
            raise ValueError(f"Expected x [N,D], got {tuple(x.shape)}")
        if task_index.ndim != 1 or task_index.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected task_index [N] matching x rows, got {tuple(task_index.shape)} for x={tuple(x.shape)}"
            )

        task_idx = task_index.long().clamp(min=0, max=self.num_tasks - 1)
        router_parts = [x, self.task_embeddings(task_idx)]
        if self.gate_dim > 0:
            if router_context is None:
                router_context = torch.zeros((x.shape[0], self.gate_dim), dtype=x.dtype, device=x.device)
            if router_context.ndim != 2 or router_context.shape != (x.shape[0], self.gate_dim):
                raise ValueError(
                    "router_context must have shape "
                    f"[N,{self.gate_dim}] but got {tuple(router_context.shape)} for x={tuple(x.shape)}"
                )
            router_parts.append(router_context.to(dtype=x.dtype, device=x.device))
        router_in = torch.cat(router_parts, dim=-1)
        router_logits = self.router(router_in)
        router_probs = torch.softmax(router_logits, dim=-1)

        if self.top_k < self.num_experts:
            top_logits, top_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
            dispatch_mask = torch.zeros_like(router_probs).scatter_(1, top_idx, 1.0)
        else:
            top_logits = router_logits
            top_idx = torch.arange(
                self.num_experts,
                device=x.device,
                dtype=torch.long,
            ).view(1, self.num_experts).expand(x.shape[0], -1)
            dispatch_mask = torch.ones_like(router_probs)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        if self.use_shared_expert and self.shared_expert is not None and self.shared_gate is not None:
            shared_logits = self.shared_gate(router_in)
            combine_logits = torch.cat([shared_logits, top_logits], dim=-1)
            combine_weights = torch.softmax(combine_logits, dim=-1)
            shared_weight = combine_weights[:, :1]
            top_weights = combine_weights[:, 1:]
            expert_weight_sparse = torch.zeros_like(router_probs).scatter_(1, top_idx, top_weights)
            shared_output = self.shared_expert(x)
            all_expert_weights = torch.cat([shared_weight, expert_weight_sparse], dim=1)
            output = (
                shared_weight * shared_output
                + (expert_outputs * expert_weight_sparse.unsqueeze(-1)).sum(dim=1)
            )
        else:
            shared_weight = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            if self.top_k < self.num_experts:
                top_weights = torch.softmax(top_logits, dim=-1)
                expert_weight_sparse = torch.zeros_like(router_probs).scatter_(1, top_idx, top_weights)
            else:
                expert_weight_sparse = router_probs
            all_expert_weights = expert_weight_sparse
            output = (expert_outputs * expert_weight_sparse.unsqueeze(-1)).sum(dim=1)

        expert_importance = all_expert_weights.mean(dim=0)
        expert_load = expert_importance
        n_balance_units = int(all_expert_weights.shape[1])
        balance_raw = float(n_balance_units) * torch.sum(expert_importance.pow(2)) - 1.0
        load_balance_loss = float(self.load_balance_weight) * torch.clamp(balance_raw, min=0.0)
        z_loss = float(self.z_loss_weight) * torch.mean(torch.logsumexp(router_logits, dim=-1).pow(2))
        router_entropy = -(router_probs.clamp_min(1e-8) * router_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        entropy_bonus = float(self.router_entropy_weight) * router_entropy
        router_aux_loss = load_balance_loss + z_loss - entropy_bonus

        info: Dict[str, Any] = {
            "router_logits": router_logits,
            "router_probs": router_probs,
            "top_indices": top_idx,
            "expert_importance": expert_importance,
            "expert_load": expert_load,
            "expert_weight_sparse": expert_weight_sparse,
            "shared_weight": shared_weight.squeeze(-1),
            "all_expert_weights": all_expert_weights,
            "private_dispatch_rate": dispatch_mask.mean(dim=0),
            "load_balance_loss": load_balance_loss,
            "z_loss": z_loss,
            "router_entropy": router_entropy,
            "router_entropy_bonus": entropy_bonus,
            "router_aux_loss": router_aux_loss,
        }
        return output, info


__all__ = ["TaskAwareMoEMixer"]
