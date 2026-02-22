from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from ..explainability.lambda_vol import LambdaVolConfig
    from ..explainability.lambda_vol.analytics import LambdaVolQueryService
    from ..explainability.lambda_vol.config import ExportConfig, StoreConfig
    from ..explainability.lambda_vol.monitor import LambdaVolMonitor
except Exception:  # pragma: no cover
    from explainability.lambda_vol import LambdaVolConfig
    from explainability.lambda_vol.analytics import LambdaVolQueryService
    from explainability.lambda_vol.config import ExportConfig, StoreConfig
    from explainability.lambda_vol.monitor import LambdaVolMonitor

LOGGER = logging.getLogger("lambda_vol_demo")


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=-1, keepdims=True)


def _build_epoch_frames(
    *,
    epoch: int,
    task_ids: Sequence[str],
    concept_ids: Sequence[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, float]]:
    n_tasks = len(task_ids)
    n_concepts = len(concept_ids)

    base_tcav = rng.normal(loc=0.08, scale=0.02, size=(n_tasks, n_concepts)).astype(np.float32)
    tcav = base_tcav.copy()

    for ti in range(n_tasks):
        dominant = int((3 * ti + 1) % n_concepts)
        tcav[ti, dominant] += float(0.02 * epoch)

    if n_concepts > 0:
        tcav[0, 0] += float(max(0, epoch - 4) * 0.015)

    tcav += rng.normal(loc=0.0, scale=0.01, size=tcav.shape).astype(np.float32)
    tcav = np.clip(tcav, -1.0, 1.0)

    logits = rng.normal(loc=0.0, scale=0.5, size=(n_tasks, n_concepts)).astype(np.float32)
    for ti in range(n_tasks):
        dominant = int((3 * ti + 1) % n_concepts)
        logits[ti, dominant] += float(0.7 + 0.16 * epoch)
    if n_concepts > 0:
        logits[0, 0] += float(max(0, epoch - 4) * 0.35)

    prevalence = _softmax(logits).astype(np.float32)

    attention_support = np.clip(
        0.25 + 0.6 * np.maximum(tcav, 0.0) + 0.25 * prevalence + rng.normal(0.0, 0.02, size=tcav.shape),
        0.0,
        1.0,
    ).astype(np.float32)

    tcav_rows = []
    concept_rows = []
    for ti, task_id in enumerate(task_ids):
        for ci, concept_id in enumerate(concept_ids):
            tcav_rows.append(
                {
                    "task_id": str(task_id),
                    "concept_id": str(concept_id),
                    "tcav": float(tcav[ti, ci]),
                }
            )
            concept_rows.append(
                {
                    "task_id": str(task_id),
                    "concept_id": str(concept_id),
                    "attention_support": float(attention_support[ti, ci]),
                    "prevalence": float(prevalence[ti, ci]),
                }
            )

    task_attention_rows = []
    task_metrics_rows = []
    for ti, task_id in enumerate(task_ids):
        p = np.clip(prevalence[ti], 1e-12, 1.0)
        entropy = float(-np.sum(p * np.log(p)) / max(np.log(len(p)), 1e-12))
        witness_rate = float(np.mean(np.sort(attention_support[ti])[-max(1, n_concepts // 4) :]))

        train_metric = float(0.18 + 0.030 * epoch + rng.normal(0.0, 0.002))
        val_metric = float(0.17 + 0.024 * min(epoch, 7) - 0.007 * max(0, epoch - 8) + rng.normal(0.0, 0.0025))
        loss = float(max(0.03, 1.15 - 0.085 * epoch + 0.015 * max(0, epoch - 10) + rng.normal(0.0, 0.01)))
        calibration_error = float(0.20 - 0.012 * min(epoch, 6) + 0.009 * max(0, epoch - 8) + rng.normal(0.0, 0.003))

        task_attention_rows.append(
            {
                "task_id": str(task_id),
                "attention_entropy": entropy,
                "witness_rate": witness_rate,
            }
        )
        task_metrics_rows.append(
            {
                "task_id": str(task_id),
                "train_metric": train_metric,
                "val_metric": val_metric,
                "loss": loss,
                "calibration_error": calibration_error,
            }
        )

    context = {
        "val_macro": float(np.mean([x["val_metric"] for x in task_metrics_rows])),
        "train_macro": float(np.mean([x["train_metric"] for x in task_metrics_rows])),
        "loss_macro": float(np.mean([x["loss"] for x in task_metrics_rows])),
        "attention_entropy_macro": float(np.mean([x["attention_entropy"] for x in task_attention_rows])),
    }

    return (
        pd.DataFrame(tcav_rows),
        pd.DataFrame(concept_rows),
        pd.DataFrame(task_attention_rows),
        pd.DataFrame(task_metrics_rows),
        context,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lambda-Vol concept-pressure demo")
    ap.add_argument("--output_dir", default="lambda_vol_demo_out")
    ap.add_argument("--db_uri", default=None)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--num_tasks", type=int, default=4)
    ap.add_argument("--num_concepts", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run_name", default="lambda_vol_demo")
    ap.add_argument("--blocked_concepts", nargs="*", default=["concept_00"])
    ap.add_argument("--no_plotly", action="store_true")
    ap.add_argument("--no_parquet", action="store_true")
    ap.add_argument("--export_vtk", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [f"task_{i}" for i in range(int(args.num_tasks))]
    concept_ids = [f"concept_{i:02d}" for i in range(int(args.num_concepts))]

    concept_metadata = {
        cid: {
            "family": "aromatic" if (i % 3 == 0) else ("pharmacophore" if (i % 3 == 1) else "charge"),
            "name": cid,
        }
        for i, cid in enumerate(concept_ids)
    }

    db_uri = args.db_uri or f"sqlite:///{(out_dir / 'lambda_vol.sqlite3').as_posix()}"
    cfg = LambdaVolConfig(
        run_name=str(args.run_name),
        seed=int(args.seed),
        blocked_concepts=tuple(str(x) for x in args.blocked_concepts),
        exporter=ExportConfig(
            output_dir=str(out_dir),
            export_parquet=(not bool(args.no_parquet)),
            export_plotly_html=(not bool(args.no_plotly)),
            export_vtk=bool(args.export_vtk),
            top_k_lattice=min(20, len(concept_ids)),
        ),
        store=StoreConfig(db_uri=db_uri),
    )

    monitor = LambdaVolMonitor(
        config=cfg,
        task_ids=task_ids,
        concept_ids=concept_ids,
        concept_metadata=concept_metadata,
    )

    rng = np.random.default_rng(int(args.seed))
    epoch_logs: list[dict[str, object]] = []

    for epoch in range(int(args.epochs)):
        tcav_df, concept_attn_df, task_attn_df, task_metrics_df, context = _build_epoch_frames(
            epoch=epoch,
            task_ids=task_ids,
            concept_ids=concept_ids,
            rng=rng,
        )
        result = monitor.step_from_frames(
            epoch=epoch,
            tcav_df=tcav_df,
            concept_attention_df=concept_attn_df,
            task_attention_df=task_attn_df,
            task_metrics_df=task_metrics_df,
            context_covariates=context,
        )
        epoch_logs.append(
            {
                "epoch": int(result.epoch),
                "regime": result.regime_label.value,
                "n_alerts": int(result.n_alerts),
                "n_recommendations": int(result.n_recommendations),
                "rho_mean": float(result.rho_mean),
                "rho_max": float(result.rho_max),
            }
        )

    artifacts = monitor.finalize()

    query_service = LambdaVolQueryService(repository=monitor.repository)
    query_payload = {
        "high_tcav_low_prevalence": query_service.high_tcav_low_prevalence(
            run_id=monitor.run_id,
            task_id=task_ids[0],
            epoch=max(0, int(args.epochs) - 1),
            tcav_thr=0.12,
            prevalence_thr=0.06,
        ),
        "top_trend_loops": query_service.top_trend_loops(
            run_id=monitor.run_id,
            last_n_epochs=min(8, int(args.epochs)),
            limit=10,
        ),
        "recommendations": query_service.recommendations(
            run_id=monitor.run_id,
            last_n_epochs=min(8, int(args.epochs)),
        ),
    }

    summary = {
        "run_id": monitor.run_id,
        "db_uri": db_uri,
        "task_ids": task_ids,
        "concept_ids": concept_ids,
        "epochs": int(args.epochs),
        "blocked_concepts": list(cfg.blocked_concepts),
        "artifacts": {
            "tensor_npz": artifacts.tensor_npz,
            "long_csv": artifacts.long_csv,
            "long_parquet": artifacts.long_parquet,
            "metadata_json": artifacts.metadata_json,
            "manifold_html_by_task": dict(artifacts.manifold_html_by_task),
            "lattice_html": artifacts.lattice_html,
            "coupling_html": artifacts.coupling_html,
            "alerts_json": artifacts.alerts_json,
            "alerts_md": artifacts.alerts_md,
            "recommendations_json": artifacts.recommendations_json,
            "vtk_path": artifacts.vtk_path,
        },
        "epoch_logs": epoch_logs,
        "queries": query_payload,
    }

    summary_path = out_dir / "lambda_vol_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOGGER.info("Lambda-Vol demo complete. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
