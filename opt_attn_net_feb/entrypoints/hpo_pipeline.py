from __future__ import annotations

"""
CLI-oriented HPO/training pipeline entrypoint.

This module intentionally contains orchestration only. Public API exports live
in package-level modules (for example `interface.py` and `__init__.py`).
"""

from typing import Any


def _parse_args(argv: Any | None = None):
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--labels", required=True)
    ap.add_argument("--feat2d_scaled", required=True)
    ap.add_argument("--feat3d_scaled", required=True)
    ap.add_argument("--feat3d_qm_scaled", required=True)

    ap.add_argument("--study_dir", required=True)

    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--conf_col", default="conf_id")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--fold_col", default="cv_fold")

    ap.add_argument("--use_splits", nargs="+", default=["train"])  # for HPO subset
    ap.add_argument("--folds", nargs="+", type=int, default=None)

    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--trials", type=int, default=50)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--nn_accelerator", default="gpu")
    ap.add_argument("--nn_devices", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument("--pin_memory", action="store_true")

    # final export
    ap.add_argument("--export_leaderboard_attn", action="store_true")
    ap.add_argument("--leaderboard_split", default="leaderboard")
    ap.add_argument("--attn_out", default=None)

    return ap.parse_args(argv)


def _setup_env_and_meta(args, argv):
    import os, time, json
    from pathlib import Path
    import torch
    from ..utils.ops import set_all_seeds, maybe_set_torch_fast_flags
    from ..utils.constants import WEIGHT_COLS

    set_all_seeds(int(args.seed))
    maybe_set_torch_fast_flags()

    outdir = Path(args.study_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "nn_accelerator": str(args.nn_accelerator),
        "nn_devices": int(args.nn_devices),
        "precision": str(args.precision),
        "patience": int(args.patience),
        "argv": " ".join([str(x) for x in (argv if argv is not None else os.sys.argv)]),
        "weight_cols": WEIGHT_COLS,
        "model": "MILTaskAttnMixerWithAux (task-specific attention queries)",
    }
    (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    # num_workers auto
    if int(args.num_workers) < 0:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
        if cpus <= 0:
            cpus = os.cpu_count() or 0
        num_workers = max(2, min(23, cpus - 2)) if cpus >= 4 else 0
    else:
        num_workers = int(args.num_workers)

    pin_memory = bool(args.pin_memory) and torch.cuda.is_available()

    return outdir, num_workers, pin_memory


def _prepare_hpo_inputs(args):
    import numpy as np
    from ..utils.data_io import load_labels, load_2d, align_by_id
    from ..utils.ops import (
        coerce_binary_labels,
        build_task_weights,
        build_aux_targets_and_masks,
        build_aux_weights,
        fold_indices,
    )
    from ..utils.instances import load_and_merge_instances, build_instance_index

    # Load full labels (needed for final export)
    df_full = load_labels(args.labels, id_col=args.id_col)
    df_full[args.id_col] = df_full[args.id_col].astype(str)
    df_full[args.split_col] = df_full[args.split_col].astype(str)

    # HPO subset
    df_hpo = df_full[df_full[args.split_col].isin(args.use_splits)].copy().reset_index(drop=True)
    if len(df_hpo) == 0:
        raise ValueError(f"No rows in labels match use_splits={args.use_splits}")

    ids_hpo = df_hpo[args.id_col].astype(str).tolist()

    # 2D features
    ids_2d_file, X2d_file = load_2d(args.feat2d_scaled, id_col=args.id_col)
    X2d_hpo = align_by_id(ids_2d_file, X2d_file, ids_hpo)

    y_cls = coerce_binary_labels(df_hpo)
    w_cls = build_task_weights(df_hpo)
    y_abs, m_abs, y_fluo, m_fluo = build_aux_targets_and_masks(df_hpo)
    w_abs, w_fluo = build_aux_weights(df_hpo)

    # folds
    if args.folds is None:
        folds = sorted(df_hpo[args.fold_col].dropna().astype(int).unique().tolist())
    else:
        folds = list(map(int, args.folds))
    folds_info = fold_indices(df_hpo, args.fold_col, folds)

    # instances (for HPO IDs)
    allowed_hpo = set(ids_hpo)
    ids_conf_hpo, conf_ids_hpo, Xinst_hpo = load_and_merge_instances(
        args.feat3d_scaled,
        args.feat3d_qm_scaled,
        allowed_ids=allowed_hpo,
        id_col=args.id_col,
        conf_col=args.conf_col,
    )
    _, starts_hpo, counts_hpo, id2pos_hpo, Xinst_sorted_hpo, conf_sorted_hpo = build_instance_index(
        ids_conf_hpo, conf_ids_hpo, Xinst_hpo
    )

    # drop molecules that ended up with 0 conformers
    have_bag_mask = np.array([(i in id2pos_hpo) for i in ids_hpo], dtype=bool)
    if not have_bag_mask.all():
        missing = int((~have_bag_mask).sum())
        ex = [ids_hpo[i] for i in np.where(~have_bag_mask)[0][:10]]
        print(f"[WARN] Dropping {missing} HPO IDs with 0 conformers after merge. Examples: {ex}")

        df_hpo = df_hpo.loc[have_bag_mask].reset_index(drop=True)
        ids_hpo = df_hpo[args.id_col].astype(str).tolist()

        X2d_hpo = X2d_hpo[have_bag_mask]
        y_cls = y_cls[have_bag_mask]
        w_cls = w_cls[have_bag_mask]
        y_abs = y_abs[have_bag_mask]
        m_abs = m_abs[have_bag_mask]
        y_fluo = y_fluo[have_bag_mask]
        m_fluo = m_fluo[have_bag_mask]
        w_abs = w_abs[have_bag_mask]
        w_fluo = w_fluo[have_bag_mask]

        folds = sorted(df_hpo[args.fold_col].dropna().astype(int).unique().tolist())
        folds_info = fold_indices(df_hpo, args.fold_col, folds)

    print(f"[DATA-HPO] n_ids={len(ids_hpo)} | X2d_dim={X2d_hpo.shape[1]}")
    print(f"[DATA-HPO] n_conf={Xinst_sorted_hpo.shape[0]} | inst_dim={Xinst_sorted_hpo.shape[1]}")

    return {
        "df_full": df_full,
        "ids_2d_file": ids_2d_file,
        "X2d_file": X2d_file,
        "X2d_hpo": X2d_hpo,
        "y_cls": y_cls,
        "w_cls": w_cls,
        "y_abs": y_abs,
        "m_abs": m_abs,
        "w_abs": w_abs,
        "y_fluo": y_fluo,
        "m_fluo": m_fluo,
        "w_fluo": w_fluo,
        "ids_hpo": ids_hpo,
        "folds_info": folds_info,
        "starts_hpo": starts_hpo,
        "counts_hpo": counts_hpo,
        "id2pos_hpo": id2pos_hpo,
        "Xinst_sorted_hpo": Xinst_sorted_hpo,
        "conf_sorted_hpo": conf_sorted_hpo,
    }


def _run_hpo(args, outdir, num_workers, pin_memory, hpo):
    import optuna
    from ..training.hpo import objective_mil_cv, save_study_artifacts, save_best_fold_metrics

    print(f"[DATALOADER] num_workers={num_workers} pin_memory={pin_memory} precision={args.precision}")

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

    def make_storage(name: str) -> str:
        from pathlib import Path as _P
        return f"sqlite:///{(_P(outdir) / f'{name}.sqlite3').as_posix()}"

    ckpt_root = outdir / "_tmp_best_ckpts"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    study_name = "mil_task_attn_mixer_aux_gpu"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=make_storage(study_name),
        load_if_exists=True,
    )
    study.optimize(
        lambda tr: objective_mil_cv(
            tr,
            X2d_scaled=hpo["X2d_hpo"],
            y_cls=hpo["y_cls"], w_cls=hpo["w_cls"],
            y_abs=hpo["y_abs"], m_abs=hpo["m_abs"], w_abs=hpo["w_abs"],
            y_fluo=hpo["y_fluo"], m_fluo=hpo["m_fluo"], w_fluo=hpo["w_fluo"],
            ids=hpo["ids_hpo"],
            folds_info=hpo["folds_info"],
            starts=hpo["starts_hpo"], counts=hpo["counts_hpo"], id2pos=hpo["id2pos_hpo"], Xinst_sorted=hpo["Xinst_sorted_hpo"],
            seed=int(args.seed),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            accelerator=str(args.nn_accelerator),
            devices=int(args.nn_devices),
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            precision=str(args.precision),
            ckpt_root=ckpt_root,
        ),
        n_trials=int(args.trials),
        gc_after_trial=True,
        catch=(RuntimeError, ValueError, FloatingPointError),
    )

    save_study_artifacts(outdir, study, prefix=study_name)
    save_best_fold_metrics(outdir, study_name, study.best_trial.user_attrs.get("fold_detail", {}))
    print(f"[HPO] best macro AP (CV mean) = {study.best_value:.6f}")

    return study, ckpt_root


def _run_final_with_best_if_needed(args, outdir, study, hpo, num_workers, pin_memory):
    if not args.export_leaderboard_attn:
        return
    from ..utils.instances import load_and_merge_instances, build_instance_index
    from ..training.hpo import train_best_and_export

    df_full = hpo["df_full"]

    # final export step needs instance index for union(train+leaderboard) not just HPO IDs
    allowed_final = set(
        df_full[df_full[args.split_col].isin(["train", args.leaderboard_split])][args.id_col]
        .astype(str)
        .tolist()
    )
    ids_conf_all, conf_ids_all, Xinst_all = load_and_merge_instances(
        args.feat3d_scaled,
        args.feat3d_qm_scaled,
        allowed_ids=allowed_final,
        id_col=args.id_col,
        conf_col=args.conf_col,
    )
    _, starts_all, counts_all, id2pos_all, Xinst_sorted_all, conf_sorted_all = build_instance_index(
        ids_conf_all, conf_ids_all, Xinst_all
    )

    train_best_and_export(
        outdir=outdir,
        df_full=df_full.rename(columns={args.id_col: "ID"}),
        best_params=dict(study.best_params),
        args=args,
        X2d_file_ids=hpo["ids_2d_file"],
        X2d_file=hpo["X2d_file"],
        starts=starts_all,
        counts=counts_all,
        id2pos=id2pos_all,
        Xinst_sorted=Xinst_sorted_all,
        conf_sorted=conf_sorted_all,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def main(argv: Any | None = None) -> None:
    """Short and clear entrypoint:
    1) read input params; 2) build model inputs; 3) optimize hyperparams; 4) run with best params.
    """
    import shutil
    import torch

    # 1) Read input params + set up environment
    args = _parse_args(argv)
    outdir, num_workers, pin_memory = _setup_env_and_meta(args, argv)

    # 2) Build model inputs (data + indices). The model class is MILTaskAttnMixerWithAux.
    hpo = _prepare_hpo_inputs(args)

    # 3) Optimize hyperparameters (CV with Optuna)
    study, ckpt_root = _run_hpo(args, outdir, num_workers, pin_memory, hpo)

    # 4) Run model with best params and optionally export leaderboard attention
    _run_final_with_best_if_needed(args, outdir, study, hpo, num_workers, pin_memory)

    # cleanup
    try:
        shutil.rmtree(ckpt_root, ignore_errors=True)
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc as _gc
    _gc.collect()


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
