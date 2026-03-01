#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import optuna
from optuna.trial import Trial
from catboost import CatBoostClassifier


TASK_COLS = [
    "Transmittance_340",
    "Transmittance_450",
    "Fluorescence_340_450",
    "Fluorescence_more_than_480",
]

# Only tasks 0 and 1 have external weights; tasks 2/3 have none.
WEIGHT_COLS = {
    0: "sample_weight_340",
    1: "sample_weight_450",
    2: None,
    3: None,
}

NONFEAT_2D = {"ID", "curated_SMILES", "split"}


# metrics
def ap_score(
    y_true_1d: np.ndarray,
    y_score_1d: np.ndarray,
    sample_weight_1d: Optional[np.ndarray] = None,
) -> float:
    """Average precision with optional sample_weight"""
    y_true_1d = y_true_1d.astype(int)
    if y_true_1d.sum() == 0:
        return 0.0
    y_score_1d = np.nan_to_num(y_score_1d, nan=0.0, posinf=1.0, neginf=0.0)
    y_score_1d = np.clip(y_score_1d, 0.0, 1.0)
    if sample_weight_1d is not None:
        return float(average_precision_score(y_true_1d, y_score_1d, sample_weight=sample_weight_1d))
    return float(average_precision_score(y_true_1d, y_score_1d))


def macro_ap(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_val: Optional[np.ndarray] = None,
) -> float:
    """
    Macro-average AP across 4 tasks.
    If w_val is provided, it's (n,4). In this script only tasks 0/1 are non-trivial.
    Passing all-ones for tasks 2/3 is equivalent to unweighted.
    """
    aps: List[float] = []
    for t in range(4):
        wt = None
        if w_val is not None:
            wt = w_val[:, t]
            wt = np.where(np.isfinite(wt), wt, 1.0).astype(float)
            wt = np.clip(wt, 0.0, np.inf)
        aps.append(ap_score(y_true[:, t], p_pred[:, t], sample_weight_1d=wt))
    return float(np.mean(aps))


def coerce_binary_labels(df: pd.DataFrame) -> np.ndarray:
    y = df[TASK_COLS].fillna(0).astype(int).to_numpy()
    return (y > 0).astype(np.int64)


def infer_feature_cols(df: pd.DataFrame, nonfeat: set) -> List[str]:
    return sorted([c for c in df.columns if c not in nonfeat])


def load_labels(labels_csv: str, id_col="ID") -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if id_col not in df.columns:
        raise ValueError(f"labels missing {id_col}")
    df[id_col] = df[id_col].astype(str)
    return df


def load_2d(feat2d_csv: str, id_col="ID") -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(feat2d_csv)
    if id_col not in df.columns:
        raise ValueError(f"2d features missing {id_col}")
    df[id_col] = df[id_col].astype(str)
    feat_cols = infer_feature_cols(df, NONFEAT_2D)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    ids = df[id_col].astype(str).tolist()
    return ids, X


def align_by_id(ids_file: List[str], X: np.ndarray, ids_target: List[str]) -> np.ndarray:
    id2row = {str(i): r for r, i in enumerate(ids_file)}
    miss = [i for i in ids_target if str(i) not in id2row]
    if miss:
        raise ValueError(f"[2D RAW] Missing {len(miss)} IDs (first 10): {miss[:10]}")
    rows = np.array([id2row[str(i)] for i in ids_target], dtype=np.int64)
    return X[rows]


def fold_indices(df: pd.DataFrame, fold_col: str, folds: List[int]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    out = []
    arr = df[fold_col].astype(int).to_numpy()
    for f in folds:
        va = np.where(arr == f)[0]
        tr = np.where(arr != f)[0]
        if len(va) == 0:
            raise ValueError(f"Fold {f} has 0 samples.")
        out.append((tr, va, f))
    return out


# optuna space
def search_space_catboost(trial: Trial) -> Dict[str, Any]:
    p = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        "pos_weight_clip": trial.suggest_float("pos_weight_clip", 30.0, 200.0, log=True),
    }
    if p["bootstrap_type"] == "Bayesian":
        p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    else:
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return p


# save helpers
def save_study_artifacts(outdir: Path, study: optuna.Study, prefix: str):
    df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df_trials.to_csv(outdir / f"{prefix}_trials.csv", index=False)

    best = dict(study.best_params)
    best["best_value"] = float(study.best_value)
    (outdir / f"{prefix}_best_params.json").write_text(json.dumps(best, indent=2))


def save_best_fold_metrics(outdir: Path, prefix: str, fold_metrics: Dict[str, Any]):
    (outdir / f"{prefix}_best_fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))


# catboost helpers
def _catboost_common_params(
    params: Dict[str, Any],
    seed: int,
    thread_count: int,
    scale_pos_weight: float,
) -> Dict[str, Any]:
    cb = dict(
        iterations=4000,
        loss_function="Logloss",
        eval_metric="PRAUC",
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        min_data_in_leaf=params["min_data_in_leaf"],
        random_strength=params["random_strength"],
        rsm=params["rsm"],
        bootstrap_type=params["bootstrap_type"],
        random_seed=int(seed),
        scale_pos_weight=float(scale_pos_weight),
        verbose=False,
        task_type="CPU",
        thread_count=int(thread_count),
        allow_writing_files=False,
    )
    if "bagging_temperature" in params:
        cb["bagging_temperature"] = params["bagging_temperature"]
    if "subsample" in params:
        cb["subsample"] = params["subsample"]
    return cb


def _get_weights_for_task(df_lab: pd.DataFrame, idx: np.ndarray, task_idx: int) -> Optional[np.ndarray]:
    w_col = WEIGHT_COLS.get(task_idx, None)
    if not w_col:
        return None
    if w_col not in df_lab.columns:
        return None
    w = df_lab.iloc[idx][w_col].astype(float).fillna(1.0).to_numpy()
    return np.clip(w, 0.0, np.inf)


# objectives
def objective_catboost_task_cv(
    trial: Trial,
    task_idx: int,
    X2d_raw: np.ndarray,
    df_lab: pd.DataFrame,
    y_cls: np.ndarray,
    folds_info,
    seed: int,
    cpu_threads: int,
) -> float:
    params = search_space_catboost(trial)
    scores: List[float] = []
    fold_detail: Dict[str, Any] = {}

    for step, (tr, va, f) in enumerate(folds_info):
        pos = float(y_cls[tr, task_idx].sum())
        neg = float(len(tr) - pos)
        spw = min(neg / max(pos, 1.0), float(params["pos_weight_clip"]))

        w_train = _get_weights_for_task(df_lab, tr, task_idx)
        w_val = _get_weights_for_task(df_lab, va, task_idx)  # used only for weighted AP (0/1)

        model = CatBoostClassifier(
            **_catboost_common_params(params, seed=seed + 10000 * f + 97 * task_idx, thread_count=cpu_threads, scale_pos_weight=spw)
        )
        model.fit(
            X2d_raw[tr], y_cls[tr, task_idx],
            sample_weight=w_train,
            eval_set=(X2d_raw[va], y_cls[va, task_idx]),
            use_best_model=True,
            early_stopping_rounds=200,
            verbose=False,
        )

        pred = model.predict_proba(X2d_raw[va])[:, 1]

        ap = ap_score(y_cls[va, task_idx], pred, sample_weight_1d=w_val if task_idx in (0, 1) else None)
        scores.append(ap)

        best_iter = None
        try:
            best_iter = int(model.get_best_iteration())
        except Exception:
            best_iter = None

        fold_detail[str(f)] = {
            "task": TASK_COLS[task_idx],
            "ap": float(ap),
            "ap_weighted": bool(task_idx in (0, 1)),
            "scale_pos_weight": float(spw),
            "best_iteration": best_iter,
            "weight_col_used": WEIGHT_COLS.get(task_idx, None),
            "threads": int(cpu_threads),
        }

        trial.report(float(np.mean(scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

        del model, pred
        gc.collect()

    trial.set_user_attr("task_idx", int(task_idx))
    trial.set_user_attr("task_name", TASK_COLS[task_idx])
    trial.set_user_attr("fold_detail", fold_detail)
    return float(np.mean(scores))


def objective_catboost_macro_cv(
    trial: Trial,
    X2d_raw: np.ndarray,
    df_lab: pd.DataFrame,
    y_cls: np.ndarray,
    folds_info,
    seed: int,
    cpu_threads: int,
    parallel_tasks: bool,
    task_jobs: int,
) -> float:
    """
    Single Optuna study, one param set for all tasks.
    Within each fold: train 4 task-models in parallel (optional).
    """
    params = search_space_catboost(trial)
    scores: List[float] = []
    fold_detail: Dict[str, Any] = {}

    jobs = int(max(1, min(4, task_jobs)))
    threads_per_model = max(1, int(cpu_threads) // jobs)

    for step, (tr, va, f) in enumerate(folds_info):
        preds = np.zeros((len(va), 4), dtype=float)

        # validation weight matrix: only tasks 0/1 have non-trivial weights; tasks 2/3 are ones
        w_val_mat = np.ones((len(va), 4), dtype=float)
        for t in (0, 1):
            wv = _get_weights_for_task(df_lab, va, t)
            if wv is not None:
                w_val_mat[:, t] = wv

        def train_one_task(t: int) -> Tuple[int, np.ndarray, float, float, Optional[int]]:
            pos = float(y_cls[tr, t].sum())
            neg = float(len(tr) - pos)
            spw = min(neg / max(pos, 1.0), float(params["pos_weight_clip"]))

            w_train = _get_weights_for_task(df_lab, tr, t)

            model = CatBoostClassifier(
                **_catboost_common_params(
                    params,
                    seed=seed + 10000 * f + 97 * t + 7919 * trial.number,
                    thread_count=threads_per_model,
                    scale_pos_weight=spw,
                )
            )
            model.fit(
                X2d_raw[tr], y_cls[tr, t],
                sample_weight=w_train,
                eval_set=(X2d_raw[va], y_cls[va, t]),
                use_best_model=True,
                early_stopping_rounds=200,
                verbose=False,
            )

            pred_t = model.predict_proba(X2d_raw[va])[:, 1]

            wv = w_val_mat[:, t] if t in (0, 1) else None
            ap_t = ap_score(y_cls[va, t], pred_t, sample_weight_1d=wv)

            best_iter = None
            try:
                best_iter = int(model.get_best_iteration())
            except Exception:
                best_iter = None

            # free early
            del model
            return t, pred_t, float(ap_t), float(spw), best_iter

        aps: Dict[str, float] = {}
        iters: Dict[str, Any] = {}

        if parallel_tasks:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = [ex.submit(train_one_task, t) for t in range(4)]
                for fut in as_completed(futs):
                    t, pred_t, ap_t, spw, best_iter = fut.result()
                    preds[:, t] = pred_t
                    aps[f"{TASK_COLS[t]}_ap"] = float(ap_t)
                    aps[f"{TASK_COLS[t]}_ap_weighted"] = bool(t in (0, 1))
                    iters[f"{TASK_COLS[t]}_scale_pos_weight"] = float(spw)
                    iters[f"{TASK_COLS[t]}_best_iteration"] = best_iter
        else:
            for t in range(4):
                t, pred_t, ap_t, spw, best_iter = train_one_task(t)
                preds[:, t] = pred_t
                aps[f"{TASK_COLS[t]}_ap"] = float(ap_t)
                aps[f"{TASK_COLS[t]}_ap_weighted"] = bool(t in (0, 1))
                iters[f"{TASK_COLS[t]}_scale_pos_weight"] = float(spw)
                iters[f"{TASK_COLS[t]}_best_iteration"] = best_iter
                gc.collect()

        s = macro_ap(y_cls[va], preds, w_val=w_val_mat)
        scores.append(float(s))

        fold_detail[str(f)] = {
            "macro_ap": float(s),
            "threads_total": int(cpu_threads),
            "threads_per_model": int(threads_per_model),
            "task_parallel": bool(parallel_tasks),
            "task_jobs": int(jobs),
            **aps,
            **iters,
        }

        trial.report(float(np.mean(scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("fold_detail", fold_detail)
    return float(np.mean(scores))


# cli helpers
def _parse_tasks(task_args: Optional[List[str]]) -> List[int]:
    if not task_args:
        return [0, 1, 2, 3]
    out: List[int] = []
    name2idx = {n: i for i, n in enumerate(TASK_COLS)}
    for x in task_args:
        x = str(x)
        if x.isdigit():
            i = int(x)
            if i < 0 or i > 3:
                raise ValueError(f"--tasks: invalid index {i}, expected 0..3")
            out.append(i)
        else:
            if x not in name2idx:
                raise ValueError(f"--tasks: unknown task name '{x}'. Valid: {TASK_COLS} or 0..3")
            out.append(int(name2idx[x]))
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--labels", required=True)
    ap.add_argument("--feat2d_raw", required=True)
    ap.add_argument("--study_dir", required=True)

    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--fold_col", default="cv_fold")
    ap.add_argument("--use_splits", nargs="+", default=["train"])
    ap.add_argument("--folds", nargs="+", type=int, default=None)

    ap.add_argument("--trials_catboost", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cat_threads", type=int, default=-1)

    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Which tasks to optimize in per-task mode: indices 0..3 or exact task names.")
    ap.add_argument("--per_task", action="store_true",
                    help="Run separate Optuna study per task (best params per task).")
    ap.add_argument("--do_macro", action="store_true",
                    help="Run single-study macro optimizer (one param set for all tasks).")

    ap.add_argument("--per_task_parallel", action="store_true",
                    help="Run per-task studies in parallel (up to 4 tasks).")
    ap.add_argument("--per_task_jobs", type=int, default=4,
                    help="How many per-task studies to run concurrently (default 4).")

    ap.add_argument("--macro_task_parallel", action="store_true",
                    help="In macro study, train 4 task models in parallel inside each fold.")
    ap.add_argument("--macro_task_jobs", type=int, default=4,
                    help="How many task models to train concurrently in macro objective (default 4).")

    args = ap.parse_args()

    if not args.per_task and not args.do_macro:
        args.per_task = True

    outdir = Path(args.study_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.cat_threads < 0:
        args.cat_threads = int(os.cpu_count() or 8)

    df = load_labels(args.labels, id_col=args.id_col)
    df[args.split_col] = df[args.split_col].astype(str)
    df = df[df[args.split_col].isin(args.use_splits)].copy().reset_index(drop=True)

    ids_all = df[args.id_col].astype(str).tolist()

    ids_raw, X_raw_file = load_2d(args.feat2d_raw, id_col=args.id_col)
    X2d_raw = align_by_id(ids_raw, X_raw_file, ids_all)
    y_cls = coerce_binary_labels(df)

    if args.folds is None:
        folds = sorted(df[args.fold_col].dropna().astype(int).unique().tolist())
    else:
        folds = args.folds
    folds_info = fold_indices(df, args.fold_col, folds)

    run_meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "cat_threads": int(args.cat_threads),
        "argv": " ".join([str(x) for x in os.sys.argv]),
        "mode": {"per_task": bool(args.per_task), "macro": bool(args.do_macro)},
        "weight_cols": WEIGHT_COLS,
        "weighted_pr_auc_tasks": [0, 1],
        "parallel": {
            "per_task_parallel": bool(args.per_task_parallel),
            "per_task_jobs": int(args.per_task_jobs),
            "macro_task_parallel": bool(args.macro_task_parallel),
            "macro_task_jobs": int(args.macro_task_jobs),
        }
    }
    (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    print(f"[DATA] n_ids={len(ids_all)} | folds={folds} | cat_threads={args.cat_threads}")
    print(f"[WEIGHTS] task0={WEIGHT_COLS[0]} task1={WEIGHT_COLS[1]} task2=None task3=None")
    print("[METRIC] validation AP is WEIGHTED for tasks 0/1, UNWEIGHTED for tasks 2/3")

    summary: Dict[str, Any] = {}

    if args.per_task:
        task_idxs = _parse_tasks(args.tasks)
        print(f"[HPO] Per-task CatBoost studies: {[(i, TASK_COLS[i]) for i in task_idxs]}")

        jobs = int(max(1, min(args.per_task_jobs, len(task_idxs), 4)))
        threads_per_study = max(1, int(args.cat_threads) // jobs) if args.per_task_parallel else int(args.cat_threads)

        def run_one_task_study(t: int) -> Tuple[str, Dict[str, Any]]:
            task_name = TASK_COLS[t]
            study_name = f"catboost_task{t}_raw_cpu"
            storage = f"sqlite:///{(outdir / f'{study_name}.sqlite3').as_posix()}"

            sampler = optuna.samplers.TPESampler(seed=args.seed + 1234 * t)
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )

            study.optimize(
                lambda tr: objective_catboost_task_cv(
                    tr,
                    task_idx=t,
                    X2d_raw=X2d_raw,
                    df_lab=df,
                    y_cls=y_cls,
                    folds_info=folds_info,
                    seed=args.seed,
                    cpu_threads=threads_per_study,
                ),
                n_trials=args.trials_catboost,
                gc_after_trial=True,
                catch=(RuntimeError,),
            )

            save_study_artifacts(outdir, study, prefix=study_name)
            fold_detail = study.best_trial.user_attrs.get("fold_detail", {})
            save_best_fold_metrics(outdir, study_name, fold_detail)

            entry = {
                "best_value_ap_cvmean": float(study.best_value),
                "best_params": dict(study.best_params),
                "val_ap_weighted": bool(t in (0, 1)),
                "weight_col": WEIGHT_COLS.get(t, None),
                "threads_used": int(threads_per_study),
            }
            print(f"[HPO] {task_name}: best AP (CV-mean) = {study.best_value:.5f} | threads={threads_per_study}")
            return task_name, entry

        if args.per_task_parallel and len(task_idxs) > 1:
            print(f"[PARALLEL] per-task studies in parallel: jobs={jobs} threads_per_study={threads_per_study}")
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = [ex.submit(run_one_task_study, t) for t in task_idxs]
                for fut in as_completed(futs):
                    task_name, entry = fut.result()
                    summary[task_name] = entry
        else:
            print(f"[SERIAL] per-task studies serial: threads_per_study={threads_per_study}")
            for t in task_idxs:
                task_name, entry = run_one_task_study(t)
                summary[task_name] = entry
                gc.collect()

        (outdir / "catboost_best_params_per_task.json").write_text(json.dumps(summary, indent=2))

    if args.do_macro:
        print("[HPO] Macro CatBoost study (single param set for all tasks) ...")
        study_name = "catboost_raw_cpu_macro"
        storage = f"sqlite:///{(outdir / f'{study_name}.sqlite3').as_posix()}"

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        study.optimize(
            lambda tr: objective_catboost_macro_cv(
                tr,
                X2d_raw=X2d_raw,
                df_lab=df,
                y_cls=y_cls,
                folds_info=folds_info,
                seed=args.seed,
                cpu_threads=args.cat_threads,
                parallel_tasks=bool(args.macro_task_parallel),
                task_jobs=int(args.macro_task_jobs),
            ),
            n_trials=args.trials_catboost,
            gc_after_trial=True,
            catch=(RuntimeError,),
        )

        save_study_artifacts(outdir, study, prefix=study_name)
        fold_detail = study.best_trial.user_attrs.get("fold_detail", {})
        save_best_fold_metrics(outdir, study_name, fold_detail)

        print(f"[HPO] Macro best (CV-mean) = {study.best_value:.5f}")


if __name__ == "__main__":
    main()

