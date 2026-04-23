#!/usr/bin/env python3
"""
Hyperparameter tuning utilities for the global supervised CNN baseline.

This file provides Bayesian hyperparameter tuning for the global supervised CNN baseline.

"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

try:
    import keras_tuner as kt
except ImportError:  # fallback for older environments
    import kerastuner as kt  # type: ignore

from src.classifier_utils import (
    ALLOWED_SCENARIOS,
    BASE_DATA_DIR,
    load_label_data,
    load_signal_data,
)
from src.signal_utils import WINDOW_SIZE
from src.compare_pipelines import (
    bootstrap_threshold_metrics,
    collect_windows,
    derive_negative_labels,
    ensure_train_val_test_days,
    _bp_load_all,
    _bp_pid_from_user_dir,
    plot_clf_losses,
)

def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_cnn_baseline_from_hp(
    hp: Any,
    window_size: int,
) -> Model:
    """KerasTuner model builder mirroring build_cnn_baseline style."""
    filters      = hp.Choice("filters",     values=[8],          default=8)
    kernel_size  = hp.Choice("kernel_size", values=[5, 8, 12],   default=8)
    dropout      = hp.Float("dropout",      min_value=0.1,
                                            max_value=0.3,
                                            step=0.05,           default=0.2)
    weight_decay = hp.Float("weight_decay", min_value=1e-3,
                                            max_value=1e-2,
                                            sampling="log",      default=1e-3)
    lr           = hp.Float("lr",           min_value=1e-5,
                                            max_value=1e-4,      # hard ceiling
                                            sampling="log",      default=5e-5)
    inp = layers.Input(shape=(window_size, 2))
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_fixed_cnn(window_size: int, y_train: np.ndarray) -> Model:
    """
    Fixed conservative config for low-sample settings.
    """
    inp = layers.Input(shape=(window_size, 2))
    x = layers.Conv1D(
        16,
        kernel_size=8,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(3e-3),
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)
    # model.compile(
    #     optimizer=Adam(learning_rate=9e-4, clipnorm=1.0),
    #     loss="binary_crossentropy",
    #     metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    # )
    neg_count = np.sum(y_train == 0)  # 226
    pos_count = np.sum(y_train == 1)  # 113
    pos_weight = neg_count / pos_count  # 2.0

    # Custom weighted loss
    def weighted_bce(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight = tf.where(tf.equal(y_true, 1), pos_weight, 1.0)
        return tf.reduce_mean(tf.cast(bce, tf.float32) * tf.cast(weight, tf.float32))
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=weighted_bce,  
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model


def _build_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([np.vstack([h, s]).T for h, s in zip(df["hr_seq"], df["st_seq"])])
    y = df["state_val"].values
    return X, y


def build_gs_tuning_data_from_splits(
    fruit: str,
    scenario: str,
    all_splits: dict,
    *,
    collect_windows_fn: Callable,
    derive_negative_labels_fn: Callable,
    load_signal_data_fn: Callable = load_signal_data,
    load_label_data_fn: Callable = load_label_data,
    bp_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X_train, y_train, X_val, y_val using the same logic as
    compare_pipelines.py::build_cnn_baseline.
    """
    if bp_mode:
        user_iter = all_splits.keys()
    else:
        user_iter = [u for u, pairs in ALLOWED_SCENARIOS.items() if (fruit, scenario) in pairs]

    # TRAIN windows from each user's train days
    df_train_list = []
    for u in user_iter:
        tr_days_u, _, _ = all_splits.get(u, ([], [], []))
        if len(tr_days_u) == 0:
            continue

        hr_df, st_df = load_signal_data_fn(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data_fn(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data_fn(Path(BASE_DATA_DIR) / u, fruit, "None")

        if bp_mode:
            neg_df = orig_neg
        else:
            if len(orig_neg) < len(pos_df):
                extra = derive_negative_labels_fn(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

        df_u = collect_windows_fn(pos_df, neg_df, hr_df, st_df, tr_days_u)
        if len(df_u) > 0:
            df_train_list.append(df_u)

    if not df_train_list:
        raise RuntimeError("No train windows built for GS tuning.")

    df_train = pd.concat(df_train_list, axis=0, ignore_index=True)
    X_train, y_train = _build_xy(df_train)

    # POOLED VAL windows from each user's val days
    X_val_list, y_val_list = [], []
    for u in user_iter:
        _, val_days_u, _ = all_splits.get(u, ([], [], []))
        if len(val_days_u) == 0:
            continue

        hr_df_u, st_df_u = load_signal_data_fn(Path(BASE_DATA_DIR) / u)
        pos_df_u = load_label_data_fn(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg_u = load_label_data_fn(Path(BASE_DATA_DIR) / u, fruit, "None")

        if bp_mode:
            neg_df_u = orig_neg_u
        else:
            if len(orig_neg_u) < len(pos_df_u):
                extra_u = derive_negative_labels_fn(
                    hr_df_u, pos_df_u, len(pos_df_u) - len(orig_neg_u)
                )
                neg_df_u = pd.concat([orig_neg_u, extra_u], ignore_index=True)
            else:
                neg_df_u = orig_neg_u

        df_val_u = collect_windows_fn(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)
        if len(df_val_u) == 0:
            continue

        X_val_u, y_val_u = _build_xy(df_val_u)
        X_val_list.append(X_val_u)
        y_val_list.append(y_val_u)

    if not X_val_list:
        raise RuntimeError("No validation windows built for GS tuning.")

    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    return X_train, y_train, X_val, y_val


def tune_build_cnn_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    window_size: int,
    max_trials: int = 30,
    epochs: int = 200,
    batch_size: int = 16,
    patience: int = 7,
    min_delta: float = 2e-3,
    directory: str | Path = "tuner_results",
    project_name: str = "cnn_baseline",
    overwrite: bool = True,
    seed: int = 42,
    gap_weight: float = 0.5,
    force_fixed_model: bool = False,
) -> tuple[Any, Model, Any]:
    """
    Bayesian tuning for build_cnn_baseline-like architecture.

    Returns:
        best_hp: best KerasTuner HyperParameters
        best_model: best trained Keras model
        tuner: fitted KerasTuner object or None when fixed model path is used
    """
    out_dir = Path(directory) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
    )
    class ConvergenceScoreCallback(Callback):
        """Adds a scalar objective that rewards low val_loss + small train/val gap."""
        def __init__(self, gap_weight_value: float):
            super().__init__()
            self.gap_weight = float(gap_weight_value)

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            tr = logs.get("loss")
            va = logs.get("val_loss")
            if tr is None or va is None:
                return
            logs["convergence_score"] = float(va + self.gap_weight * abs(va - tr))

    if force_fixed_model:
        tuner = None
        best_hp = {
            "filters": 16,
            "kernel_size": 8,
            "dropout": 0.2,
            "weight_decay": 1e-3,
            "lr": 1e-4,
            "fixed_model": True,
        }
        best_model = build_fixed_cnn(window_size, y_train)
    else:
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: _build_cnn_baseline_from_hp(hp, window_size=window_size),
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=max_trials,
            overwrite=overwrite,
            directory=str(directory),
            project_name=project_name,
            seed=seed,
        )

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[ConvergenceScoreCallback(gap_weight), es],
            verbose=1,
        )
        # Use partial correction

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hp)


    set_seed(42)
    best_hist = best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )

    plot_clf_losses(
        best_hist.history["loss"],
        best_hist.history["val_loss"],
        out_dir,
        "cnn_baseline_best_hp_loss",
    )

    if isinstance(best_hp, dict):
        hp_values = best_hp
    elif hasattr(best_hp, "values") and isinstance(best_hp.values, dict):
        hp_values = best_hp.values
    else:
        hp_values = {"best_hp": str(best_hp)}
    # print(f"Best filters:      {hp_values.get('filters')}") 
    # print(f"Best kernel_size:  {hp_values.get('kernel_size')}")
    # print(f"Best dropout:      {hp_values.get('dropout')}")
    # print(f"Best weight_decay: {hp_values.get('weight_decay')}")
    # print(f"Best lr:           {hp_values.get('lr')}")

    probs_train = best_model.predict(X_train, verbose=0).ravel()
    probs_val = best_model.predict(X_val, verbose=0).ravel()

    _, auc_train_mean, auc_train_std = bootstrap_threshold_metrics(
        y_train,
        probs_train,
        thresholds=np.array([0.5]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=seed,
    )
    _, auc_val_mean, auc_val_std = bootstrap_threshold_metrics(
        y_val,
        probs_val,
        thresholds=np.array([0.5]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=seed,
    )

    print(f"AUC_Train (bootstrap): {auc_train_mean:.4f} ± {auc_train_std:.4f}")
    print(f"AUC_Val (bootstrap):   {auc_val_mean:.4f} ± {auc_val_std:.4f}")

    with open(out_dir / "auc_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "AUC_Train_Mean": float(auc_train_mean),
                "AUC_Train_STD": float(auc_train_std),
                "AUC_Val_Mean": float(auc_val_mean),
                "AUC_Val_STD": float(auc_val_std),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    return best_hp, best_model, tuner


def tune_build_cnn_baseline_from_splits(
        fruit: str,
        scenario: str,
        all_splits: dict,
        *,
        window_size: int,
        collect_windows_fn: Callable,
        derive_negative_labels_fn: Callable,
        load_signal_data_fn: Callable = load_signal_data,
        load_label_data_fn: Callable = load_label_data,
        bp_mode: bool = False,
        max_trials: int = 30,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 7,
        min_delta: float = 2e-3,
        directory: str | Path = "tuner_results",
        project_name: str = "cnn_baseline",
        overwrite: bool = True,
        seed: int = 42,
        gap_weight: float = 0.5,
    ) -> tuple[Any, Model, Any]:
        """
        Convenience wrapper:
        1) Build train/val arrays from compare_pipelines-style splits.
        2) Run Bayesian tuning.
        """
        X_train, y_train, X_val, y_val = build_gs_tuning_data_from_splits(
            fruit=fruit,
            scenario=scenario,
            all_splits=all_splits,
            collect_windows_fn=collect_windows_fn,
            derive_negative_labels_fn=derive_negative_labels_fn,
            load_signal_data_fn=load_signal_data_fn,
            load_label_data_fn=load_label_data_fn,
            bp_mode=bp_mode,
        )

        fruit_key = str(fruit).casefold()
        scenario_key = str(scenario).casefold()
        use_fixed_model = (
           (fruit_key== "carrot") and scenario_key == "use"
        )


        return tune_build_cnn_baseline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        window_size=window_size,
        max_trials=max_trials,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        min_delta=min_delta,
        directory=directory,
        project_name=project_name,
        overwrite=overwrite,
        seed=seed,
        gap_weight=gap_weight,
        force_fixed_model=use_fixed_model,
    )


def _build_all_splits_fruit(fruit: str, scenario: str) -> dict:
    all_splits = {}
    for u, pairs in ALLOWED_SCENARIOS.items():
        if (fruit, scenario) not in pairs:
            continue
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, "None")
        if len(orig_neg) < len(pos_df):
            extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg
        try:
            tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as exc:
            print(f"Skipping user {u}: {exc}")
            continue
        all_splits[u] = (tr_u, val_u, te_u)
    return all_splits


def _build_all_splits_bp() -> tuple[dict, Callable, Callable]:
    all_splits = {}
    bp_users = []
    for p in sorted(Path("DATA/Cardiomate/hp").glob("hp*")):
        try:
            pid = _bp_pid_from_user_dir(p)
        except Exception:
            continue
        base = Path("DATA/Cardiomate/hp") / f"hp{pid}"
        if not (base / f"hp{pid}_hr.csv").exists():
            continue
        if not (base / f"hp{pid}_steps.csv").exists():
            continue
        if not (base / f"blood_pressure_readings_ID{pid}_cleaned.csv").exists():
            continue
        bp_users.append(pid)

    for pid in bp_users:
        hr_df, st_df, pos_df, neg_df = _bp_load_all(pid)
        try:
            tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as exc:
            print(f"Skipping user {pid}: {exc}")
            continue
        all_splits[pid] = (tr_u, val_u, te_u)

    def _bp_load_signal_data(user_dir: Path):
        pid = _bp_pid_from_user_dir(user_dir)
        hr_df, st_df, _, _ = _bp_load_all(pid)
        return hr_df, st_df

    def _bp_load_label_data(user_dir: Path, _fruit: str, scenario: str):
        pid = _bp_pid_from_user_dir(user_dir)
        _, _, pos_df, neg_df = _bp_load_all(pid)
        if scenario == "None":
            return neg_df.copy()
        return pos_df.copy()

    return all_splits, _bp_load_signal_data, _bp_load_label_data


def main() -> None:
    parser = argparse.ArgumentParser(description="KerasTuner tuning for global supervised baseline CNN.")
    parser.add_argument("--task", choices=["fruit", "bp"], default="fruit")
    parser.add_argument("--fruit", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=2e-3)
    parser.add_argument("--directory", type=str, default="tuner_results")
    parser.add_argument("--project-name", type=str, default="cnn_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gap-weight",
        type=float,
        default=0.5,
        help="Penalty weight for |val_loss - train_loss| in tuning objective.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.task == "fruit":
        if not (args.fruit and args.scenario):
            raise SystemExit("For --task fruit, provide --fruit and --scenario.")
        fruit, scenario = args.fruit, args.scenario
        all_splits = _build_all_splits_fruit(fruit, scenario)
        load_signal_fn = load_signal_data
        load_label_fn = load_label_data
        bp_mode = False
    else:
        fruit = args.fruit or "BP"
        scenario = args.scenario or "spike"
        all_splits, load_signal_fn, load_label_fn = _build_all_splits_bp()
        bp_mode = True

    if not all_splits:
        raise SystemExit("No valid user splits found for tuning.")

    task_tag = "bp" if bp_mode else "fruit"
    fruit_tag = str(fruit).replace("/", "_").replace(" ", "_")
    scenario_tag = str(scenario).replace("/", "_").replace(" ", "_")
    run_subdir = Path(task_tag) / fruit_tag / scenario_tag
    tuned_directory = Path(args.directory) / run_subdir
    tuned_project_name = args.project_name

    best_hp, best_model, tuner = tune_build_cnn_baseline_from_splits(
        fruit=fruit,
        scenario=scenario,
        all_splits=all_splits,
        window_size=args.window_size,
        collect_windows_fn=collect_windows,
        derive_negative_labels_fn=derive_negative_labels,
        load_signal_data_fn=load_signal_fn,
        load_label_data_fn=load_label_fn,
        bp_mode=bp_mode,
        max_trials=args.max_trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        directory=tuned_directory,
        project_name=tuned_project_name,
        overwrite=args.overwrite,
        seed=args.seed,
        gap_weight=args.gap_weight,
    )

    out_dir = tuned_directory / tuned_project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model.save(out_dir / "best_model.keras")
    if isinstance(best_hp, dict):
        hp_payload = best_hp
    elif hasattr(best_hp, "values") and isinstance(best_hp.values, dict):
        hp_payload = best_hp.values
    else:
        hp_payload = {"best_hp": str(best_hp)}
    with open(out_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(hp_payload, f, indent=2, sort_keys=True)

    print(f"Saved best model to: {out_dir / 'best_model.keras'}")
    print(f"Saved best hyperparameters to: {out_dir / 'best_hyperparameters.json'}")


if __name__ == "__main__":
    main()
