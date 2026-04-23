import os
import pickle
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import pandas as pd
import random
import os
from collections import Counter
from pathlib import Path
import argparse

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from preprocess import prepare_data
from src.signal_utils import WINDOW_SIZE

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def build_XY_from_df(df: pd.DataFrame):
    X = np.stack([np.vstack([h, s]).T for h, s in zip(df["hr_seq"], df["st_seq"])]).astype("float32")
    y = df["state_val"].astype(int).values
    return X, y

def best_f1_on_val(probs_val, y_val):
    thresholds = np.linspace(0.05, 0.95, 19)
    f1_vals = []
    for t in thresholds:
        y_pred = (probs_val >= t).astype(int)
        f1_vals.append(f1_score(y_val, y_pred, zero_division=0))
    best_idx = int(np.argmax(f1_vals))
    return float(thresholds[best_idx]), float(f1_vals[best_idx])


def mc_dropout_predict(model, X, T):
    preds = []
    for _ in range(T):
        p = model(X, training=True).numpy().ravel()
        preds.append(p)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def build_global_supervised_model(window_size: int, dropout_rate: float = 0.3):
    inp = layers.Input(shape=(window_size, 2))

    x = layers.Conv1D(64, 8, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(4, activation="relu")(se)
    se = layers.Dense(64, activation="sigmoid")(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(128, 5, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(8, activation="relu")(se)
    se = layers.Dense(128, activation="sigmoid")(se)
    se = layers.Reshape((1, 128))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)

    lstm_out = layers.LSTM(128)(inp)
    lstm_out = layers.Dropout(0.5)(lstm_out)

    combined = layers.concatenate([cnn_out, lstm_out])
    out = layers.Dense(1, activation="sigmoid")(combined)

    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def train_global_supervised(
    df_lab: pd.DataFrame,
    df_val: pd.DataFrame,
    window_size: int,
    dropout_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
):
    reset_seeds(42)
    X_lab, y_lab = build_XY_from_df(df_lab)
    X_val, y_val = build_XY_from_df(df_val)

    model = build_global_supervised_model(window_size, dropout_rate=dropout_rate)

    classes = np.unique(y_lab)
    class_weight = None
    if len(classes) == 2:
        cw = compute_class_weight("balanced", classes=classes, y=y_lab)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0)

    model.fit(
        X_lab, y_lab,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[es, lr],
        verbose=0,
        shuffle=False,
    )
    return model

def split_global_pool_stratified(df_all_tr: pd.DataFrame, labeled_frac: float = 0.1, seed: int = 42):
    idx = np.arange(len(df_all_tr))
    y = df_all_tr["state_val"].astype(int).values
    idx_lab, idx_unl = train_test_split(idx, test_size=(1 - labeled_frac), random_state=seed, stratify=y)
    df_lab = df_all_tr.iloc[idx_lab].copy()
    df_unl = df_all_tr.iloc[idx_unl].copy()
    return df_lab, df_unl


# def bootstrap_auc(y_true, probs, n_iters=1000, sample_frac=0.7, rng_seed=42):
#     y_true = np.asarray(y_true).astype(int)
#     probs = np.asarray(probs).astype(float)
#     rng = np.random.default_rng(rng_seed)

#     n = len(y_true)
#     m = max(2, int(sample_frac * n))
#     aucs = []

#     for _ in range(n_iters):
#         idx = rng.integers(0, n, size=m)
#         yb = y_true[idx]
#         pb = probs[idx]
#         if len(np.unique(yb)) < 2:
#             continue
#         aucs.append(roc_auc_score(yb, pb))

#     if not aucs:
#         return np.nan, np.nan
#     return float(np.mean(aucs)), float(np.std(aucs))
