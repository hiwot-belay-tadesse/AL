import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D, Dense, Input)

tf.config.set_visible_devices([], "GPU")
print(tf.config.list_logical_devices("GPU")) 

# ------------------------------------------------------------------------- #
# Configuration
# ------------------------------------------------------------------------- #
BASE_DATA_DIR = '.'
RESULTS_DIR   = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZE = 30      # samples per window
STEP_SIZE   = 15      # stride


# ------------------------------------------------------------------------- #
# Data utilities
# ------------------------------------------------------------------------- #
def load_and_prepare(path, dtype):
    df = pd.read_csv(path)
    df = df[df['data_type'] == dtype].copy()
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    return df


def split_data_based_on_days(df, train_frac=0.75):
    days     = np.sort(df['time'].dt.date.unique())
    n_train  = int(len(days) * train_frac)
    train_df = df[df['time'].dt.date.isin(days[:n_train])]
    test_df  = df[df['time'].dt.date.isin(days[n_train:])]
    return train_df, test_df


def create_windows(vals, win, step):
    return np.asarray([vals[s:s + win]
                       for s in range(0, len(vals) - win + 1, step)])


# ------------------------------------------------------------------------- #
# Augmentations
# ------------------------------------------------------------------------- #
def jitter(x, noise=0.05):
    return x + np.random.normal(0, noise, x.shape)


def flip(x):
    return x[::-1]


def stretch_crop(x, scale_rng=(0.8, 1.2)):
    flat = x.flatten()
    L    = len(flat)

    scale = np.random.uniform(*scale_rng)
    new_L = int(round(L * scale))
    stretched = interp1d(np.arange(L), flat, kind='linear')(
        np.linspace(0, L - 1, new_L)
    )

    if new_L > L:                      # crop
        start = np.random.randint(0, new_L - L + 1)
        cropped = stretched[start:start + L]
    elif new_L < L:                    # pad
        pad   = L - new_L
        left  = pad // 2
        right = pad - left
        cropped = np.pad(stretched, (left, right), mode='edge')
    else:
        cropped = stretched

    return cropped.reshape(x.shape)


def apply_augmentations(segs):
    out = np.zeros_like(segs)
    for i, s in enumerate(segs):
        v = s.copy()
        if np.random.rand() < 0.5:
            v = jitter(v)
        if np.random.rand() < 0.5:
            v = flip(v)
        if np.random.rand() < 0.5:
            v = stretch_crop(v)
        out[i] = v
    return out


# ------------------------------------------------------------------------- #
# Model helpers
# ------------------------------------------------------------------------- #
def create_encoder(win, feats):
    return Sequential([
        Input(shape=(win, feats)),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        GlobalAveragePooling1D(),
        # Dense(128, activation='relu') ## reduced dimension from 128 -> 32 for AL experiment
        Dense(32, activation='relu')

    ])


def create_projection_head():
    return Sequential([Dense(64, activation='relu'), Dense(32)])



def contrastive_loss(z_i, z_j, temperature=0.1):
    """
    NT‑Xent loss:
    • concatenate projections
    • mask diagonal so self‑similarity is never a negative
    • positives are (i, i+batch)
    """

    # Convert the PyTorch tensors to CPU-based NumPy arrays
    # z_i_np = z_i.detach().cpu().numpy()
    # z_j_np = z_j.detach().cpu().numpy()
    
    z = tf.concat([z_i, z_j], axis=0)                 # (2B, D)
    z = tf.math.l2_normalize(z, axis=1)
    logits = tf.matmul(z, z, transpose_b=True) / temperature

    bsz = tf.shape(z_i)[0]
    mask = tf.eye(2 * bsz, dtype=logits.dtype) * -1e9
    logits = logits + mask

    positives = tf.concat([tf.range(bsz, 2 * bsz),
                        tf.range(0,   bsz   )], axis=0)

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        positives, logits, from_logits=True
    )

    return tf.reduce_mean(loss)

# ------------------------------------------------------------------------- #
# SimCLR training loop
# ------------------------------------------------------------------------- #
def train_simclr(encoder, head, train_segs, val_segs,
                 batch_size=128, epochs=80):
    opt   = tf.keras.optimizers.Adam(1e-3)
    n_tr  = len(train_segs)
    val_ds = tf.data.Dataset.from_tensor_slices(val_segs).batch(batch_size)

    tr_loss_hist, va_loss_hist = [], []

    # Early‐stopping & LR‐scheduler state
    best_val = float('inf')
    wait_es  = 0
    wait_lr  = 0
    patience_es = 15
    patience_lr = 5
    lr_factor   = 0.5
    min_lr      = 1e-6

    for ep in range(1, epochs + 1):
        # ---- training ----
        idx = np.random.permutation(n_tr)
        total = 0.0
        for i in range(0, n_tr, batch_size):
            b   = idx[i:i + batch_size]
            x_i = train_segs[b]
            x_j = apply_augmentations(x_i.copy())

            with tf.GradientTape() as tape:
                z_i  = encoder(x_i, training=True)
                z_j  = encoder(x_j, training=True)
                p_i  = head(z_i, training=True)
                p_j  = head(z_j, training=True)
                # breakpoint()
                loss = contrastive_loss(p_i, p_j)

            vars_ = encoder.trainable_weights + head.trainable_weights
            grads  = tape.gradient(loss, vars_)
            opt.apply_gradients(zip(grads, vars_))
            total += loss.numpy() * len(b)
        tr_loss = total / n_tr
        tr_loss_hist.append(tr_loss)

        # ---- validation ----
        vm = tf.keras.metrics.Mean()
        for x in val_ds:
            x_rev = tf.reverse(x, axis=[1])
            z_i = encoder(x,     training=False)
            z_j = encoder(x_rev, training=False)
            p_i = head(z_i, training=False)
            p_j = head(z_j, training=False)
            vm.update_state(contrastive_loss(p_i, p_j))
        va_loss = vm.result().numpy()
        va_loss_hist.append(va_loss)

        # ---- callbacks logic ----
        # Early stopping
        if va_loss < best_val:
            best_val = va_loss
            wait_es  = 0
            wait_lr  = 0
        else:
            wait_es += 1
            wait_lr += 1

        # Reduce LR on plateau
        if wait_lr >= patience_lr:
            new_lr = max(opt.learning_rate.numpy() * lr_factor, min_lr)
            opt.learning_rate.assign(new_lr)
            print(f"[SimCLR] reducing lr to {new_lr:.2e}")
            wait_lr = 0

        # Break if no improvement for patience_es
        if wait_es >= patience_es:
            print(f"[SimCLR] early stopping at epoch {ep} (no val_loss improvement for {patience_es} epochs).")
            break

        print(f"[SimCLR {ep:>3}/{epochs}]  train={tr_loss:.4f}  val={va_loss:.4f}")

    return tr_loss_hist, va_loss_hist

def build_simclr_encoder(win_size):
    return create_encoder(win_size, 1)