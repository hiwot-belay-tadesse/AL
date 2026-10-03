"""Train ADARP SimCLR encoders with Matton et al. augmentations on 4 Hz EDA.

Self-contained alternative to the encoder step inside
`preprocess_adarp_data.prepare_data` / `src.compare_pipelines._train_or_load_encoder`.
Nothing in those files is modified; the pieces that should stay identical are
imported from them:

    load_adarp              read_signal / session_dirs       (raw 4 Hz EDA, via data4hz)
    preprocess_adarp_data   load_windows, windows_to_frame, split_sessions,
                            load_signals                     (the split; the 1 Hz HR stream)
    src.signal_utils        build_simclr_encoder, create_projection_head,
                            contrastive_loss, apply_augmentations   (model + HR augs)

How this differs from the existing encoder training:

  * EDA is the E4's native 4 Hz signal, unfiltered, in microsiemens, cut into
    60 s windows of 240 samples -- the paper's setting. The pipeline's encoder
    instead sees EDA averaged to 1 Hz and binned to 30 points.
  * EDA views are made with `eda_augmentations.MattonAugmenter` at fs=4: one
    transform per view sampled from the paper's 17, with the paper's parameter
    ranges exactly as in their pretraining config (no rescaling is needed at 4 Hz).
    Both views are augmented, as in the paper.
  * Each view is z-scored per window after augmentation. Several transforms are
    defined in µS, so standardising first would break them.
  * HR keeps the pipeline's 1 Hz exported stream and its own augmentations
    (`apply_augmentations(task="adarp")`), in 60 s windows of 60 samples so the
    HR encoder sees the same seconds the EDA encoder does.
  * The SSL validation loss uses 20% of training sessions held out by session.

Unchanged: encoder architecture (`create_encoder`, length-agnostic thanks to
global pooling, 32-d output), projection head, NT-Xent at temperature 0.1,
Adam 1e-3, early stopping (patience 15), LR halving on plateau (patience 5),
training-session-only streams from the pipeline's split.

Because the EDA encoder takes 240-sample 4 Hz windows, the pipeline's 30-point
scoring path cannot feed it. Use `score_one_class_svm.py` in this folder, which
cuts every labelled window from the 4 Hz stream by timestamp.

Outputs:
    <out_dir>/seed_<seed>/_global_encoders/adarp_{hr,steps}_encoder.keras       --pool global
    <out_dir>/seed_<seed>/<pid>/ADARP_stress/personal/models_saved/{hr,steps}_encoder.keras
    <out_dir>/seed_<seed>/.../ssl_history_<channel>.csv, ssl_loss_<channel>.png
    <out_dir>/seed_<seed>/encoder_config.json      (fs, window, stride, aug table)

    python ADARP/matton_ssl/train_encoders.py                        # global, eda + hr
    python ADARP/matton_ssl/train_encoders.py --channels eda
    python ADARP/matton_ssl/train_encoders.py --pool personal --participants 105 112
    ADARP_BATCH_SSL=32 ADARP_SSL_EPOCHS=100 python ADARP/matton_ssl/train_encoders.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ADARP_DIR = _HERE.parent
_REPO = _ADARP_DIR.parent
for p in (_HERE, _ADARP_DIR, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from preprocess_adarp_data import (  # noqa: E402  (reused unchanged)
    HR_HZ,
    PROCESSED_DIR,
    load_signals,
    load_windows,
    split_sessions,
    windows_to_frame,
)
from data4hz import (  # noqa: E402
    ENV_SENSOR_DIR, FS_EDA, WindowBank, load_eda_4hz, resolve_sensor_dir,
    session_arrays_4hz, zscore_rows,
)
from eda_augmentations import MattonAugmenter, TRANSFORM_NAMES, paper_params  # noqa: E402

OUT_DIR = _HERE / "encoders"
FRUIT_SCENARIO = "ADARP_stress"
BATCH_SSL = int(os.environ.get("ADARP_BATCH_SSL", "32"))      # as run_adarp / full_data_auc
SSL_EPOCHS = int(os.environ.get("ADARP_SSL_EPOCHS", "100"))

WINDOW_SEC = 60            # the paper's (and the labelled pool's) window
STEP_SEC = 30              # stride for the SSL pool; the labelled pool uses 30 s too
BUFFER_SEC = 60            # context either side for the time shift (paper: up to 60 s)
VAL_SESSION_FRAC = 0.2
CHANNEL_FS = {"eda": FS_EDA, "hr": float(HR_HZ)}
CHANNEL_SLOT = {"eda": "steps", "hr": "hr"}     # the pipeline stores EDA in the "steps" slot


# ------------------------------------------------------------------- data

def training_sessions(processed_dir, seed):
    """{participant: [train session names]} -- the pipeline's split, train block only."""
    eda, hr, meta = load_windows(processed_dir)
    frame = windows_to_frame(eda, hr, meta)
    splits = split_sessions(frame, seed=seed)
    return {pid: list(s["train"]) for pid, s in splits.items()}


def channel_sessions(channel, participants, sessions_by_pid, processed_dir,
                     sensor_dir=None, verbose=True):
    """[(pid, session, values)] at the channel's native rate over training sessions."""
    if channel == "eda":
        frames = {}
        for pid in participants:
            t0 = time.time()
            frames[pid] = load_eda_4hz(pid, sensor_dir=sensor_dir)
            if verbose:
                print(f"[matton-ssl] 4 Hz EDA {pid}: {len(frames[pid]):,} samples "
                      f"({time.time() - t0:.0f}s)", flush=True)
        return session_arrays_4hz(frames, sessions_by_pid, participants)

    signals = load_signals(processed_dir)          # 1 Hz, the pipeline's own export
    out = []
    for pid in participants:
        frame = signals.get(pid)
        if frame is None:
            continue
        for sess in sessions_by_pid.get(pid, []):
            vals = frame.loc[frame["session"] == sess, "hr"].to_numpy(dtype=float)
            if len(vals):
                out.append((pid, sess, vals))
    return out


def split_sessions_for_ssl(sessions, frac, seed):
    """Hold out `frac` of (participant, session) pairs for the SSL validation loss."""
    rng = np.random.default_rng(seed)
    keys = sorted({(p, s) for p, s, _ in sessions})
    n_val = max(1, int(round(frac * len(keys)))) if len(keys) > 1 else 0
    perm = [keys[i] for i in rng.permutation(len(keys))]
    val_keys = set(perm[:n_val])
    tr = [t for t in sessions if (t[0], t[1]) not in val_keys]
    va = [t for t in sessions if (t[0], t[1]) in val_keys]
    return tr, va


# ------------------------------------------------------------- augmenters

def make_view_fn(channel, fs, seed, aug_names=None):
    """Returns f(X, L, R) -> augmented, per-window standardised view, shape (n, w, 1)."""
    if channel == "eda":
        aug = MattonAugmenter(fs=fs, seed=seed, names=tuple(aug_names or TRANSFORM_NAMES))

        def view(X, L, R):
            Y, _ = aug.batch(X, L, R)
            return zscore_rows(Y)[..., None]
        view.params = aug.params
        return view

    from src.signal_utils import apply_augmentations     # the pipeline's HR augs

    def view(X, L, R):
        Z = zscore_rows(X)[..., None]
        return apply_augmentations(Z.copy(), task="adarp").astype(np.float32)
    view.params = {"hr": "src.signal_utils.apply_augmentations(task='adarp')"}
    return view


# --------------------------------------------------------------- training

def train_one(bank_tr, bank_va, view_fn, batch_size, epochs, seed, log_prefix):
    import tensorflow as tf
    from src.signal_utils import (build_simclr_encoder, contrastive_loss,
                                  create_projection_head)

    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    enc = build_simclr_encoder(bank_tr.window)
    head = create_projection_head()
    opt = tf.keras.optimizers.Adam(1e-3)

    n_tr = len(bank_tr)
    steps = -(-n_tr // batch_size)
    heartbeat = int(os.environ.get("BAN_AL_SSL_PROGRESS_EVERY", "200"))
    print(f"{log_prefix} {n_tr:,} train / {len(bank_va):,} val windows of {bank_tr.window} "
          f"samples, batch={batch_size}, {steps:,} steps/epoch, up to {epochs} epochs",
          flush=True)

    best, wait_es, wait_lr = float("inf"), 0, 0
    patience_es, patience_lr, lr_factor, min_lr = 15, 5, 0.5, 1e-6
    hist = []
    val_rows = np.arange(len(bank_va))

    for ep in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(n_tr)
        total = 0.0
        for i in range(0, n_tr, batch_size):
            rows = order[i:i + batch_size]
            X, L, R = bank_tr.gather(rows, with_buffers=True)
            x_i = view_fn(X, L, R)
            x_j = view_fn(X, L, R)
            with tf.GradientTape() as tape:
                p_i = head(enc(x_i, training=True), training=True)
                p_j = head(enc(x_j, training=True), training=True)
                loss = contrastive_loss(p_i, p_j)
            vars_ = enc.trainable_weights + head.trainable_weights
            opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
            total += float(loss) * len(rows)
            step = i // batch_size + 1
            if heartbeat and step % heartbeat == 0:
                print(f"{log_prefix} ep {ep:>3} step {step:,}/{steps:,} "
                      f"train={total / min(i + batch_size, n_tr):.4f} {time.time() - t0:.0f}s",
                      flush=True)
        tr_loss = total / n_tr

        vm = tf.keras.metrics.Mean()
        for i in range(0, len(val_rows), batch_size):
            X, L, R = bank_va.gather(val_rows[i:i + batch_size], with_buffers=True)
            p_i = head(enc(view_fn(X, L, R), training=False), training=False)
            p_j = head(enc(view_fn(X, L, R), training=False), training=False)
            vm.update_state(contrastive_loss(p_i, p_j))
        va_loss = float(vm.result())
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                     "lr": float(opt.learning_rate.numpy())})

        if va_loss < best:
            best, wait_es, wait_lr = va_loss, 0, 0
        else:
            wait_es += 1
            wait_lr += 1
        if wait_lr >= patience_lr:
            new_lr = max(float(opt.learning_rate.numpy()) * lr_factor, min_lr)
            opt.learning_rate.assign(new_lr)
            print(f"{log_prefix} lr -> {new_lr:.2e}", flush=True)
            wait_lr = 0
        print(f"{log_prefix} ep {ep:>3}/{epochs} train={tr_loss:.4f} val={va_loss:.4f} "
              f"{time.time() - t0:.0f}s", flush=True)
        if wait_es >= patience_es:
            print(f"{log_prefix} early stop at epoch {ep}", flush=True)
            break

    enc.trainable = False
    return enc, pd.DataFrame(hist)


def plot_history(hist, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist["epoch"], hist["train_loss"], label="train")
    ax.plot(hist["epoch"], hist["val_loss"], label="val (held-out sessions)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------- run

def encoder_target(out_dir, seed, pool, channel, pid=None):
    base = Path(out_dir) / f"seed_{seed}"
    slot = CHANNEL_SLOT[channel]
    if pool == "global":
        d = base / "_global_encoders"
        return d / f"adarp_{slot}_encoder.keras", d
    d = base / str(pid) / FRUIT_SCENARIO / "personal" / "models_saved"
    return d / f"{slot}_encoder.keras", d


def train_channel(channel, sessions, out_path, out_dir, seed, batch_size, epochs,
                  window_sec=WINDOW_SEC, step_sec=STEP_SEC, aug_names=None, overwrite=False):
    tag = f"[matton-ssl {channel}]"
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"{tag} exists, skipping: {out_path}", flush=True)
        return None

    fs = CHANNEL_FS[channel]
    tr_sessions, va_sessions = split_sessions_for_ssl(sessions, VAL_SESSION_FRAC, seed)
    bank_tr = WindowBank(tr_sessions, window_sec * fs, step_sec * fs, BUFFER_SEC * fs)
    bank_va = WindowBank(va_sessions, window_sec * fs, step_sec * fs, BUFFER_SEC * fs)
    if len(bank_tr) == 0:
        raise SystemExit(f"{tag} no windows to train on")
    if len(bank_va) == 0:
        bank_va = bank_tr      # tiny personal pools: validate on train rather than crash

    view_fn = make_view_fn(channel, fs=fs, seed=seed, aug_names=aug_names)
    enc, hist = train_one(bank_tr, bank_va, view_fn, batch_size, epochs, seed, tag)

    out_dir.mkdir(parents=True, exist_ok=True)
    enc.save(out_path)
    hist.to_csv(out_dir / f"ssl_history_{channel}.csv", index=False)
    plot_history(hist, out_dir / f"ssl_loss_{channel}.png",
                 f"{channel} SimCLR @ {fs:g} Hz, "
                 f"{'Matton augs' if channel == 'eda' else 'pipeline augs'}")
    print(f"{tag} saved {out_path}  ({len(hist)} epochs, best val={hist['val_loss'].min():.4f})",
          flush=True)
    return view_fn.params


def main():
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--processed_dir", default=str(PROCESSED_DIR),
                    help="Pipeline outputs: adarp_windows.{npz,csv} and signals/.")
    pa.add_argument("--sensor_dir", default=None,
                    help=f"Folder holding 'Part 101C' ... 'Part 112C' with the raw E4 CSVs. "
                         f"Default: ${ENV_SENSOR_DIR} if set, else <repo>/DATA/ADARP/Sensor Data "
                         f"(the pipeline's location). On the cluster: ~/AL/DATA/ADARP/Sensor\ Data.")
    pa.add_argument("--out_dir", default=str(OUT_DIR))
    pa.add_argument("--seed", type=int, default=42, help="Split seed, as the pipeline.")
    pa.add_argument("--pool", default="global", choices=["global", "personal"])
    pa.add_argument("--participants", nargs="*", default=None,
                    help="Subset of participant ids (default: everyone in the split).")
    pa.add_argument("--channels", nargs="+", default=["eda", "hr"], choices=["eda", "hr"])
    pa.add_argument("--augmentations", nargs="*", default=None,
                    help=f"Subset of EDA transforms (default: all 17). Choices: {TRANSFORM_NAMES}")
    pa.add_argument("--window_sec", type=float, default=WINDOW_SEC)
    pa.add_argument("--step_sec", type=float, default=STEP_SEC)
    pa.add_argument("--batch_ssl", type=int, default=BATCH_SSL)
    pa.add_argument("--ssl_epochs", type=int, default=SSL_EPOCHS)
    pa.add_argument("--overwrite", action="store_true")
    args = pa.parse_args()

    processed_dir, out_dir = Path(args.processed_dir), Path(args.out_dir)
    sensor_dir = resolve_sensor_dir(args.sensor_dir) if "eda" in args.channels else None
    if sensor_dir is not None:
        print(f"[matton-ssl] raw 4 Hz EDA from {sensor_dir}", flush=True)
    sessions_by_pid = training_sessions(processed_dir, args.seed)
    participants = args.participants or sorted(sessions_by_pid)

    seed_dir = out_dir / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "seed": args.seed, "pool": args.pool, "channels": args.channels,
        "batch_ssl": args.batch_ssl, "ssl_epochs": args.ssl_epochs,
        "window_sec": args.window_sec, "step_sec": args.step_sec, "buffer_sec": BUFFER_SEC,
        "fs_hz": CHANNEL_FS, "val_session_frac": VAL_SESSION_FRAC,
        "eda_source": "raw E4 EDA.csv at 4 Hz, unfiltered, microsiemens (data4hz.load_eda_4hz)",
        "sensor_dir": str(sensor_dir),
        "hr_source": "processed/signals/<pid>_hr.csv at 1 Hz (pipeline export)",
        "eda_transforms": args.augmentations or TRANSFORM_NAMES,
        "eda_params": paper_params(FS_EDA),
        "views": "two augmented views, one transform each (Matton et al. All DAs)",
        "standardisation": "per-window z-score after augmentation",
    }
    with open(seed_dir / "encoder_config.json", "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    groups = [(None, participants)] if args.pool == "global" else [(p, [p]) for p in participants]
    for channel in args.channels:
        sessions_all = channel_sessions(channel, participants, sessions_by_pid, processed_dir,
                                        sensor_dir=sensor_dir)
        for pid, members in groups:
            sessions = [t for t in sessions_all if t[0] in members]
            if not sessions:
                print(f"[matton-ssl] no {channel} training sessions for {pid or 'global'}; skipped")
                continue
            out_path, out_d = encoder_target(out_dir, args.seed, args.pool, channel, pid)
            train_channel(channel, sessions, out_path, out_d, args.seed,
                          args.batch_ssl, args.ssl_epochs,
                          window_sec=args.window_sec, step_sec=args.step_sec,
                          aug_names=args.augmentations, overwrite=args.overwrite)
    print(f"[matton-ssl] done -> {seed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
