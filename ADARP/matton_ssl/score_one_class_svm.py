"""One-class SVM on the Matton-augmentation encoders (4 Hz EDA), same protocol as ADARP/one_class_svm.py.

The encoders trained by `train_encoders.py` take 240-sample 4 Hz EDA windows,
so the pipeline's 30-point scoring path cannot feed them. This script:

  1. takes the pipeline's labelled windows (`load_windows`) and its session split
     (`eda_baseline.assign_split`, val folded into train), and asserts that no
     test window overlaps a train window in time;
  2. cuts each labelled window out of the raw 4 Hz EDA stream by its timestamps
     (`data4hz.labelled_eda_windows`), z-scores it per window, and encodes it;
     HR windows come from the saved 60-sample 1 Hz arrays, z-scored the same way;
  3. fits StandardScaler + OneClassSVM on the training STRESS windows only and
     scores the test windows, global and personalized, with bootstrap CIs --
     by calling `one_class_svm.run_global` / `run_personalized` unchanged.

    python ADARP/matton_ssl/score_one_class_svm.py                      # eda, seed 42
    python ADARP/matton_ssl/score_one_class_svm.py --channels hr_eda
    python ADARP/matton_ssl/score_one_class_svm.py --run_dir ADARP/matton_ssl/encoders --seed 43

Writes <results_dir>/one_class_svm_matton_<channels>.csv and _summary.json, and
(with --save_embeddings) the encoded windows with provenance as .npz.
"""

import argparse
import json
import os
import sys
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

from preprocess_adarp_data import PROCESSED_DIR, load_windows  # noqa: E402
from eda_baseline import assert_no_overlap, assign_split, mark_split  # noqa: E402
from one_class_svm import (  # noqa: E402  (protocol reused unchanged)
    GAMMA, KERNEL, NU, RESULT_COLS, run_global, run_personalized,
)
from data4hz import FS_EDA, labelled_eda_windows, load_eda_4hz, zscore_rows  # noqa: E402
from train_encoders import OUT_DIR, WINDOW_SEC, encoder_target  # noqa: E402

RESULTS_DIR = _HERE / "results"


def _present(path):
    return path.exists() and path.stat().st_size > 0


def load_encoder(path):
    import keras
    return keras.models.load_model(path, compile=False)


def encoders_for(run_dir, seed, channels, pid=None):
    """(enc_hr, enc_eda) or (None, reason). pid=None -> the shared global encoders."""
    hr_path, _ = encoder_target(run_dir, seed, "global" if pid is None else "personal", "hr", pid)
    eda_path, _ = encoder_target(run_dir, seed, "global" if pid is None else "personal", "eda", pid)
    needed = [eda_path] + ([hr_path] if channels == "hr_eda" else [])
    missing = [str(p) for p in needed if not _present(p)]
    if missing:
        return None, f"encoder missing: {missing}"
    enc_eda = load_encoder(eda_path)
    enc_hr = load_encoder(hr_path) if channels == "hr_eda" else None
    return (enc_hr, enc_eda), None


def encode(enc_hr, enc_eda, X_eda, X_hr):
    S = enc_eda.predict(zscore_rows(X_eda)[..., None], verbose=0)
    if enc_hr is None:
        return S.astype("float32")
    H = enc_hr.predict(zscore_rows(X_hr)[..., None], verbose=0)
    return np.concatenate([H, S], axis=1).astype("float32")


def build_table(processed_dir, seed, verbose=True):
    """Window table with split + the raw 4 Hz EDA and 1 Hz HR arrays aligned to it."""
    eda1, hr, meta = load_windows(processed_dir)
    table = meta.copy()
    table["participant"] = table["participant"].astype(str)
    table["label"] = table["label"].astype(int)

    frames = {pid: load_eda_4hz(pid) for pid in sorted(table["participant"].unique())}
    X_eda = labelled_eda_windows(table, frames, fs=FS_EDA, window_sec=WINDOW_SEC)

    ok = ~np.isnan(X_eda).any(axis=1)
    if verbose and (~ok).any():
        print(f"[matton-ocsvm] {int((~ok).sum())} of {len(ok)} windows not found whole on the "
              f"4 Hz stream; dropped", flush=True)
    table, X_eda, hr = table[ok].reset_index(drop=True), X_eda[ok], hr[ok]

    table = mark_split(table, assign_split(processed_dir, seed=seed))
    keep = (table["split"] != "unassigned").to_numpy()
    table, X_eda, hr = table[keep].reset_index(drop=True), X_eda[keep], hr[keep]
    assert_no_overlap(table[table["split"] == "train"], table[table["split"] == "test"])
    if verbose:
        tr, te = table["split"] == "train", table["split"] == "test"
        print(f"[matton-ocsvm] seed={seed}: train={int(tr.sum())} (pos={int(table.loc[tr, 'label'].sum())}) "
              f"test={int(te.sum())} (pos={int(table.loc[te, 'label'].sum())}); "
              f"EDA windows {X_eda.shape[1]} samples @ {FS_EDA:g} Hz", flush=True)
    return table, X_eda, hr.astype(np.float32)


def run(processed_dir=PROCESSED_DIR, run_dir=OUT_DIR, results_dir=RESULTS_DIR, seed=42,
        channels="eda", n_boot=1000, nu=NU, gamma=GAMMA, save_embeddings=False, verbose=True):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    gamma_val = gamma if isinstance(gamma, str) else float(gamma)
    svm_kw = dict(nu=nu, gamma=gamma_val)
    notes = {"encoder_run_dir": str(run_dir), "channels": channels}

    table, X_eda, X_hr = build_table(processed_dir, seed, verbose)
    pids = table["participant"].to_numpy()

    encs, reason = encoders_for(run_dir, seed, channels)
    if encs is None:
        if verbose:
            print(f"[matton-ocsvm] global arm skipped: {reason}", flush=True)
        g_rows, g_skipped, Z_global = [], ["all"], None
        notes["global_skipped"] = reason
    else:
        Z_global = encode(*encs, X_eda, X_hr)
        g_rows, g_skipped = run_global(table, Z_global, n_boot, seed, verbose, **svm_kw)

    def features_for(pid, block):
        encs, reason = encoders_for(run_dir, seed, channels, pid=pid)
        if encs is None:
            return None, reason
        m = pids == pid
        return encode(*encs, X_eda[m], X_hr[m]), None

    p_rows, p_skipped = run_personalized(table, features_for, n_boot, seed,
                                         verbose=verbose, **svm_kw)

    tag = f"matton_{channels}"
    results = pd.DataFrame(g_rows + p_rows)
    results["features"] = tag
    results = results.reindex(columns=RESULT_COLS)
    out_csv = results_dir / f"one_class_svm_{tag}.csv"
    results.to_csv(out_csv, index=False)

    with open(results_dir / f"one_class_svm_{tag}_summary.json", "w") as fh:
        json.dump({
            "processed_dir": str(processed_dir), "seed": seed, "n_boot": n_boot,
            "features": tag, "eda_fs_hz": FS_EDA, "window_sec": WINDOW_SEC,
            "svm": {"kernel": KERNEL, "nu": nu, "gamma": gamma_val},
            "trained_on": "training stress windows only; scaler fitted on the same",
            "skipped": {"global": g_skipped, "personalized": p_skipped},
            **notes,
        }, fh, indent=2, default=str)

    if save_embeddings and Z_global is not None:
        np.savez_compressed(
            results_dir / f"embeddings_global_{tag}_seed{seed}.npz", Z=Z_global,
            participant=pids, window_id=table["window_id"].astype(str).to_numpy(),
            session=table["session"].astype(str).to_numpy(),
            label=table["label"].to_numpy(), split=table["split"].astype(str).to_numpy())

    if verbose:
        cols = ["participant", "model", "train_pos", "test_pos", "test_neg",
                "auroc", "ci_low", "ci_high", "skip_reason"]
        with pd.option_context("display.width", 160, "display.max_rows", 100):
            print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"[matton-ocsvm] -> {out_csv}")
    return results


def parse_args():
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--processed_dir", default=str(PROCESSED_DIR))
    pa.add_argument("--run_dir", default=str(OUT_DIR),
                    help="Encoder tree from train_encoders.py (holds seed_<seed>/).")
    pa.add_argument("--results_dir", default=str(RESULTS_DIR))
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--channels", default="eda", choices=["eda", "hr_eda"])
    pa.add_argument("--n_boot", type=int, default=1000)
    pa.add_argument("--nu", type=float, default=NU)
    pa.add_argument("--gamma", default=GAMMA)
    pa.add_argument("--save_embeddings", action="store_true")
    pa.add_argument("--quiet", action="store_true")
    return pa.parse_args()


def main():
    a = parse_args()
    run(processed_dir=Path(a.processed_dir), run_dir=Path(a.run_dir),
        results_dir=Path(a.results_dir), seed=a.seed, channels=a.channels,
        n_boot=a.n_boot, nu=a.nu, gamma=a.gamma, save_embeddings=a.save_embeddings,
        verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
