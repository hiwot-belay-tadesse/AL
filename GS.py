import json
import os 
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys, shutil, random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import layers
from src.signal_utils import (
    WINDOW_SIZE, STEP_SIZE, create_windows,
    build_simclr_encoder, create_projection_head, train_simclr
)
import src.compare_pipelines as compare_pipelines
from src.classifier_utils import (
    BASE_DATA_DIR, ALLOWED_SCENARIOS,
    load_signal_data, load_label_data, process_label_window
)
from tensorflow.keras import Model
GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 15
WINDOW_LEN                              = WINDOW_SIZE


def _build_XY_from_windows(df: pd.DataFrame):
    """
    Convert a windows DataFrame into X (N, WINDOW_SIZE, 2) and y (N,).
    Assumes df has columns: 'hr_seq', 'st_seq', 'state_val'
    """
    X = np.stack([np.vstack([h, s]).T for h, s in zip(df["hr_seq"], df["st_seq"])])
    y = df["state_val"].values.astype(int)
    return X, y


def _bootstrap_auc(y_true, probs, n_iters=1000, sample_frac=0.7, rng_seed=42):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    rng = np.random.default_rng(rng_seed)

    n = len(y_true)
    m = max(2, int(sample_frac * n))
    aucs = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=m)
        yb = y_true[idx]
        pb = probs[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))

    if not aucs:
        return np.nan, np.nan
    return float(np.mean(aucs)), float(np.std(aucs))


def _best_f1_on_val(y_val, p_val, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_f1, best_thr = -1.0, 0.5
    for t in thresholds:
        yhat = (p_val >= t).astype(int)
        f1 = f1_score(y_val, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_f1, best_thr

# ─── Shared CNN for Global-Supervised ─────────────────────────────────────
def ensure_global_supervised(shared_cnn_root, fruit, scenario, all_splits, uid, dropout_rate):
    """
    Train (or load) a single global CNN on ALL users' train-day windows,
    validating only on the target user's validation windows.

    Returns:
        m     : the trained Keras model
        sdir  : the directory where the model was saved/loaded
    """
    sdir = Path(shared_cnn_root) / f"{fruit}_{scenario}"
    sdir.mkdir(parents=True, exist_ok=True)
    model_path = sdir / 'cnn_classifier.keras'
    if model_path.exists():
        m = load_model(model_path)
        return m, sdir

    # 1) Gather all users' TRAIN windows
    X_list, y_list = [], []
    for u, pairs in ALLOWED_SCENARIOS.items():
        if (fruit, scenario) not in pairs:
            continue
        tr_days_u, _, _ = all_splits.get(u, ([], [], []))
        if len(tr_days_u) == 0:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if len(orig_neg) < len(pos_df):
            extra = compare_pipelines.derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_u = compare_pipelines.collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
        for h_seq, s_seq, label in zip(df_u['hr_seq'], df_u['st_seq'], df_u['state_val']):
            X_list.append(np.vstack([h_seq, s_seq]).T)
            y_list.append(label)

    if not X_list:
        raise RuntimeError("No train windows for global-supervised!")

    X = np.stack(X_list)
    y = np.array(y_list)

    # 2) Build the target user's VAL set
    tr_days_u, val_days_u, _ = all_splits[uid]
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    if neg_df_u.empty or len(neg_df_u) < len(pos_df_u):
        neg_df_u = compare_pipelines.derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))

    df_val_u = compare_pipelines.collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)

    def build_XY(df):
        X = np.stack([np.vstack([h,s]).T for h,s in zip(df['hr_seq'], df['st_seq'])])
        return X, df['state_val'].values

    X_val_u, y_val_u = build_XY(df_val_u)

    # 3) Build & compile the model
    inp = layers.Input(shape=(WINDOW_SIZE, 2))
    x = layers.Conv1D(64, 8, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(4, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(8, activation='relu')(se)
    se = layers.Dense(128, activation='sigmoid')(se)
    se = layers.Reshape((1, 128))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)

    lstm_out = layers.LSTM(128)(inp)
    lstm_out = layers.Dropout(0.5)(lstm_out)

    combined = layers.concatenate([cnn_out, lstm_out])
    out = layers.Dense(1, activation='sigmoid')(combined)

    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # 4) Compute class weights
    classes = np.unique(y)
    if len(classes) == 2:
        cw_vals = compute_class_weight('balanced', classes=classes, y=y)
        class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    else:
        class_weight = {0:1, 1:1}

    # 5) Callbacks
    es = EarlyStopping(monitor='val_loss', patience=GS_PATIENCE, restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 6) SINGLE TRAINING pass
    hist = m.fit(
        X, y,
        validation_data=(X_val_u, y_val_u),
        batch_size=GS_BATCH,
        epochs=GS_EPOCHS,
        class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=2
    )

    # 7) Save & plot
    compare_pipelines.plot_clf_losses(hist.history['loss'], hist.history['val_loss'], sdir, 'global_cnn_lstm_loss')
    m.save(model_path)
    return m, sdir

def _mc_dropout_predict(model, X, T):
    """
    Mean/std under MC Dropout. IMPORTANT: BatchNorm should be frozen outside.
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    preds = []
    for _ in range(T):
        p = model(X, training=True).numpy().ravel()
        preds.append(p)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def run_active_learning_global_supervised(
    args,
    BASE_DATA_DIR,
    ALLOWED_SCENARIOS,
    load_signal_data,
    load_label_data,
    compare_pipelines,  # must provide: derive_negative_labels, ensure_train_val_test_days, collect_windows, ensure_global_supervised
    K: int,
    Budget: int | None,
    T: int = 30,
    acquisition: str = "uncertainty",  # "random" or "uncertainty"
    gs_epochs: int = 30,
    gs_batch: int = 32,
    verbose_fit: int = 0,
    unlabeled_frac: float | None = None,
):
    """
    End-to-end Active Learning for the GLOBAL-SUPERVISED CNN.
    - Builds splits across all users (all_splits, all_negatives)
    - Loads/trains a global supervised CNN architecture (can start from scratch or from ensure_global_supervised)
    - Creates a GLOBAL labeled pool and GLOBAL unlabeled pool (from ALL users' train windows)
    - Keeps validation/test fixed for the TARGET user (args.user)
    - Runs AL rounds using random or MC-dropout uncertainty acquisition on the unlabeled pool
    - Retrains model each round on expanded labeled pool
    - Returns: al_progress_df, trained_model, df_tr_labeled, df_tr_unlabeled, df_te, df_val, counts_df
    """

    # ----------------------------
    # 0) Setup dirs
    # ----------------------------
    top_out = Path(args.output_dir)
    user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"
    user_root.mkdir(parents=True, exist_ok=True)

    out_dir = user_root / "global_supervised_al"
    models_d = out_dir / "models_saved"
    results_d = out_dir / "results"
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    shared_cnn_root = top_out / "global_cnns"
    shared_cnn_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1) Build day-level splits + negatives for all users
    # ----------------------------
    all_splits = {}
    all_negatives = {}

    for u, pairs in ALLOWED_SCENARIOS.items():
        if (args.fruit, args.scenario) not in pairs:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")

        if len(orig_neg) < len(pos_df):
            extra = compare_pipelines.derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        all_negatives[u] = neg_df

        try:
            tr_u, val_u, te_u = compare_pipelines.ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as e:
            print(f"Skipping user {u}: {e}")
            continue

        all_splits[u] = (tr_u, val_u, te_u)

    if args.user not in all_splits:
        print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
        sys.exit(0)

    neg_df_u = all_negatives[args.user]
    tr_days_u, val_days_u, te_days_u = all_splits[args.user]

    # ----------------------------
    # 2) Train/load the base global model (optional warm start)
    #    NOTE: This model was trained using the old "all train windows labeled" assumption.
    #    We'll STILL use it as initialization and then do AL properly.
    # ----------------------------
    base_model, src_dir = ensure_global_supervised(
        shared_cnn_root, args.fruit, args.scenario, all_splits, args.user,args.dropout_rate
    )

    # copy artifacts
    for fpath in Path(src_dir).glob("*"):
        if fpath.suffix == ".keras":
            shutil.copy2(fpath, models_d / fpath.name)
        elif fpath.suffix == ".png":
            shutil.copy2(fpath, results_d / fpath.name)

    # We'll clone for safety so each AL run is clean
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())
    ## Recompile with same optimizer/loss/metrics
    model.compile(optimizer=base_model.optimizer.__class__.from_config(base_model.optimizer.get_config()),
                  loss=base_model.loss, metrics=base_model.metrics)

    # ----------------------------
    # 3) Build per-user train_info/val_info (DataFrames)
    # ----------------------------
    train_info = {}
    val_info = {}
    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
        neg_df = all_negatives[u]

        df_tr_u = compare_pipelines.collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_tr_u["user_id"] = u

        df_val_u = compare_pipelines.collect_windows(pos_df, neg_df, hr_df, st_df, val_days)
        df_val_u["user_id"] = u

        train_info[u] = {"days": tr_days.tolist(), "df": df_tr_u}
        val_info[u] = {"days": val_days.tolist(), "df": df_val_u}

    df_all_tr = pd.concat([info["df"] for info in train_info.values()], ignore_index=True)

    # Target user's fixed val/test
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / args.user)
    pos_df_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
    if neg_df_u.empty or len(neg_df_u) < len(pos_df_u):
        neg_df_u = compare_pipelines.derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))
    df_val = val_info[args.user]["df"].copy()
    df_te = compare_pipelines.collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
    df_te["user_id"] = args.user

    # ----------------------------
    # 4) Create GLOBAL labeled/unlabeled pools from df_all_tr
    #    You need a policy for what starts labeled.
    #    Here: start labeled = all windows from TARGET user only (common AL personalization baseline),
    #    and everything else global pool starts unlabeled.
    #    If you want a different init, change this block.
    # ----------------------------
    # df_tr_labeled = df_all_tr[df_all_tr["user_id"] == args.user].copy()
    # df_tr_unlabeled = df_all_tr[df_all_tr["user_id"] != args.user].copy()
    df_tr_labeled, df_tr_unlabeled = train_test_split(df_all_tr, test_size=unlabeled_frac, random_state=42, stratify=df_all_tr['state_val'], shuffle=True)

    
    # Make sure indices are unique/stable for dropping
    df_tr_labeled.reset_index(drop=False, inplace=True)   # keep old index in 'index'
    df_tr_unlabeled.reset_index(drop=False, inplace=True)

    # We'll treat the new DataFrame index as "pool id"
    df_tr_labeled.set_index("index", inplace=True, drop=True)
    df_tr_unlabeled.set_index("index", inplace=True, drop=True)
    # ----------------------------
    # 5) Precompute fixed val/test tensors
    # ----------------------------
    X_val, y_val = _build_XY_from_windows(df_val)
    X_te, y_te = _build_XY_from_windows(df_te)
    # ----------------------------
    # 5.1) Precompute fixed val/test tensors
    # ----------------------------
    X_full, y_full = _build_XY_from_windows(df_all_tr)
    clf_probs_full = model.predict(X_full, verbose=0).ravel()
    
    # ----------------------------
    # 6) Helper: train/eval on current labeled set
    # ----------------------------
    def train_and_eval(current_df_lab: pd.DataFrame, init_model: tf.keras.Model):
        X_lab, y_lab = _build_XY_from_windows(current_df_lab)

        # class weights (balanced)
        classes = np.unique(y_lab)
        if len(classes) == 2:
            cw_vals = compute_class_weight("balanced", classes=classes, y=y_lab)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
        else:
            class_weight = None

        # Fit (fresh weights each round)
        m = tf.keras.Model.from_config(init_model.get_config())
        m.set_weights(init_model.get_weights())
        # Recompile with a standard optimizer to avoid non-serializable configs.
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

        # basic callbacks
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
        )

        fit_kw = dict(
            epochs=gs_epochs,
            batch_size=gs_batch,
            verbose=verbose_fit,
            validation_data=(X_val, y_val),
            callbacks=[es],
            shuffle=True,
        )
        if class_weight is not None:
            fit_kw["class_weight"] = class_weight

        m.fit(X_lab, y_lab, **fit_kw)

        # Evaluate
        p_val = m.predict(X_val, verbose=0).ravel()
        p_te = m.predict(X_te, verbose=0).ravel()

        val_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) == 2 else np.nan
        test_auc_mean, test_auc_std = _bootstrap_auc(y_te, p_te)
        best_f1, best_thr = _best_f1_on_val(y_val, p_val)

        return m, val_auc, test_auc_mean, test_auc_std, best_f1, best_thr

    # ----------------------------
    # 7) Acquisition functions
    # ----------------------------
    def pick_random(df_unlb, k):
        ''' 
        Random without replacement
        '''
        k = min(k, len(df_unlb))
        idx = random.sample(list(df_unlb.index), k)
        return idx, df_unlb.loc[idx]

    def pick_uncertainty(m, df_unlb, k, Tmc):
        # Build X for unlabeled
        X_unlb, _ = _build_XY_from_windows(df_unlb)

        # Freeze BN before MC (important)
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        _, std_p = _mc_dropout_predict(m, X_unlb, T=Tmc)

        df_tmp = df_unlb.copy()
        df_tmp["uncertainty"] = std_p
        df_tmp = df_tmp.sort_values("uncertainty", ascending=False)

        k = min(k, len(df_tmp))
        idx = df_tmp.head(k).index.tolist()
        
        return idx, df_tmp.loc[idx]

    # ----------------------------
    # 8) Round 0 train/eval
    # ----------------------------
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    if Budget is None:
        if unlabeled_frac is None:
            raise ValueError("Provide Budget or unlabeled_frac for AL rounds.")
        Budget = int(np.ceil(unlabeled_frac * len(df_tr_unlabeled) / K))
        Budget = max(1, Budget)
        args.Budget = Budget

    print("\n=== GLOBAL-SUPERVISED AL INPUT DEBUG ===")
    print(f"global labeled  : n={len(df_tr_labeled)} pos={(df_tr_labeled['state_val']==1).sum()} neg={(df_tr_labeled['state_val']==0).sum()}")
    print(f"global unlabeled: n={len(df_tr_unlabeled)}")
    print(f"val fixed       : n={len(df_val)} pos={(df_val['state_val']==1).sum()} neg={(df_val['state_val']==0).sum()}")
    print(f"test fixed      : n={len(df_te)} pos={(df_te['state_val']==1).sum()} neg={(df_te['state_val']==0).sum()}")

    results = []
    participants_count_per_round = {}
    queried_all = []


    
    m_cur, val_auc, auc_m, auc_s, best_f1, best_thr = train_and_eval(df_tr_labeled, model)
    print(f"Round 0: TestAUC={auc_m:.3f}±{auc_s:.3f} | ValAUC={val_auc:.3f} | BestF1={best_f1:.3f}")

    _, _, auc_mean, auc_std, _, _ = train_and_eval(df_all_tr, model)
    print(f"Upper Bound (all labeled) TestAUC={auc_mean:.3f}±{auc_std:.3f}")
    ###save upper bound to file 
    upper_bound_path = results_d / "upper_bound_auc.npy"
    np.save(upper_bound_path, np.array([auc_mean, auc_std]))
    print(f"Saved upper bound AUC to: {upper_bound_path}")

    results.append(pd.DataFrame([{
        "round": 0,
        "AUC_Mean": auc_m,
        "AUC_STD": auc_s,
        "Val_AUC": val_auc,
        "Best_F1": best_f1,
        "Best_Thr": best_thr,
        "labeled_n": len(df_tr_labeled),
        "unlabeled_n": len(df_tr_unlabeled),
    }]))

    # ----------------------------
    # 9) Active learning loop
    # ----------------------------
    for b in range(1, Budget + 1):
        if len(df_tr_unlabeled) == 0:
            break

        k_actual = min(K, len(df_tr_unlabeled))
        print(f"\nAL round {b}/{Budget} (global labeled={len(df_tr_labeled)}, global unlabeled={len(df_tr_unlabeled)}, K={k_actual})")

        if acquisition == "random":
            q_idx, df_q = pick_random(df_tr_unlabeled, k_actual)
        elif acquisition == "uncertainty":
            q_idx, df_q = pick_uncertainty(m_cur, df_tr_unlabeled, k_actual, Tmc=T)
        else:
            raise ValueError("acquisition must be 'random' or 'uncertainty'")

        queried_all.extend(q_idx)
        participants_count_per_round[b] = Counter(df_q["user_id"].tolist())

        # Move from unlabeled -> labeled
        df_tr_labeled = pd.concat([df_tr_labeled, df_tr_unlabeled.loc[q_idx]], axis=0)
        df_tr_unlabeled = df_tr_unlabeled.drop(index=q_idx)
        # 
        # retrain + eval
        m_cur, val_auc, auc_m, auc_s, best_f1, best_thr = train_and_eval(df_tr_labeled, m_cur)
        print(f"  → Round {b}: TestAUC={auc_m:.3f}±{auc_s:.3f} | ValAUC={val_auc:.3f} | BestF1={best_f1:.3f}")

        results.append(pd.DataFrame([{
            "round": b,
            "AUC_Mean": auc_m,
            "AUC_STD": auc_s,
            "Val_AUC": val_auc,
            "Best_F1": best_f1,
            "Best_Thr": best_thr,
            "labeled_n": len(df_tr_labeled),
            "unlabeled_n": len(df_tr_unlabeled),
        }]))

    al_progress = pd.concat(results, ignore_index=True)

    # Participant counts table
    counts_df = (
        pd.DataFrame.from_dict(participants_count_per_round, orient="index")
        .fillna(0)
        .astype(int)
    )
    counts_df.index.name = "round"
    counts_df.columns.name = "user_id"

    # Save outputs
    al_progress.to_csv(results_d / "al_progress.csv", index=False)
    counts_df.to_csv(results_d / "queried_participants_counts.csv")

    # Save final model
    m_cur.save(models_d / "final_global_supervised_al.keras")

    return al_progress, m_cur, df_tr_labeled, df_tr_unlabeled, df_val, df_te, counts_df


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--user", default="ID21")
    pa.add_argument("--fruit", default="Nectarine")
    pa.add_argument("--scenario", default="Crave")
    pa.add_argument("--output_dir", default="output_GS")
    pa.add_argument("--K", type=int, default=10)
    pa.add_argument("--dropout_rate", type=float, default=0.3)
    pa.add_argument("--Budget", type=int, default=None)
    pa.add_argument("--unlabeled_frac", type=float, default=0.7)
    pa.add_argument("--T", type=int, default=50)
    pa.add_argument("--acquisition", default="uncertainty", choices=["uncertainty", "random"])
    pa.add_argument("--gs_epochs", type=int, default=200)
    pa.add_argument("--gs_batch", type=int, default=32)
    pa.add_argument("--verbose_fit", type=int, default=0)
    args = pa.parse_args()

    run_active_learning_global_supervised(
        args=args,
        BASE_DATA_DIR=BASE_DATA_DIR,
        ALLOWED_SCENARIOS=ALLOWED_SCENARIOS,
        load_signal_data=load_signal_data,
        load_label_data=load_label_data,
        compare_pipelines=compare_pipelines,
        K=args.K,
        Budget=args.Budget,
        T=args.T,
        acquisition=args.acquisition,
        gs_epochs=args.gs_epochs,
        gs_batch=args.gs_batch,
        verbose_fit=args.verbose_fit,
        unlabeled_frac=args.unlabeled_frac,
    )
    summary = {
        "user": args.user,
        "fruit": args.fruit,
        "scenario": args.scenario,
        "K": args.K,
        "Budget": args.Budget,
        "unlabeled_frac": args.unlabeled_frac,
        "T": args.T,
        "acquisition": args.acquisition,
        "gs_epochs": args.gs_epochs,
        "gs_batch": args.gs_batch,
        "dropout_rate": args.dropout_rate,
    }
    results_dir = Path(args.output_dir) / args.user / f"{args.fruit}_{args.scenario}" / "global_supervised_al" / "results" / args.acquisition
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "exp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved exp_summary.json to:", summary_path)
    breakpoint()
if __name__ == "__main__":
    main()
 
