from multiprocessing import pool
import os
import sys
import shutil
import numpy as np
import pandas as pd
import random
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import pairwise_distances
from collections import Counter
import numpy as np
import pandas as pd
## import typiclust
import uq_utility


# Add project root to path for src imports
# utility.py is at src/AL_MC/AL/utility.py, so go up 3 levels to project root
sys.path.append(str(Path(__file__).resolve().parents[3]))

import preprocess
from src.compare_pipelines import select_threshold_train
from src.chart_utils import bootstrap_threshold_metrics
from sklearn.metrics import roc_auc_score, log_loss
# from src.explanations.mini_test import X_test, X_train
os.environ["KERAS_BACKEND"] = "tensorflow"
# Set deterministic operations for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from src import compare_pipelines

BATCH_SSL, SSL_EPOCHS = 32, 100

def compute_user_stats(df_all):
    """
    df_all: the full dataset (labeled + unlabeled) with user_id, hr_seq, st_seq
    returns: dict of per-user signal statistics to inform augmentation bounds
    """
    user_stats = {}

    for uid, group in df_all.groupby('user_id'):
        hr_means = group['hr_seq'].apply(np.mean)
        st_means = group['st_seq'].apply(np.mean)

        user_stats[uid] = {
            'hr_std': hr_means.std(),
            'hr_mean': hr_means.mean(),
            'st_std': st_means.std(),
            'st_mean': st_means.mean(),
        }

    return user_stats

def compute_n_slices(df_queried, budget, target_ratio=0.43 ):
    """
    compute n_slices_pos to achieve target_ratio after slicing
    target_ratio = 0.43 matches your true event rate
    
    """
    n_pos = (df_queried['state_val'] == 1).sum()
    n_neg = (df_queried['state_val'] == 0).sum()


    total_pos_target = budget * target_ratio
    total_neg_target = budget * (1 - target_ratio)

    n_slices_pos = max(0, round(total_pos_target / n_pos) - 1) if n_pos > 1 else 0
    n_slices_neg = max(0, round(total_neg_target / n_neg) - 1) if n_neg > 1 else 0


    return n_slices_neg, n_slices_pos

def augment_labeled_windows(df_queried, target_ratio=0.43, 
                              min_len=24, n_aug=3):
    '''
    augment each queried window by slicing into subwindows
    '''

    n_slices_neg, n_slices_pos = compute_n_slices(
        df_queried, budget=30, target_ratio=target_ratio
    )

    rows = []
    for _, row in df_queried.iterrows():
        hr  = np.array(row['hr_seq'])
        st  = np.array(row['st_seq'])
        y   = row['state_val']
        uid = row['user_id']
        T   = len(hr)

        rows.append({'user_id': uid, 'hr_seq': hr, 
                     'st_seq': st, 'state_val': y})

        # n = n_slices_pos if y == 1 else n_slices_neg
        n = n_aug
        
        for _ in range(n):
            length = np.random.randint(min_len, T + 1)
            start  = np.random.randint(0, T - length + 1)
            hr_sl  = np.pad(hr[start:start+length], 
                           (0, T-length), mode='edge')
            st_sl  = np.pad(st[start:start+length], 
                           (0, T-length), mode='constant')
            rows.append({'user_id': uid, 'hr_seq': hr_sl, 
                        'st_seq': st_sl, 'state_val': y})

    return pd.DataFrame(rows).reset_index(drop=True)



def split_fingerprint(
    df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    label_col: str = "state_val",
    time_candidates: tuple[str, ...] = ("hawaii_createdat_time", "datetime_local"),
) -> tuple[str, dict]:
    """
    Deterministic SHA-256 fingerprint of a labeled split based on
    (user_id, timestamp, label), sorted for stable ordering.
    """
    if df is None or len(df) == 0:
        return hashlib.sha256(b"").hexdigest(), {
            "rows": 0,
            "time_col": None,
        }

    time_col = next((c for c in time_candidates if c in df.columns), None)

    users = (
        df[user_col].astype(str).tolist()
        if user_col in df.columns
        else [""] * len(df)
    )
    labels = (
        pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int).astype(str).tolist()
        if label_col in df.columns
        else [""] * len(df)
    )
    if time_col is not None:
        times = (
            pd.to_datetime(df[time_col], errors="coerce")
            .dt.tz_localize(None)
            .astype(str)
            .tolist()
        )
    else:
        times = [""] * len(df)

    keys = [f"{u}|{t}|{y}" for u, t, y in zip(users, times, labels)]
    keys_sorted = sorted(keys)
    payload = "\n".join(keys_sorted).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest, {"rows": int(len(df)), "time_col": time_col}


def fit_and_eval(fit_kwargs, clf_builder, input_dim, X_lab, y_lab, X_val, y_val, X_test, y_test, CLF_PATIENCE, dropout_rate, clf=None):
    """
    Train a model and evaluate; also pick a decision threshold on the labeled set.
    args:
        fit_kwargs:  dict of kwargs for model.fit()
        clf_builder: function that builds and compiles a tf.keras.Model
        input_dim:   Number of input features(X_train.shape[0])
        X_lab:       Labeled training data, shape (n_samples, n_features)
        y_lab:       Labeled training labels, shape (n_samples,)
        X_val:       Validation data, shape (n_samples, n_features)
        y_val:       Validation labels, shape (n_samples,)
        X_test:      Test data, shape (n_samples, n_features)
        y_test:      Test labels, shape (n_samples,)
    returns:
        df_boot:     DataFrame of bootstrap results on test set
        auc_m:       Mean AUC on test set
        auc_s:       Stddev of AUC on test set
        clf:         Trained classifier model
        
    """

    #________________SEED FOR REPRODUCIBILITY________________
    # Reset seeds before each model creation for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #_______________________________________________________
    if X_lab is None or len(X_lab) == 0:
        raise ValueError("Empty training set in fit_and_eval.")
    clf, es = clf_builder(input_dim, CLF_PATIENCE, dropout_rate, seed)
    # if clf is None:
    #     clf, es = clf_builder(input_dim, CLF_PATIENCE, dropout_rate)
    # else:
    #     es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE, restore_best_weights=True, verbose=0)
    
    # NEW: compute balanced class weights
    # cw_vals = compute_class_weight('balanced', classes=np.unique(y_lab), y=y_lab)
    # class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    # y_lab = y_lab.astype(int)c
    
    unique_classes = np.unique(y_lab)

    assert X_lab.shape[0] > 0, f"Empty X_lab: {X_lab.shape}"
    assert y_lab.shape[0] > 0, f"Empty y_lab: {y_lab.shape}"
    assert X_lab.shape[0] == y_lab.shape[0], f"Mismatch X_lab {X_lab.shape} vs y_lab {y_lab.shape}"


    cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_lab)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    if len(class_weight) < 2:
        print(f"Warning: Only one class present in y_lab. Class weights: {class_weight}")
    
    # Train the model
    fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}
    
    ####[NEW] fit with class weights
    # if len(class_weight) == 2:
    #     fit_kwargs_with_callbacks['class_weight'] = class_weight
    
    # Remove any non-fit() arguments that might have been accidentally included
    # fit_kwargs_with_callbacks.pop('CLF_PATIENCE', None)
    # fit_kwargs_with_callbacks.pop('dropout_rate', None)
    # Ensure shuffle is controlled for reproducibility (default to False if not specified)
    if 'shuffle' not in fit_kwargs_with_callbacks:
        fit_kwargs_with_callbacks['shuffle'] = True
    if 'callbacks' in fit_kwargs_with_callbacks:
        callbacks = fit_kwargs_with_callbacks['callbacks']
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks = callbacks + [es]
        fit_kwargs_with_callbacks['callbacks'] = callbacks
    else:
        fit_kwargs_with_callbacks['callbacks'] = [es]
    
    # Right before clf.fit():
    print(f"\n=== Class Weights Check ===")
    # print(f"Class weights: {class_weight}")
    print(f"Training class distribution: Positive (1): {(y_lab == 1).sum()}, Negative (0): {(y_lab == 0).sum()}")

    # history = clf.fit(X_lab, y_lab, class_weight=class_weight, **fit_kwargs_with_callbacks)
    history = clf.fit(X_lab, y_lab, **fit_kwargs_with_callbacks)

    from sklearn.metrics import roc_auc_score
    probs_train = clf.predict(X_lab, verbose=0).ravel()
   
    
    # print(f"Train AUC = {train_auc:.3f}")
    probs_te = clf.predict(X_test, verbose=0).ravel()
    
    ###predict on validation set for auc_val 
    probs_val = clf.predict(X_val, verbose=0).ravel()
    val_auc = roc_auc_score(y_val, probs_val)
    print(f"Validation AUC = {val_auc:.3f}")
    # Debug: monitor test size, prediction spread, and direct AUC per round.
    print(f"Test size: {len(y_test)}")
    print(
        f"Test probs: min={probs_te.min():.3f}, "
        f"max={probs_te.max():.3f}, mean={probs_te.mean():.3f}"
    )
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_test)) == 2:
        direct_auc = roc_auc_score(y_test, probs_te)
        flipped_auc = roc_auc_score(y_test, 1 - probs_te)
        print(f"Direct Test AUC: {direct_auc:.3f}")
        print(f"Flipped Test AUC: {flipped_auc:.3f}")
    
    ##New: direct bootstrap function for the subset experiments
    auc_mean, auc_std, valid_frac = bootstrap_auc(y_test, probs_te)
    auc_m_train, auc_s_train, _ = bootstrap_auc(y_lab, probs_train)
    auc_m_val, auc_s_val, _ = bootstrap_auc(y_val, probs_val)
    ## f1 computation
    from sklearn.metrics import roc_auc_score, f1_score

    thresholds = np.linspace(0.05, 0.95, 19)
    probs_ref = clf.predict(X_val, verbose=0).ravel()
    y_ref = y_val
    f1_vals = []
    for t in thresholds:
        y_pred = (probs_ref >= t).astype(int)
        f1_vals.append(f1_score(y_ref, y_pred))

    best_idx = int(np.argmax(f1_vals))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1_vals[best_idx])
    
    return auc_mean, auc_std, best_f1, auc_m_val,auc_s_val, auc_m_train, auc_s_train,  clf

def encode(df, enc_hr, enc_st):
    hr_seq = np.stack(df['hr_seq'])[..., None]
    st_seq = np.stack(df['st_seq'])[..., None]
    H = enc_hr.predict(hr_seq, verbose=0)
    S = enc_st.predict(st_seq, verbose=0)
    return H, S


# def run_al( Aq, df_tr_labeled, df_tr_unlabeled, df_val, df_te, enc_hr,
#         enc_st,  clf_builder, mc_predict, K, budget, T, CLF_PATIENCE, dropout_rate, fit_kwargs,  pool,
#         models_d, results_d, gs_src_dir=None, active_model=None,init_weights_trained=None):
#     '''Active learning loop'''
#     ## unpack from fit_kwargs
    
#     print(f"\n=== run_al INPUT DEBUG ===")
#     print(f"df_tr_labeled - Positive: {(df_tr_labeled['state_val'] == 1).sum()}, Negative: {(df_tr_labeled['state_val'] == 0).sum()}")
#     print(f"df_val - Positive: {(df_val['state_val'] == 1).sum()}, Negative: {(df_val['state_val'] == 0).sum()}")
#     print(f"df_te - Positive: {(df_te['state_val'] == 1).sum()}, Negative: {(df_te['state_val'] == 0).sum()}")
    
    
#     if pool == "personal" or pool == "global":
#         H_tr_labeled,  S_tr_labeled  = encode(df_tr_labeled, enc_hr, enc_st)
#         H_tr_unlabeled, S_tr_unlabeled = encode(df_tr_unlabeled, enc_hr, enc_st)

#         H_val, S_val = encode(df_val, enc_hr, enc_st)
#         H_te,  S_te  = encode(df_te, enc_hr, enc_st)

#         ##Encoded representations
#         Z_tr_labeled   = np.concatenate([H_tr_labeled,  S_tr_labeled],  axis=1).astype('float32')
#         y_tr_labeled   = df_tr_labeled['state_val'].values.astype('float32')
#         Z_tr_unlabeled = np.concatenate([H_tr_unlabeled, S_tr_unlabeled],axis=1).astype('float32')
#         y_tr_unlabeled = df_tr_unlabeled['state_val'].values.astype('float32')

        
#         # Fixed validation and test sets (do NOT change across rounds)
#         Z_val  = np.concatenate([H_val, S_val], axis=1).astype('float32')
#         y_val  = df_val['state_val'].values.astype('float32')
#         Z_te   = np.concatenate([H_te,  S_te],  axis=1).astype('float32')
#         y_te   = df_te['state_val'].values.astype('float32')
#     elif pool == "global_supervised":
        
#         # mu_global  = np.array([-1.8227721e-10,  1.2759405e-10], dtype=np.float32)
#         # std_global = np.array([0.97679454, 0.9546299], dtype=np.float32)
#         Z_tr_labeled, y_tr_labeled = uq_utility._build_XY_from_windows(df_tr_labeled, "none")
#         Z_tr_unlabeled, y_tr_unlabeled = uq_utility._build_XY_from_windows(df_tr_unlabeled, "none")

#         Z_val, y_val = uq_utility._build_XY_from_windows(df_val, "none")
#         Z_te, y_te = uq_utility._build_XY_from_windows(df_te, "none")
#     else:
#         raise ValueError(f"Unknown pool type: {pool}. Must be 'personal', 'global', or 'global_supervised'.")


#     def compute_density(Z, k=7):
#         # pairwise distance matrix
#         dist = pairwise_distances(Z, Z, metric='euclidean')

#         # sort distances and take mean distance to the kNN
#         knn_distances = np.sort(dist, axis=1)[:, 1:k+1]
#         density = 1.0 / (1e-6 + np.mean(knn_distances, axis=1))
#         return density
    
#     def pick_random(K, df_tr_unlabeled):
#         """Return K random indices from the unlabeled pool."""
#         if len(df_tr_unlabeled) < K:
#             K = len(df_tr_unlabeled)
#         queried_indices = random.sample(list(df_tr_unlabeled.index), K)
#         df_queried = df_tr_unlabeled.loc[queried_indices]
#         return queried_indices, df_queried
   
#     def pick_most_uncertain(
#         clf, df_tr_unlabeled, Z_tr_unlabeled, K, T,mc_predict
#     ):
#         """MC‑Dropout acquisition on encoded unlabeled set."""
#         # ---------------------------
#         # Freeze BatchNorm for MC dropout
#         # ---------------------------
#         for layer in clf.layers:
#             if isinstance(layer, tf.keras.layers.BatchNormalization):
#                 layer.trainable = False

#         # ---------------------------
#         # Step 1: MC-Dropout uncertainty
#         # ---------------------------
#         _, std_p, BALD = mc_predict(clf, Z_tr_unlabeled, T=T)

#         if len(std_p) != len(df_tr_unlabeled):
#             raise ValueError(
#                 f"std_p has {len(std_p)} entries but df_tr_unlabeled has {len(df_tr_unlabeled)} rows."
#             )
#         # Attach uncertainty to rows (keeping their original indices!)
#         df_unlb = df_tr_unlabeled.copy()
#         df_unlb["uncertainty"] = std_p
#         # df_unlb["uncertainty"] = BALD


#         # Sort high → low uncertainty
#         df_unlb = df_unlb.sort_values("uncertainty", ascending=False)

#         # ---------------------------
#         # Step 2: Select top‑K uncertain windows
#         # ---------------------------
#         queried_indices = df_unlb.head(K).index.tolist()
#         df_queried = df_unlb.loc[queried_indices]
#         # density = compute_density(Z_tr_unlabeled, k=10)
#         # score = std_p * density 
#         # topk_pos = np.argsort(-score)[:K]
#         # # queried_indices = df_tr_unlabeled.index[topk_pos].tolist()
#         # queried_indices = df_tr_unlabeled.iloc[topk_pos].index.tolist()

#         # df_queried = df_tr_unlabeled.loc[queried_indices]
        
#         return queried_indices, df_queried

#     # fit_kwargs['z_val'] = Z_val
#     # fit_kwargs['y_val'] = y_val
#     # 3. Train classifier on initial labeled pool
#     def reset_seeds(seed=42):
#         import random, numpy as np, tensorflow as tf
#         random.seed(seed)
#         np.random.seed(seed)
#         tf.random.set_seed(seed)

#     results = []
#     input_dim = Z_tr_labeled.shape[1]
    

#     if pool == "personal" or pool == "global":
#         active_model = None
#         # reset_seeds(seed=42)
#         clf, es  = clf_builder(input_dim, CLF_PATIENCE, dropout_rate)

#         fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}

#         if 'shuffle' not in fit_kwargs_with_callbacks:
#             fit_kwargs_with_callbacks['shuffle'] = False
#         if 'callbacks' in fit_kwargs_with_callbacks:
#             callbacks = fit_kwargs_with_callbacks['callbacks']
#             if not isinstance(callbacks, list):
#                 callbacks = [callbacks]
#             callbacks = callbacks + [es]
#             fit_kwargs_with_callbacks['callbacks'] = callbacks
#         else:
#             fit_kwargs_with_callbacks['callbacks'] = [es]
        
#         # Compute class weights for consistency with fit_and_eval and run.py
#         unique_classes = np.unique(y_tr_labeled)
#         cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_tr_labeled)
#         class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
        
#         history = clf.fit(Z_tr_labeled, 
#                         y_tr_labeled,
#                         class_weight=class_weight,
#                         **fit_kwargs_with_callbacks)
#         active_model = clf
#         random.seed(42)
#         np.random.seed(42)
#         tf.random.set_seed(42)
    
#     elif pool == "global_supervised":
#     # copy artifacts (optional)
#         if gs_src_dir is not None:
#             for fpath in Path(gs_src_dir).glob("*"):
#                 if fpath.suffix == ".keras":
#                     shutil.copy2(fpath, models_d / fpath.name)
#                 elif fpath.suffix == ".png":
#                     shutil.copy2(fpath, results_d / fpath.name)

#         results = []
#         participants_count_per_round = {}

#         # ---- Round 0 (cold-start from pretrained-on-10% init weights) ----
#         # active_model, auc_m_pre, auc_s_pre, thr, hist = uq_utility.train_and_eval_v2(
#         #     df_tr_labeled,
#         #     Z_val, y_val,
#         #     Z_te, y_te,
#         #     model_builder=uq_utility.build_global_cnn_lstm,
#         #     init_weights=init_weights_trained,
#         # )
        
#         auc_m_pre, auc_s_pre, _  = bootstrap_auc(y_te, active_model.predict(Z_te, verbose=0).ravel())
#         auc_m_train_pre, auc_s_train_pre, _ = bootstrap_auc(y_tr_labeled, active_model.predict(Z_tr_labeled, verbose=0).ravel())
#         auc_m_val_pre, auc_s_val_pre, _ = bootstrap_auc(y_val, active_model.predict(Z_val, verbose=0).ravel())
#         log_loss_pre = log_loss(y_te, active_model.predict(Z_te, verbose=0).ravel())
#         auc_plain = roc_auc_score(y_te, active_model.predict(Z_te, verbose=0).ravel())
#         print(f"Initial model plain AUC: {auc_plain:.4f}")

#         # breakpoint()
#          #auc_m_pre, auc_s_pre

#     best_f1 = np.nan
#     print(f"Round 0: TestAUC={auc_m_pre:.3f}±{auc_s_pre:.3f} | BestF1={best_f1:.3f}")
#     # if pool == "personal" or pool == "global":
#     #     train_loss = history.history['loss']
#     #     validation_loss = history.history['val_loss']
    
#     metrics_0 = pd.DataFrame(index=[0])
#     metrics_0["round"] = 0
#     metrics_0["AUC_Mean_Train"] = auc_m_train_pre
#     metrics_0["AUC_STD_Train"] = auc_s_train_pre
#     metrics_0["AUC_Mean_Val"] = auc_m_val_pre
#     metrics_0["AUC_STD_Val"] = auc_s_val_pre
#     metrics_0["AUC_Mean"] = auc_m_pre
#     metrics_0["AUC_STD"] = auc_s_pre
#     results.append(metrics_0)

#     queried_all = []
#     queried_participants = {}
#     participants_count_per_round = {}
#     # 4. Active learning loop
#     for b in range(1, budget + 1):
#         if len(df_tr_unlabeled) == 0:
#             break

#         k_actual = min(K, len(df_tr_unlabeled))
#         print(f"AL round {b}/{budget}  (labeled={len(df_tr_labeled)}, unlabeled={len(df_tr_unlabeled)}, K={k_actual})")

#         # Encode current unlabeled pool
#         if pool != "global_supervised":
#             H_tr_unlabeled, S_tr_unlabeled = encode(df_tr_unlabeled, enc_hr, enc_st)
#             Z_tr_unlabeled = np.concatenate([H_tr_unlabeled, S_tr_unlabeled], axis=1).astype("float32")
#         else: 
            
#             Z_tr_unlabeled, y_tr_unlabeled = uq_utility._build_XY_from_windows(df_tr_unlabeled, "none")
#         # Acquisition: random or most‑uncertain K windows
#         if Aq == "random":
#             queried_indices, df_queried = pick_random(k_actual, df_tr_unlabeled)
#         elif Aq == "uncertainty":
#             queried_indices, df_queried = pick_most_uncertain(
#                 active_model, df_tr_unlabeled, Z_tr_unlabeled, k_actual, T, mc_predict
#             )
#         elif Aq == "typiclust":
#             queried_indices, df_queried = typiclust.pick_typiclust(
#             df_tr_labeled=df_tr_labeled,
#             df_tr_unlabeled=df_tr_unlabeled,
#             Z_L=Z_tr_labeled,
#             Z_U=Z_tr_unlabeled,
#             K=k_actual,
#             n_clusters=None,  # let heuristic pick
#             k_nn=10,
#         )
#         else:
#             raise ValueError(f"Unknown acquisition function: {Aq}. Must be 'random' or 'uncertainty'.")
#         queried_all.extend(queried_indices)
#         queried_participants_per_round = df_queried["user_id"].tolist()
#         queried_participants[b] = queried_participants_per_round
#         participants_count_per_round[b] = Counter(queried_participants_per_round)
    

#         # Move queried from UNLABELED → LABELED

#         # df_queried['state_val'] = df_queried['true_label']  # restore true labels

#         df_tr_labeled = pd.concat([df_tr_labeled, df_queried], axis=0)
#         df_tr_unlabeled = df_tr_unlabeled.drop(index=queried_indices)
#         # Re‑encode labeled set
#         if pool != "global_supervised":
#             H_tr_labeled, S_tr_labeled = encode(df_tr_labeled, enc_hr, enc_st)
#             Z_tr_labeled = np.concatenate([H_tr_labeled, S_tr_labeled], axis=1).astype("float32")
#             y_tr_labeled = df_tr_labeled["state_val"].values.astype("float32")
#         else: 
#             Z_tr_labeled, y_tr_labeled = uq_utility._build_XY_from_windows(df_tr_labeled, "none")

        

#         # Retrain / fine‑tune classifier on expanded labeled pool
#         if len(df_tr_labeled) == 0:
#             print("Skipping training: labeled pool is empty.")
#             break

#         if pool == "personal" or pool == "global":
#             auc_m_post, auc_s_post, best_f1_post,val_auc, clf = fit_and_eval(
#                 fit_kwargs, clf_builder, input_dim,
#                 Z_tr_labeled, y_tr_labeled,
#                 Z_val, y_val, Z_te, y_te,
#                 CLF_PATIENCE, dropout_rate,
#                 clf=clf,
#             )
#             active_model = clf
#         elif pool == "global_supervised":
            
            
#             X_lab, y_lab = uq_utility._build_XY_from_windows(df_tr_labeled, "none")
#             class_weight = None
#             classes = np.unique(y_lab)
#             if len(classes) == 2:
#                 cw_vals = compute_class_weight("balanced", classes=classes, y=y_lab)
#                 class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
#             # set low lr for warm-start fine-tune
#             try:
#                 active_model.optimizer.learning_rate.assign(1e-5)
#             except Exception:
#                 tf.keras.backend.set_value(active_model.optimizer.learning_rate, 1e-5)
#             hist = active_model.fit(
#             X_lab, y_lab,
#             validation_data=(Z_val, y_val),
#             epochs=5,
#             batch_size=32,
#             class_weight=class_weight,
#             callbacks=[
#                 tf.keras.callbacks.EarlyStopping(
#                     monitor="val_loss", patience=CLF_PATIENCE, restore_best_weights=True, verbose=1
#                 )
#             ],
#             verbose=0,
#             shuffle=True,
#     )
#             auc_m_post, auc_s_post, _ = bootstrap_auc(y_te, active_model.predict(Z_te, verbose=0).ravel())
#             auc_m_train_post, auc_s_train_post, _ = bootstrap_auc(y_tr_labeled, active_model.predict(Z_tr_labeled, verbose=0).ravel())
#             auc_m_val_post, auc_s_val_post, _ = bootstrap_auc(y_val, active_model.predict(Z_val, verbose=0).ravel())
#             log_loss_post = log_loss(y_te, active_model.predict(Z_te, verbose=0).ravel())
            
            
#             best_f1_post = np.nan
        
#         # metrics = df_boot_post.copy()
#         metrics = pd.DataFrame(index=[0]) 
#         metrics["round"] = b
#         metrics['log_loss'] = log_loss_post
#         metrics["AUC_Mean_Train"] = auc_m_train_post
#         metrics["AUC_STD_Train"] = auc_s_train_post
#         metrics["AUC_Mean_Val"] = auc_m_val_post
#         metrics["AUC_STD_Val"] = auc_s_val_post
#         metrics["AUC_Mean"] = auc_m_post
#         metrics["AUC_STD"] = auc_s_post
#         metrics["Best_F1"] = best_f1_post
#         # metrics["Val_AUC"] = val_auc
        
#         results.append(metrics) 
#             ## model on full train set to get upper bound AUC
#         ### auc for full train set
        
#         # ---- New:count df ----
#         df_counts_wide = (
#             pd.DataFrame.from_dict(
#                 participants_count_per_round,
#                 orient="index"
#             )
#             .fillna(0)
#             .astype(int)
#         )

#         df_counts_wide.index.name = "round"
#         df_counts_wide.columns.name = "user_id"

#         print(f"  → Round {b}: AUC={auc_m_post:.3f} ± {auc_s_post:.3f}")
#     al_progress = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
#     return al_progress, active_model, df_tr_labeled, df_tr_unlabeled, queried_all,df_counts_wide

def stratified_split_min(df, label_col, test_size, min_per_class=5, random_state=42):
    labeled_idx, unlabeled_idx = [], []
    for label, group in df.groupby(label_col):
        n_unlabeled = int(len(group) * test_size)
        n_unlabeled = min(len(group) - min_per_class, n_unlabeled)
        n_unlabeled = max(0, n_unlabeled)
        unlab_idx = group.sample(n=n_unlabeled, random_state=random_state).index
        unlabeled_idx.extend(unlab_idx)
    labeled_idx = df.index.difference(unlabeled_idx)
    df_unlabeled = df.loc[unlabeled_idx].copy()
    df_unlabeled["true_label"] = df_unlabeled["state_val"]   # backup original labels
    df_unlabeled['state_val'] = -1
    return df.loc[labeled_idx], df_unlabeled

def stratified_day_class_seed(df, unlabeled_frac,label_col="state_val", day_col="day",
                               random_state=42):
    """
    Build a representative initial labeled set by:
      1. Ensuring at least 1 sample per day.
      2. Ensuring class balance among remaining labeled samples.
      3. Producing EXACTLY the number of labeled/unlabeled samples dictated by `unlabeled_frac`.

    Returns:
        df_labeled, df_unlabeled
    """

    df = df.copy()
    N = len(df)
    rng = np.random.default_rng(random_state)

    # ------------------------------------------
    # 1. Determine exact number of labeled samples
    # ------------------------------------------
    n_labeled = int((1 - unlabeled_frac) * N)
    n_labeled = max(n_labeled, 1)  # avoid empty
    print(f"Total samples: {N}, Labeled: {n_labeled}, Unlabeled: {N - n_labeled}")
    # ------------------------------------------
    # 2. Sample 1 per day (guarantee day coverage)
    # ------------------------------------------
    per_day = df.groupby(day_col).apply(
        lambda g: g.sample(n=1, random_state=random_state)
    ).reset_index(drop=True)

    labeled_indices = set(per_day.index)

    # How many more samples we need after per-day picks
    remaining_needed = n_labeled - len(labeled_indices)
    if remaining_needed < 0:
        raise ValueError(
            "Not enough total samples to satisfy at least 1 per day + unlabeled fraction constraint."
        )

    # # ------------------------------------------
    # 3. From REMAINDER: enforce class balance
    # ------------------------------------------
    remaining_df = df.drop(labeled_indices)

    # target additional samples per class
    # (balanced: half positive, half negative)
    n_pos = remaining_needed // 2
    n_neg = remaining_needed - n_pos  # handle odd case

    # class subsets
    pos_pool = remaining_df[remaining_df[label_col] == 1]
    neg_pool = remaining_df[remaining_df[label_col] == 0]

    # if not enough in either pool, fall back automatically
    n_pos = min(n_pos, len(pos_pool))
    n_neg = min(n_neg, len(neg_pool))

    pos_samples = pos_pool.sample(n=n_pos, random_state=random_state)
    neg_samples = neg_pool.sample(n=n_neg, random_state=random_state)

    # Add balanced class picks
    extra_indices = set(pos_samples.index) | set(neg_samples.index)
    labeled_indices.update(extra_indices)

    # If still short (because not enough pos or neg), fill purely at random
    still_needed = n_labeled - len(labeled_indices)
    if still_needed > 0:
        filler_pool = remaining_df.drop(labeled_indices)
        filler_samples = filler_pool.sample(n=still_needed, random_state=random_state)
        labeled_indices.update(filler_samples.index)

    # ------------------------------------------
    # Final subsets
    #------------------------------------------
    labeled_indices = list(labeled_indices)
    df_labeled = df.loc[labeled_indices].copy()
    df_unlabeled = df.drop(labeled_indices).copy()

    # Sanity check: exact sizes
    assert len(df_labeled) == n_labeled, f"Labeled count mismatch ({len(df_labeled)} != {n_labeled})"
    assert len(df_unlabeled) == N - n_labeled

    return df_labeled, df_unlabeled


def seed_variance_stratified(df,unlabeled_frac, hr_col="hr_seq", st_col="st_seq", random_state=42, n_bins=10):
    """
    Build a representative initial labeled set directly from raw df by:
      1. Computing physiological variability from HR + ST.
      2. Stratifying df into variance bins.
      3. Sampling evenly from bins.
      4. Producing EXACT labeled/unlabeled sizes from unlabeled_frac.

    Returns:
        df_labeled, df_unlabeled
    """
    df = df.copy()
    
    # Compute variability features
    df["hr_std"] = df[hr_col].apply(np.std)
    df["st_std"] = df[st_col].apply(np.std)
    df["var_score"] = df["hr_std"] + df["st_std"]   # simple composite variability score

    # Number of labeled samples required
    N = len(df)
    n_labeled = int((1 - unlabeled_frac) * N)

    # Create variance bins
    df["var_bin"] = pd.qcut(df["var_score"], q=n_bins, labels=False, duplicates="drop")

    # First, sample 1 per variance bin
    per_bin = df.groupby("var_bin").apply(
        lambda g: g.sample(n=1, random_state=random_state)
    ).reset_index(drop=True)

    labeled_idx = set(per_bin.index)

    # Remaining needed
    remaining = n_labeled - len(labeled_idx)
    if remaining < 0:
        raise ValueError("Too many variance bins for small dataset; reduce n_bins.")

    # Randomly fill the rest
    remaining_pool = df.drop(labeled_idx)
    filler = remaining_pool.sample(n=remaining, random_state=random_state)
    labeled_idx.update(filler.index)

    df_labeled = df.loc[list(labeled_idx)].copy()
    df_unlabeled = df.drop(list(labeled_idx)).copy()
    # df_unlabeled["true_label"] = df_unlabeled["state_val"]   # backup original labels
    # df_unlabeled['state_val'] = -1
    
    return df_labeled, df_unlabeled

def plot_top_k_picks(top_k_indices, df_tr, out_dir: Path) -> None:
    """
    Plot HR and steps time series for the queried windows.
    Saves one PNG per queried index into out_dir.
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Use label-based indexing: top_k_indices are df_tr indices
    df_topk = df_tr.loc[top_k_indices]

    for idx, (_, row) in enumerate(df_topk.iterrows()):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Heart rate plot
        hr = row.get("hr_raw", row.get("hr_seq"))
        st = row.get("st_raw", row.get("st_seq"))

        ax[0].plot(range(len(hr)), hr, color="blue", label="Heart Rate")
        ax[0].set_xlabel("Time Steps")
        ax[0].set_ylabel("Heart Rate")
        ax[0].set_title(f"Queried {idx} - Heart Rate")
        ax[0].legend()

        # Steps plot
        ax[1].plot(range(len(st)), st, color="green", label="Steps")
        ax[1].set_xlabel("Time Steps")
        ax[1].set_ylabel("Steps")
        ax[1].set_title(f"Queried {idx} - Steps")
        ax[1].legend()

        plt.tight_layout()
        out_path = out_dir / f"queried_window_{idx}.png"
        plt.savefig(out_path)
        plt.close(fig)


# def mc_predict_last_dropout(model, x, T, dropout_layer_name="dropout_last"):
#     """
#     Runs MC dropout **only on the final dropout layer** of the classifier.
#     """

#     # Convert input to tensor
#     x = tf.convert_to_tensor(x, dtype=tf.float32)

#     preds = []

#     # Identify layers
#     dropout_layer = model.get_layer(dropout_layer_name)
#     output_layer  = model.get_layer("output_layer")

#     # Build a model that outputs the pre-dropout representation
#     rep_model = tf.keras.Model(
#         inputs=model.input,
#         outputs=dropout_layer.input
#     )

#     for t in range(T):
#         # step 1: deterministic forward pass to get representation
#         rep = rep_model(x, training=False)

#         # step 2: apply MC dropout ONLY to last layer
#         rep_dp = dropout_layer(rep, training=True)

#         # step 3: final sigmoid output layer
#         logits = output_layer(rep_dp)

#         preds.append(tf.reshape(logits, [-1]).numpy())

#     preds = np.stack(preds, axis=0)
#     mean_probs = preds.mean(axis=0)
#     std_probs  = preds.std(axis=0)

#     return mean_probs, std_probs

def mc_predict(model, x, T):
    """
    Runs MC dropout on the final dropout layer of the classifier.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    preds = []
    
    for t in range(T):
        # Run the ENTIRE model with training=True to enable dropout
        logits = model(x, training=True)
        preds.append(tf.reshape(logits, [-1]).numpy())
    
    preds = np.stack(preds, axis=0)
    mean_probs = preds.mean(axis=0)
    std_probs = preds.std(axis=0)
    entropy = - (mean_probs * np.log(mean_probs + 1e-8)
             + (1 - mean_probs) * np.log(1 - mean_probs + 1e-8))
    # Entropy per MC sample
    H_per_t = - (preds * np.log(preds + 1e-8)
            + (1 - preds) * np.log(1 - preds + 1e-8))  # shape (T, N)

    H_exp = H_per_t.mean(axis=0)  # expected entropy

    BALD = entropy - H_exp         # epistemic uncertainty estimate
    
    return mean_probs, std_probs, BALD


def clf_builder(input_dim, CLF_PATIENCE, dropout_rate):
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=l2(1e-3))(inputs)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=l2(1e-3))(x)

    # dropout only here → for MC uncertainty
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    ##New layer
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=l2(1e-3))(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(5e-4),       # lower LR improves stability + AUC
        loss="binary_crossentropy",
        # metrics=["accuracy"]
         metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    es = EarlyStopping(
        # monitor="val_loss",
        monitor="val_auc",
        patience=CLF_PATIENCE,
        restore_best_weights=True,
        verbose=0
    )

    return model, es


##Note the below is used for mc dropout experiments 
# def clf_builder(input_dim, CLF_PATIENCE, dropout_rate):

#     inputs = tf.keras.Input(shape=(input_dim,), name="input")

#     x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(inputs)

#     # x = layers.BatchNormalization()(x)
#     x = layers.Dropout(dropout_rate, name="dropout_1")(x)

#     x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
#     x = layers.Dropout(dropout_rate, name="dropout_2")(x)


#     x = layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
#     x = layers.Dropout(dropout_rate, name="dropout_last")(x)  # << last dropout only

#     outputs = layers.Dense(1, activation='sigmoid', name="output_layer")(x)

#     model = tf.keras.Model(inputs, outputs, name="classifier")
#     model.compile(
#         optimizer=Adam(5e-3),
#         loss='binary_crossentropy',
#         metrics=['accuracy'],
#         run_eagerly=True
#     )

#     es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE,
#                        restore_best_weights=True, verbose=0)

#     return model, es




def seed_representative_raw(df,unlabeled_frac, hr_col="hr_seq", st_col="st_seq",
                             method="kmeans",
                            random_state=42):
    """
    Representative initial labeled set using raw HR & ST summary features.

    """

    df = df.copy()
    N = len(df)
    rng = np.random.default_rng(random_state)

    # ---- 1. Exact labeled size ----
    n_labeled = int((1 - unlabeled_frac) * N)
    n_labeled = max(n_labeled, 1)

    # ---- 2. Raw summary features ----
    df["hr_mean"] = df[hr_col].apply(np.mean)
    df["hr_std"]  = df[hr_col].apply(np.std)
    df["st_mean"] = df[st_col].apply(np.mean)
    df["st_std"]  = df[st_col].apply(np.std)

    Z = df[["hr_mean", "hr_std", "st_mean", "st_std"]].values

    # Use POSITIONS internally (0..N-1), not df.index
    pos_indices = np.arange(N)
    labeled_pos = []

    # ---- 3. K-MEANS SEED ----
    if method == "kmeans":
        k = min(n_labeled, N)
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(Z)

        closest_pos, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, Z)
        labeled_pos = list(set(closest_pos))

    # ---- 4. K-CENTER SEED ----
    elif method == "kcenter":
        labeled_pos = []
        idx0 = rng.integers(0, N)
        labeled_pos.append(idx0)

        dist = pairwise_distances(Z, Z[[idx0]]).reshape(-1)

        for _ in range(1, n_labeled):
            next_pos = int(np.argmax(dist))
            labeled_pos.append(next_pos)
            new_dist = pairwise_distances(Z, Z[[next_pos]]).reshape(-1)
            dist = np.minimum(dist, new_dist)

        labeled_pos = list(set(labeled_pos))

    else:
        raise ValueError("method must be 'kmeans' or 'kcenter'")

    # ---- 5. If too few, fill randomly ----
    if len(labeled_pos) < n_labeled:
        remaining = n_labeled - len(labeled_pos)
        remaining_pool = list(set(pos_indices) - set(labeled_pos))
        filler = rng.choice(remaining_pool, size=remaining, replace=False)
        labeled_pos.extend(filler)

    # ---- 6. Convert positional idx → df.index labels ----
    labeled_idx = df.index[labeled_pos].tolist()

    # ---- 7. Build final splits ----
    df_labeled = df.loc[labeled_idx].copy()
    df_unlabeled = df.drop(labeled_idx).copy()
    # df_unlabeled["true_label"] = df_unlabeled["state_val"]   # backup original labels
    # df_unlabeled['state_val'] = -1

    # ---- 8. Final assertions ----
    assert len(df_labeled) == n_labeled, f"Expected n_labeled={n_labeled}, got {len(df_labeled)}"
    assert len(df_unlabeled) == N - n_labeled, f"Unlabeled mismatch ({len(df_unlabeled)} != {N-n_labeled})"

    return df_labeled, df_unlabeled




def split_labeled_unlabeled_balanced(
    df_tr: pd.DataFrame,
    participant_col: str = "user_id",
    labeled_frac: float = 0.2,
    min_per_participant: int = 2,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    n_total = len(df_tr)
    n_target = int(np.ceil(labeled_frac * n_total))

    groups = {pid: g.index.to_numpy() for pid, g in df_tr.groupby(participant_col)}
    pids = list(groups.keys())

    # Start by allocating the minimum to everyone (or all their samples if fewer)
    chosen = {pid: [] for pid in pids}
    for pid in pids:
        idxs = groups[pid]
        k = min(min_per_participant, len(idxs))
        if k > 0:
            pick = rng.choice(idxs, size=k, replace=False).tolist()
            chosen[pid].extend(pick)

    labeled_set = set(i for pid in pids for i in chosen[pid])
    remaining_budget = n_target - len(labeled_set)
    if remaining_budget <= 0:
        df_labeled = df_tr.loc[sorted(labeled_set)].copy()
        df_unlabeled = df_tr.drop(index=sorted(labeled_set)).copy()
        return df_labeled, df_unlabeled

    # Round-robin add 1 sample per participant until budget is filled
    pid_cycle = pids.copy()
    rng.shuffle(pid_cycle)

    while remaining_budget > 0:
        progressed = False
        for pid in pid_cycle:
            if remaining_budget <= 0:
                break
            idxs = groups[pid]
            available = list(set(idxs) - set(chosen[pid]))
            if len(available) == 0:
                continue
            pick = rng.choice(available, size=1, replace=False).item()
            chosen[pid].append(pick)
            labeled_set.add(pick)
            remaining_budget -= 1
            progressed = True
        if not progressed:
            # No more samples available anywhere
            break

    df_labeled = df_tr.loc[sorted(labeled_set)].copy()
    df_unlabeled = df_tr.drop(index=sorted(labeled_set)).copy()
    return df_labeled, df_unlabeled

def bootstrap_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sample_frac: float = 0.7,
    n_iters: int = 1000,
    rng_seed: int = 42,
    return_samples: bool = False,
) -> tuple:
    """
    Compute AUC with bootstrap confidence intervals.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred_proba : np.ndarray
        Predicted probabilities
    sample_frac : float, default=0.7
        Fraction of samples to use in each bootstrap iteration
    n_iters : int, default=1000
        Number of bootstrap iterations
    rng_seed : int, default=42
        Random seed for reproducibility
    return_samples : bool, default=False
        If True, return all bootstrap AUC samples
    
    Returns
    -------
    auc_mean : float
        Mean AUC across bootstrap samples
    auc_std : float
        Standard deviation of AUC across bootstrap samples
    auc_samples : np.ndarray (optional)
        All bootstrap AUC scores if return_samples=True
    
    Examples
    --------
    """
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    k = int(np.round(sample_frac * n))
    idx_all = np.arange(n)
    
    aucs = []
    
    for _ in range(n_iters):
        # Sample with replacement
        sidx = rng.choice(idx_all, size=k, replace=True)
        y_samp = y_true[sidx]
        p_samp = y_pred_proba[sidx]
        
        
        # AUC only valid with both classes present
        if len(np.unique(y_samp)) > 1:
            aucs.append(roc_auc_score(y_samp, p_samp))
    
    auc_samples = np.array(aucs)
    auc_mean = np.nanmean(auc_samples)
    auc_std = np.nanstd(auc_samples, ddof=1)
    
    if return_samples:
        return auc_mean, auc_std, auc_samples
    
    valid_frac = len(auc_samples) / n_iters
    return auc_mean, auc_std, valid_frac

def make_labeled_unlabeled_with_target_quota(
    df_all_tr: pd.DataFrame,
    target_uid: str,
    unlabeled_frac: float,
    *,
    target_labeled_frac: float = 0.2,
    target_reserve_frac: float = 0.2,
    seed: int = 42,
    stratify_col: str = "state_val",
    user_col: str = "user_id",
):
    n_total   = len(df_all_tr)
    n_labeled = max(4, int(round((1 - unlabeled_frac) * n_total)))
    
    df_target = df_all_tr[df_all_tr[user_col] == target_uid]
    df_rest   = df_all_tr[df_all_tr[user_col] != target_uid]

    # ── Step 1: guarantee 1 per class from target ─────────
    df_target_lab = (
        df_target
        .groupby(stratify_col, group_keys=False)
        .apply(lambda g: g.sample(n=1, random_state=seed))
    )

    # ── Step 2: fill remaining slots from non-target ──────
    n_rest    = n_labeled - len(df_target_lab)
    df_rest_lab = (
        df_rest
        .groupby(stratify_col, group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), max(1, int(round(n_rest * len(g) / len(df_rest))))),
            random_state=seed
        ))
        .sample(n=min(n_rest, len(df_rest)), random_state=seed)
    )

    # ── Step 3: combine and split ─────────────────────────
    df_labeled   = (
        pd.concat([df_target_lab, df_rest_lab])
        .sample(frac=1.0, random_state=seed)
    )
    df_unlabeled = df_all_tr.drop(df_labeled.index)

    return df_labeled, df_unlabeled



def compute_representations( df_tr_labeled, df_tr_unlabeled, df_val, df_te,
        enc_hr, enc_st, pool):
    """
    Computes representations for labeled, unlabeled, validation, and test datasets based on the pool type.
    """
    if pool in ["personal", "global"]:
        H_tr_labeled,  S_tr_labeled  = encode(df_tr_labeled, enc_hr, enc_st)
        H_tr_unlabeled, S_tr_unlabeled = encode(df_tr_unlabeled, enc_hr, enc_st)

        H_val, S_val = encode(df_val, enc_hr, enc_st)
        H_te,  S_te  = encode(df_te, enc_hr, enc_st)
                ##Encoded representations
        Z_tr_labeled   = np.concatenate([H_tr_labeled,  S_tr_labeled],  axis=1).astype('float32')
        y_tr_labeled   = df_tr_labeled['state_val'].values.astype('float32')
        Z_tr_unlabeled = np.concatenate([H_tr_unlabeled, S_tr_unlabeled],axis=1).astype('float32')
        y_tr_unlabeled = df_tr_unlabeled['state_val'].values.astype('float32')
        # Fixed validation and test sets (do NOT change across rounds)
        Z_val  = np.concatenate([H_val, S_val], axis=1).astype('float32')
        y_val  = df_val['state_val'].values.astype('float32')
        Z_te   = np.concatenate([H_te,  S_te],  axis=1).astype('float32')
        y_te   = df_te['state_val'].values.astype('float32')
        
        return Z_tr_labeled, y_tr_labeled, Z_tr_unlabeled, y_tr_unlabeled, Z_val, y_val, Z_te, y_te
    
    elif pool == "global_supervised":
        Z_tr_labeled, y_tr_labeled = uq_utility._build_XY_from_windows(df_tr_labeled)
        Z_tr_unlabeled, y_tr_unlabeled = uq_utility._build_XY_from_windows(df_tr_unlabeled)

        Z_val, y_val = uq_utility._build_XY_from_windows(df_val)
        Z_te, y_te = uq_utility._build_XY_from_windows(df_te)
    
        return Z_tr_labeled, y_tr_labeled, Z_tr_unlabeled, y_tr_unlabeled, Z_val, y_val, Z_te, y_te
    
    else:
        raise ValueError(f"Unknown pool type: {pool}")



def compute_density(Z, k=7):
    # pairwise distance matrix
    dist = pairwise_distances(Z, Z, metric='euclidean')

    # sort distances and take mean distance to the kNN
    knn_distances = np.sort(dist, axis=1)[:, 1:k+1]
    density = 1.0 / (1e-6 + np.mean(knn_distances, axis=1))
    return density

def pick_random(K, df_tr_unlabeled):
    """Return K random indices from the unlabeled pool."""
    if len(df_tr_unlabeled) < K:
        K = len(df_tr_unlabeled)
    queried_indices = random.sample(list(df_tr_unlabeled.index), K)
    df_queried = df_tr_unlabeled.loc[queried_indices]
    return queried_indices, df_queried

def pick_most_uncertain(
    clf, df_tr_unlabeled, Z_tr_unlabeled, K, T,mc_predict
):
    """MC‑Dropout acquisition on encoded unlabeled set."""
    # ---------------------------
    # Freeze BatchNorm for MC dropout
    # ---------------------------
    for layer in clf.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # ---------------------------
    # Step 1: MC-Dropout uncertainty
    # ---------------------------
    _, std_p, BALD = mc_predict(clf, Z_tr_unlabeled, T=T)

    if len(std_p) != len(df_tr_unlabeled):
        raise ValueError(
            f"std_p has {len(std_p)} entries but df_tr_unlabeled has {len(df_tr_unlabeled)} rows."
        )
    # Attach uncertainty to rows (keeping their original indices!)
    df_unlb = df_tr_unlabeled.copy()
    df_unlb["uncertainty"] = std_p
    # df_unlb["uncertainty"] = BALD


    # Sort high → low uncertainty
    df_unlb = df_unlb.sort_values("uncertainty", ascending=False)

    # ---------------------------
    # Step 2: Select top‑K uncertain windows
    # ---------------------------
    queried_indices = df_unlb.head(K).index.tolist()
    df_queried = df_unlb.loc[queried_indices]
    # density = compute_density(Z_tr_unlabeled, k=10)
    # score = std_p * density 
    # topk_pos = np.argsort(-score)[:K]
    # # queried_indices = df_tr_unlabeled.index[topk_pos].tolist()
    # queried_indices = df_tr_unlabeled.iloc[topk_pos].index.tolist()

    # df_queried = df_tr_unlabeled.loc[queried_indices]
    
    return queried_indices, df_queried


def reset_seeds(seed=42):
    import random, numpy as np, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def encode_single_df(df, enc_hr, enc_st, pool):
    """
    Encode a dataframe using the provided encoders based on the pool type.
    """
    if pool in ["personal", "global"]:
        H, S = encode(df, enc_hr, enc_st)
        Z = np.concatenate([H, S], axis=1).astype('float32')
    elif pool == "global_supervised":
        Z, _ = uq_utility._build_XY_from_windows(df)
    else:
        raise ValueError(f"Unknown pool type: {pool}")
    return Z

# ============================================================================
# MAIN ACTIVE LEARNING LOOP
# ============================================================================

def run_al(Aq, df_tr_labeled, df_tr_unlabeled, df_val, df_te, 
          enc_hr, enc_st, clf_builder, mc_predict, K, budget, T, 
          CLF_PATIENCE, dropout_rate, fit_kwargs, pool,
          models_d, results_d, active_model=None, 
          init_weights_trained=None, warm_start: bool = True, seed: int = 42):
    """
    Active Learning Loop - Main Entry Point
    """
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    # Reset RNGs for deterministic initialization/training
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    Z_tr_labeled, y_tr_labeled, Z_tr_unlabeled, y_tr_unlabeled, Z_val, y_val, Z_te, y_te = compute_representations(
        df_tr_labeled, df_tr_unlabeled, df_val, df_te,
        enc_hr, enc_st, pool
    )

    # Z_tr_labeled = (encode_single_df(df_tr_labeled, enc_hr, enc_st, pool)).astype('float32')
    y_tr_labeled = df_tr_labeled['state_val'].values.astype('float32')
    
    results = []
    queried_all = []
    queried_participants = {}
    participants_count_per_round = {}

    if pool in ["personal", "global"]:
        active_model = None 
                # reset_seeds(seed=42)
        reset_seeds(seed)
        input_dim = Z_tr_labeled.shape[1]
        clf, es  = clf_builder(input_dim, CLF_PATIENCE, dropout_rate)

        fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}

        if 'shuffle' not in fit_kwargs_with_callbacks:
            fit_kwargs_with_callbacks['shuffle'] = False
        if 'callbacks' in fit_kwargs_with_callbacks:
            callbacks = fit_kwargs_with_callbacks['callbacks']
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            callbacks = callbacks + [es]
            fit_kwargs_with_callbacks['callbacks'] = callbacks
        else:
            fit_kwargs_with_callbacks['callbacks'] = [es]
        
        # Compute class weights for consistency with fit_and_eval and run.py
        unique_classes = np.unique(y_tr_labeled)
        cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_tr_labeled)
        class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}

        
        history = clf.fit(Z_tr_labeled, 
                        y_tr_labeled, 
                        class_weight=class_weight, **fit_kwargs_with_callbacks)
        active_model = clf
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
    elif pool == "global_supervised":
            # copy artifacts (optional)
        # if gs_src_dir is not None:
        #     for fpath in Path(gs_src_dir).glob("*"):
        #         if fpath.suffix == ".keras":
        #             shutil.copy2(fpath, models_d / fpath.name)
        #         elif fpath.suffix == ".png":
        #             shutil.copy2(fpath, results_d / fpath.name)
        pass 
    if pool == "global_supervised":
        if warm_start:
            if active_model is None:
                raise RuntimeError("warm_start=True requires an initial active_model.")
        else:
            _, _, active_model = uq_utility.train_base_init_on_labeled(
                df_tr_labeled, Z_val, y_val, uq_utility.build_global_cnn_lstm, verbose=0
            )

    auc_m_pre, auc_s_pre, _ = bootstrap_auc(y_te, active_model.predict(Z_te, verbose=0).ravel())
    auc_m_train_pre, auc_s_train_pre, _ = bootstrap_auc(y_tr_labeled, active_model.predict(Z_tr_labeled, verbose=0).ravel())
    auc_m_val_pre, auc_s_val_pre, _ = bootstrap_auc(y_val, active_model.predict(Z_val, verbose=0).ravel())
    
    metrics_0 = pd.DataFrame(index=[0])
    metrics_0["round"] = 0
    metrics_0["AUC_Mean_Train"] = auc_m_train_pre
    metrics_0["AUC_STD_Train"] = auc_s_train_pre
    metrics_0["AUC_Mean_Val"] = auc_m_val_pre
    metrics_0["AUC_STD_Val"] = auc_s_val_pre
    metrics_0["AUC_Mean"] = auc_m_pre
    metrics_0["AUC_STD"] = auc_s_pre
    results.append(metrics_0)

    
    # ========================================================================
    # ACTIVE LEARNING LOOP
    # ========================================================================
    
    for round_num in range(1, budget + 1):
        print(f"\n{'='*70}")
        print(f"AL Round {round_num}/{budget}")
        print(f"  Labeled: {len(df_tr_labeled)}")
        print(f"  Unlabeled: {len(df_tr_unlabeled)}")
        print(f"{'='*70}")
        
        # Check if unlabeled pool is exhausted
        if len(df_tr_unlabeled) == 0:
            print("Unlabeled pool exhausted. Stopping AL loop.")
            break
        breakpoint
        k_actual = min(K, len(df_tr_unlabeled))
        Z_tr_unlabeled = encode_single_df(df_tr_unlabeled, enc_hr, enc_st, pool)
        
        if Aq == "random":
            queried_indices, df_queried = pick_random(K, df_tr_unlabeled)
        elif Aq == "uncertainty":
            queried_indices, df_queried = pick_most_uncertain(
                active_model, df_tr_unlabeled, Z_tr_unlabeled, K, T, mc_predict
            )
        else:
            raise ValueError(f"Unknown acquisition function: {Aq}. Must be 'random' or 'uncertainty'.")
    
        queried_all.extend(queried_indices)
        queried_participants_per_round = df_queried["user_id"].tolist()
        queried_participants[round_num] = queried_participants_per_round
        participants_count_per_round[round_num] = Counter(queried_participants_per_round)
        
        #### Move queried from UNLABELED → LABELED
        df_tr_labeled = pd.concat([df_tr_labeled, df_queried], axis=0)
        df_tr_unlabeled = df_tr_unlabeled.drop(index=queried_indices)
        print(f"AL round {round_num}/{budget}  (labeled={len(df_tr_labeled)}, unlabeled={len(df_tr_unlabeled)}, K={k_actual})")

        # Enforce deterministic order before encoding/training
        df_tr_labeled = df_tr_labeled.sort_index()

        # Re-encode labeled pool
        Z_tr_labeled = encode_single_df(df_tr_labeled, enc_hr, enc_st, pool)
        y_tr_labeled = df_tr_labeled['state_val'].values.astype('float32')

        ###Retrain or fine-tune model on the expanded labeled set
        
        if pool in ["personal", "global"]:
            if not warm_start:
                # Reset RNGs before rebuilding the model for deterministic training
                reset_seeds(seed)
                clf = None


            auc_m_post, auc_s_post, best_f1_post,auc_m_val_post, auc_s_val_post, auc_m_train_post, auc_s_train_post, clf = fit_and_eval(
                fit_kwargs, clf_builder, input_dim,
                Z_tr_labeled, y_tr_labeled,
                Z_val, y_val, Z_te, y_te,
                CLF_PATIENCE, dropout_rate,
                clf=clf,
            )
            active_model = clf
            log_loss_post = np.nan
        elif pool == "global_supervised":
            if not warm_start:
                _, _, active_model = uq_utility.train_base_init_on_labeled(
                    df_tr_labeled, Z_val, y_val, uq_utility.build_global_cnn_lstm, verbose=0
                )
            else:
                X_lab, y_lab = uq_utility._build_XY_from_windows(df_tr_labeled)
                class_weight = None
                classes = np.unique(y_lab)
                if len(classes) == 2:
                    cw_vals = compute_class_weight("balanced", classes=classes, y=y_lab)
                    class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
                # Reset optimizer state (use higher LR on final round)
                lr = 1e-3 if round_num == budget else 1e-5
                active_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss="binary_crossentropy",
                    metrics=["accuracy"],
                )
                hist = active_model.fit(
                    X_lab, y_lab,
                    validation_data=(Z_val, y_val),
                    epochs=50,
                    batch_size=32,
                    class_weight=class_weight,
                    # callbacks=[
                    #     tf.keras.callbacks.EarlyStopping(
                    #         monitor="val_loss", patience=CLF_PATIENCE, restore_best_weights=True, verbose=1
                    #     )
                    # ],
                    verbose=0,
                    shuffle=True,
                )
            auc_m_post, auc_s_post, _ = bootstrap_auc(y_te, active_model.predict(Z_te, verbose=0).ravel())
            auc_m_train_post, auc_s_train_post, _ = bootstrap_auc(y_tr_labeled, active_model.predict(Z_tr_labeled, verbose=0).ravel())
            auc_m_val_post, auc_s_val_post, _ = bootstrap_auc(y_val, active_model.predict(Z_val, verbose=0).ravel())
            log_loss_post = log_loss(y_te, active_model.predict(Z_te, verbose=0).ravel())
            
            
            best_f1_post = np.nan
        
        # metrics = df_boot_post.copy()
        metrics = pd.DataFrame(index=[0]) 
        metrics["round"] = round_num
        metrics['log_loss'] = log_loss_post
        metrics["AUC_Mean_Train"] = auc_m_train_post
        metrics["AUC_STD_Train"] = auc_s_train_post
        metrics["AUC_Mean_Val"] = auc_m_val_post
        metrics["AUC_STD_Val"] = auc_s_val_post
        metrics["AUC_Mean"] = auc_m_post
        metrics["AUC_STD"] = auc_s_post
        metrics["Best_F1"] = best_f1_post
        # metrics["Val_AUC"] = val_auc
        
        results.append(metrics) 
            ## model on full train set to get upper bound AUC
        ### auc for full train set
        
        # ---- New:count df ----
        df_counts_wide = (
            pd.DataFrame.from_dict(
                participants_count_per_round,
                orient="index"
            )
            .fillna(0)
            .astype(int)
        )

        df_counts_wide.index.name = "round"
        df_counts_wide.columns.name = "user_id"

        print(f"  → Round {round_num}: AUC={auc_m_post:.3f} ± {auc_s_post:.3f}")
    al_progress = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return al_progress, active_model, df_tr_labeled, df_tr_unlabeled, queried_all,df_counts_wide
