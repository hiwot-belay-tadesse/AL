import random
import os
import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import utility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ensures single-threaded deterministic ops
os.environ["TF_DISABLE_MPS"] = "1"         # <- disable MPS so TensorFlow runs on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""    # ensure no GPU usage at all
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import sys
from src import compare_pipelines
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.chart_utils import (
    bootstrap_threshold_metrics, plot_thresholds, plot_ssl_losses
)
from src.compare_pipelines import select_threshold_train
from tensorflow.keras import Model
from src.signal_utils import (
    WINDOW_SIZE, STEP_SIZE, create_windows,
    build_simclr_encoder, create_projection_head, train_simclr
)
BATCH_SSL, SSL_EPOCHS                   = 32, 100

GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 15
WINDOW_LEN                              = WINDOW_SIZE




def build_all_splits_labeled(
    df_tr_labeled: pd.DataFrame,
    all_splits: dict,
    *,
    user_col: str = "user_id",
    time_col: str = "date",
):
    """
    Construct a labeled-version of all_splits based on the currently labeled
    training windows.

    Args:
        df_tr_labeled : DataFrame containing labeled TRAIN windows only.
                        Must include user_id and timestamp columns.
        all_splits    : original all_splits dict
                        { uid: (train_days, val_days, test_days) }
        user_col      : column name identifying user
        time_col      : timestamp column used to derive days

    Returns:
        all_splits_labeled : dict
            { uid: (labeled_train_days, val_days, test_days) }
    """
    all_splits_labeled = {}

    # ensure date column exists
    if time_col not in df_tr_labeled.columns:
        raise ValueError(f"{time_col} not found in df_tr_labeled")

    # extract day-level labels per user
    df_tr_labeled = df_tr_labeled.copy()
    df_tr_labeled["day"] = pd.to_datetime(df_tr_labeled[time_col]).dt.date

    for uid, (tr_days, val_days, te_days) in all_splits.items():
        # select labeled rows for this user
        df_u = df_tr_labeled[df_tr_labeled[user_col] == uid]

        if len(df_u) == 0:
            labeled_tr_days = np.array([], dtype=object)
        else:
            labeled_tr_days = np.array(sorted(df_u["day"].unique()))

        all_splits_labeled[uid] = (
            labeled_tr_days,
            val_days,   # unchanged
            te_days     # unchanged
        )

    return all_splits_labeled

def mc_predict(model, x, T):

    '''
    Quantifies uncertainity using Monte Carlo Dropout, 
    where T stochastic forward passes are performed with dropout enabled.
    Epistemic uncertainty is estimated as the standard deviation of the T predictions.
     
    Args: 
        model: tf.keras.Model with Dropout layers
        x:     Input data, shape (n_samples, n_features, unlabled
        T:     Number of stochastic forward passes
    '''
    # Reset seeds for reproducibility (MC Dropout is stochastic)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    preds = []

# Convert to Tensor to avoid shape issues
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

    for i in range(T):
        # Reset seed before each forward pass for reproducibility
        tf.random.set_seed(SEED + i)  # Different seed per pass but deterministic
        # Dropout ON, BatchNorm frozen
        p = model(x_tensor, training=True)

        # Ensure (N,) shape
        p = tf.reshape(p, [-1])
        preds.append(p.numpy())

    preds = np.stack(preds, axis=0)  # (T, N)
    mean_probs = preds.mean(axis=0)  # (N,)
    std_probs  = preds.std(axis=0)   # (N,)
    return mean_probs, std_probs


def cluster_labeled_signals():
    '''
    Clusters labeled signals using KMeans
    Args:
        
    
    Returns:    
    
    '''

    pass

def clf_builder(input_dim, CLF_PATIENCE, dropout_rate):
    '''
    builds and compiles a feedforward neural network classifier with dropout layers.
    Args:
        input_dim:   Number of input features(X_train.shape[1])
        CLF_PATIENCE: Early stopping patience
    Returns:
        model: Compiled tf.keras.Model
        es:    EarlyStopping callback 
    '''
    model = Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(), layers.Dropout(dropout_rate, name="dropout_1"),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(dropout_rate, name="dropout_2"),
        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(dropout_rate, name="dropout_3"),
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    model.compile(optimizer=Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'], run_eagerly=True)
    es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE, restore_best_weights=True, verbose=0)
    return model, es

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
    clf, es = clf_builder(input_dim, CLF_PATIENCE, dropout_rate)
    # if clf is None:
    #     clf, es = clf_builder(input_dim, CLF_PATIENCE, dropout_rate)
    # else:
    #     es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE, restore_best_weights=True, verbose=0)
    
    # NEW: compute balanced class weights
    # cw_vals = compute_class_weight('balanced', classes=np.unique(y_lab), y=y_lab)
    # class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    # y_lab = y_lab.astype(int)
    unique_classes = np.unique(y_lab)
    cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_lab)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    
    
    # Train the model
    fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}
    # Remove any non-fit() arguments that might have been accidentally included
    fit_kwargs_with_callbacks.pop('CLF_PATIENCE', None)
    fit_kwargs_with_callbacks.pop('dropout_rate', None)
    # Ensure shuffle is controlled for reproducibility (default to False if not specified)
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
    
    clf.fit(X_lab, y_lab, class_weight=class_weight, **fit_kwargs_with_callbacks)
    
    # Threshold scan on the FULL training set 
    best_thr = select_threshold_train(clf, X_val, y_val)
    # breakpoint()
    # best_thr = select_threshold_train(clf, X_lab, y_lab)

  # best_thr = 0.35
    # 9) Bootstrap & plot on TEST, dropout is turned off here(.predict())
    probs_te = clf.predict(X_test, verbose=0).ravel()
 
    df_boot, auc_m, auc_s = bootstrap_threshold_metrics(
        y_test,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )

    return df_boot, auc_m, auc_s, clf


def stratified_split_min(df, label_col, test_size, min_per_class=5, random_state=42):
    labeled_idx, unlabeled_idx = [], []
    for label, group in df.groupby(label_col):
        n_unlabeled = int(len(group) * test_size)
        n_unlabeled = min(len(group) - min_per_class, n_unlabeled)
        n_unlabeled = max(0, n_unlabeled)
        unlab_idx = group.sample(n=n_unlabeled, random_state=random_state).index
        unlabeled_idx.extend(unlab_idx)
    labeled_idx = df.index.difference(unlabeled_idx)
    return df.loc[labeled_idx], df.loc[unlabeled_idx]

def encode(df, enc_hr, enc_st):
    hr_seq = np.stack(df["hr_seq"])[..., None]
    st_seq = np.stack(df["st_seq"])[..., None]
    H = enc_hr.predict(hr_seq, verbose=0)
    S = enc_st.predict(st_seq, verbose=0)
    return np.concatenate([H, S], axis=1)

def preprocess_df(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    df_te: pd.DataFrame,
    enc_hr: keras.Model,
    enc_st: keras.Model,
    unlabeled_frac: float,
    random_state: 42,
):
    """
    Prepare train/val/test windows, split train into labeled/unlabeled (mask unlabeled with -1),
    encode with SSL encoders, and return everything needed for downstream training/AL.

    Returns:
      {
        "df": {
          "train": df_tr,
          "train_labeled": df_tr_labeled,
          "train_unlabeled": df_tr_unlabeled,   # state_val == -1
          "val": df_val,
          "test": df_te,
        },
        "Xy": {
          "X_train_full": X_train_full, "y_train_full": y_train_full,   # full train (GT labels)
          "X_train_l": X_train_l,       "y_train_l": y_train_l,         # labeled only
          "X_train_u": X_train_u,       "y_train_u": y_train_u,         # unlabeled (all -1)
          "X_val": X_val,               "y_val": y_val,
          "X_test": X_test,             "y_test": y_test,
          "X_all": X_all_df,            "y_all": y_all_ser,             # full train as pd objects (idx-aligned)
        },
        "idx": {
          "labeled": idx_lab,
          "unlabeled": idx_unlb,
        },
        "encoders": {"hr": enc_hr, "steps": enc_st},
      }
    """


    # Split train into labeled/unlabeled; mask unlabeled labels with -1
    df_tr_labeled, df_tr_unlabeled = train_test_split(
        df_tr,
        test_size=unlabeled_frac,
        random_state=random_state,
        stratify=df_tr["state_val"]
    )
    # breakpoint()
    # # 1. Sort days chronologically
    # unique_days = np.sort(df_tr['day'].unique())

    # # 2. Determine split index
    # split_idx = int(len(unique_days) * unlabeled_frac)
    # # 3. Split into unlabeled (first X%) and labeled (remaining)
    # days_unlabeled = unique_days[:split_idx]
    # days_labeled   = unique_days[split_idx:]

    # # 4. Subset the main dataframe
    # df_tr_unlabeled = df_tr[df_tr['day'].isin(days_unlabeled)]
    # df_tr_labeled   = df_tr[df_tr['day'].isin(days_labeled)]

    df_tr_unlabeled = df_tr_unlabeled.copy()
    df_tr_unlabeled["state_val"] = -1

    # --- 4) Encode sequences -> representations ---
    def encode(df):
        hr_seq = np.stack(df["hr_seq"])[..., None]
        st_seq = np.stack(df["st_seq"])[..., None]
        H = enc_hr.predict(hr_seq, verbose=0)
        S = enc_st.predict(st_seq, verbose=0)
        return np.concatenate([H, S], axis=1)

    X_train_full = encode(df_tr)             # full train (GT labels)
    y_train_full = df_tr["state_val"].values

    X_train_l = encode(df_tr_labeled)        # labeled subset
    y_train_l = df_tr_labeled["state_val"].values

    X_train_u = encode(df_tr_unlabeled)      # unlabeled subset (labels masked)
    y_train_u = df_tr_unlabeled["state_val"].values  # all -1

    X_val = encode(df_val)
    y_val = df_val["state_val"].values

    X_test = encode(df_te)
    y_test = df_te["state_val"].values

    # Optional: full-train as pd objects with aligned index (handy for AL querying)
    X_all_df = pd.DataFrame(X_train_full, index=df_tr.index)
    y_all_ser = df_tr["state_val"]

    # Seed pools for AL
    idx_lab  = df_tr_labeled.index.copy()
    idx_unlb = df_tr_unlabeled.index.copy()

    return {
        "df": {
            "train_labeled": df_tr_labeled,
            "train_unlabeled": df_tr_unlabeled,

        },
        "Xy": {
            "X_train_full": X_train_full, "y_train_full": y_train_full,
            "X_train_l": X_train_l,       "y_train_l": y_train_l,
            "X_train_u": X_train_u,       "y_train_u": y_train_u,
            "X_val": X_val,               "y_val": y_val,
            "X_test": X_test,             "y_test": y_test,
            "X_all": X_all_df,            "y_all": y_all_ser,
        },
        "idx": {
            "labeled": idx_lab,
            "unlabeled": idx_unlb,
        },
    }
    


from pathlib import Path
from src.classifier_utils import (
    BASE_DATA_DIR, ALLOWED_SCENARIOS,
    load_signal_data, load_label_data, process_label_window
)

from src.signal_utils import (
    WINDOW_SIZE, STEP_SIZE, create_windows,
    build_simclr_encoder, create_projection_head, train_simclr
)
from src.compare_pipelines import  collect_windows




def finetune_encoders(
    enc_hr,
    enc_st,
    df_queried,
    epochs=100,
    batch_size=32,
):
    """
    Fine-tune SSL encoders on newly queried windows.
    """

    # Extract window sequences
    hr_windows = np.stack(df_queried["hr_seq"]).astype("float32")[..., None]
    st_windows = np.stack(df_queried["st_seq"]).astype("float32")[..., None]

    # Projection heads (can be fresh each time)
    head_hr = create_projection_head()
    head_st = create_projection_head()

    # Enable training
    enc_hr.trainable = True
    enc_st.trainable = True

    # Fine-tune with SSL (no labels needed)
    train_simclr(
        enc_hr, head_hr,
        hr_windows, hr_windows,
        batch_size=batch_size,
        epochs=epochs,
    )
    train_simclr(
        enc_st, head_st,
        st_windows, st_windows,
        batch_size=batch_size,
        epochs=epochs,
    )

    # Freeze again (important for AL stability)
    enc_hr.trainable = False
    enc_st.trainable = False

    return enc_hr, enc_st

# def finetune_encoders(
#     enc_hr,
#     enc_st,
#     df_new,              # newly queried batch (or newly arrived raw data)
#     df_replay=None,      # replay buffer: random sample from df_tr_unlabeled or df_all_tr
#     *,
#     epochs=3,            # << small
#     batch_size=32,
#     replay_frac=0.5,
#     rng_seed=42,
# ):
#     """
#     Fine-tune encoders with SimCLR on new windows + replay windows.
#     """
#     rng = np.random.default_rng(rng_seed)

#     def _mix_windows(col):
#         new_w = np.stack(df_new[col]).astype("float32")[..., None]

#         if df_replay is None or len(df_replay) == 0:
#             w = new_w
#         else:
#             rep_w = np.stack(df_replay[col]).astype("float32")[..., None]

#             # sample replay to match desired fraction
#             n_new = len(new_w)
#             n_rep = int((replay_frac / (1 - replay_frac)) * n_new)
#             n_rep = min(n_rep, len(rep_w))

#             rep_idx = rng.choice(len(rep_w), size=n_rep, replace=False)
#             w = np.concatenate([new_w, rep_w[rep_idx]], axis=0)

#         # shuffle
#         idx = rng.permutation(len(w))
#         return w[idx]

#     hr_windows = _mix_windows("hr_seq")
#     st_windows = _mix_windows("st_seq")

#     def _train_one(enc, windows):
#         # split train/val
#         n = len(windows)
#         n_tr = max(1, int(0.8 * n))
#         tr, va = windows[:n_tr], windows[n_tr:] if n - n_tr >= 2 else (windows, windows)

#         # IMPORTANT: reuse a head if you can; if not, ok but keep epochs small
#         head = create_projection_head()

#         enc.trainable = True
#         train_simclr(enc, head, tr, va, batch_size=batch_size, epochs=epochs)
#         enc.trainable = False

#     _train_one(enc_hr, hr_windows)
#     _train_one(enc_st, st_windows)

#     return enc_hr, enc_st

def _ensure_global_encoders(
    shared_root,
    fruit,
    scenario,
    all_splits,
    BATCH_SSL,
    SSL_EPOCHS,
    exclude_user_id=None,
    BP_MODE=False,
):
    """
    Train/load global SSL encoders for a fruit+scenario setup.

    If exclude_user_id is provided, that user's train-day windows are excluded
    from encoder training to avoid target-user leakage.
    """
    suffix = ""
    if exclude_user_id is not None:
        safe_uid = str(exclude_user_id).replace("/", "_")
        suffix = f"__exclude_{safe_uid}"
    sdir = Path(shared_root) / f"{fruit}_{scenario}{suffix}"
    sdir.mkdir(parents=True, exist_ok=True)
    paths = {
        'hr':    sdir / 'hr_encoder.keras',
        'steps': sdir / 'steps_encoder.keras'
    }
    if  all(p.exists() for p in paths.values()):
        hr = load_model(paths['hr'])
        hr.trainable = False
        st = load_model(paths['steps'])
        st.trainable = False
        return hr, st, sdir
    

    losses = {}
    for dtype in ['hr', 'steps']:
        bank = []
        if BP_MODE:
            user_iter = list(all_splits.keys())
        else:
            user_iter = [
                u for u, pairs in ALLOWED_SCENARIOS.items()
                if (fruit, scenario) in pairs
            ]



        for u in user_iter:
            if exclude_user_id is not None:
                u_str = str(u)
                ex_str = str(exclude_user_id)
                # Match exact id, and also ID-prefixed variants (e.g., "15" vs "ID15")
                if (
                    u_str == ex_str
                    or u_str.lstrip("IDid") == ex_str.lstrip("IDid")
                ):
                    continue

            # now unpack train/val/test days
            tr_days_u, _val_days_u, _te_days_u = all_splits.get(u, ([], [], []))

            if len(tr_days_u) == 0:
                continue

            if BP_MODE:
                hr_df, st_df, _, _ = compare_pipelines._bp_load_all(u)
            else:
                hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            df = hr_df if dtype == 'hr' else st_df

            # only use train days here
            mask = np.isin(df.index.date, tr_days_u)
            vals = StandardScaler()\
                   .fit_transform(df.loc[mask, 'value'].values.reshape(-1, 1))

            if len(vals) < WINDOW_SIZE:
                continue

            bank.append(create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32'))

        if not bank:
            raise RuntimeError(f"No train-day segments for global {dtype} SSL!")

        segs = np.concatenate(bank, axis=0)
        n, idx = len(segs), np.random.permutation(len(segs))
        tr, va = segs[idx[: int(0.8 * n)]], segs[idx[int(0.8 * n) :]]

        enc = build_simclr_encoder(WINDOW_SIZE)
        head = create_projection_head()
        tr_l, va_l = train_simclr(enc, head, tr, va,
                                  batch_size=BATCH_SSL, epochs=SSL_EPOCHS)

        enc.save(paths[dtype])
        enc.trainable = False
        losses[dtype] = (tr_l, va_l)

    # plot losses for both modalities
    plot_ssl_losses(*losses['hr'],    sdir, encoder_name="global_hr")
    plot_ssl_losses(*losses['steps'], sdir, encoder_name="global_steps")

    hr = load_model(paths['hr']);    hr.trainable = False
    st = load_model(paths['steps']); st.trainable = False
    
    return hr, st, sdir


def preprocess_signal_to_df(uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    tr_days_u: np.ndarray,
    val_days_u: np.ndarray,
    te_days_u: np.ndarray,
    neg_df_u: pd.DataFrame,
    pool: str, 
    ):
    
    '''
    given raw signals and labels for a user, it returns preprocessed dataframes for train, val and test along with the trained encoders
    '''
    
        # --- Directories (kept minimal; no writes here) ---
    out_dir   = user_root / pool
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # --- 1) Load raw signals & labels ---
    hr_df, st_df  = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df        = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df        = neg_df_u

    # --- 2) Train or load SSL encoders (per-user encoders) ---
    enc_hr = _train_or_load_encoder(models_d / 'hr_encoder.keras', 'hr',    hr_df, tr_days_u, results_d)
    enc_st = _train_or_load_encoder(models_d / 'steps_encoder.keras', 'steps', st_df, tr_days_u, results_d)

    # --- 3) Build windows for TRAIN / VAL / TEST ---

    df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
    # sample 10% of training data for AL
    # df_tr  = df_tr.sample(frac=0.1, random_state=42).reset_index(drop=True)
    df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days_u)
    df_te  = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)

    return df_tr, df_val, df_te, enc_hr, enc_st


def plot_active_learning_curve(
    unlabeled_frac: float,
    dropout_rate: float,
    K: int,
    T: int,
    results_df: pd.DataFrame | dict[str, pd.DataFrame],
    save_path: str | Path | None = None,
    title: str = "Active Learning Curve",
    figsize: tuple = (10, 6),
    show_error_bars: bool = True,
    # al_label: str = f"Uncertainty Sampling {unlabeled_frac*100:.0f}% Unlabeled",
    # random_label: str = f"Random Sampling {unlabeled_frac*100:.0f}% Unlabeled",
    upper_bound_auc: float | None = None,
    # upper_bound_label: str = "100% labeled",
):
    """
    Plot the active learning curve showing AUC performance over rounds.
    Can plot both uncertainty-based and random sampling results for comparison.
    
    Args:
        results_df: Either:
                   - Single DataFrame containing active learning results with columns:
                     'round', 'AUC_Mean', 'AUC_STD' (optional)
                   - Dictionary with keys 'uncertainty' and/or 'random', each containing
                     a DataFrame with the same structure
        save_path: Optional path to save the plot. If None, plot is not saved.
        title: Title for the plot
        figsize: Figure size tuple (width, height)
        show_error_bars: If True and 'AUC_STD' column exists, show error bars
        al_label: Label for uncertainty-based active learning curve
        random_label: Label for random sampling curve
        upper_bound_auc: Optional AUC value representing the upper bound (e.g., 
                        performance on full training set). If provided, shown as 
                        a horizontal reference line.
        upper_bound_label: Label for the upper bound line
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    
    ## LABELS
    if unlabeled_frac is None:
        al_label: str = "Subset Sampling"
        random_label: str = "Random Sampling"
    else:
        al_label: str = f"Uncertainty Sampling ({unlabeled_frac*100:.0f}% Unlabeled)"
        random_label: str = f"Random Sampling ({unlabeled_frac*100:.0f}% Unlabeled)"
    upper_bound_label: str = "100% labeled"
    
    # Handle both single DataFrame and dictionary of DataFrames
    if isinstance(results_df, dict):
        al_results = results_df.get('uncertainty', None)
        random_results = results_df.get('random', None)
    else:
        # Single DataFrame - assume it's uncertainty-based
        al_results = results_df
        random_results = None
    
    # Validate and process uncertainty-based results
    if al_results is not None:
        if 'round' not in al_results.columns:
            raise ValueError("results_df must contain a 'round' column")
        if 'AUC_Mean' not in al_results.columns:
            raise ValueError("results_df must contain an 'AUC_Mean' column")
        
        agg_dict = {'AUC_Mean': 'mean'}
        if 'AUC_STD' in al_results.columns:
            agg_dict['AUC_STD'] = 'mean'
        if 'AUC_CI_LOW' in al_results.columns:
            agg_dict['AUC_CI_LOW'] = 'mean'
        if 'AUC_CI_HIGH' in al_results.columns:
            agg_dict['AUC_CI_HIGH'] = 'mean'
        al_rounds_data = (
            al_results.groupby('round')
            .agg(agg_dict)
            .reset_index()
            .sort_values('round')
        )
    
    # Validate and process random sampling results
    if random_results is not None:
        if 'round' not in random_results.columns:
            raise ValueError("random_results must contain a 'round' column")
        if 'AUC_Mean' not in random_results.columns:
            raise ValueError("random_results must contain an 'AUC_Mean' column")
        
        random_agg = {'AUC_Mean': 'mean'}
        if 'AUC_STD' in random_results.columns:
            random_agg['AUC_STD'] = 'mean'
        if 'AUC_CI_LOW' in random_results.columns:
            random_agg['AUC_CI_LOW'] = 'mean'
        if 'AUC_CI_HIGH' in random_results.columns:
            random_agg['AUC_CI_HIGH'] = 'mean'
        random_rounds_data = (
            random_results.groupby('round')
            .agg(random_agg)
            .reset_index()
            .sort_values('round')
        )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Determine y-axis range from all data (including upper bound if provided)
    all_auc_means = []
    if al_results is not None:
        all_auc_means.extend(al_rounds_data['AUC_Mean'].values)
    if random_results is not None:
        all_auc_means.extend(random_rounds_data['AUC_Mean'].values)
    if upper_bound_auc is not None:
        all_auc_means.append((upper_bound_auc))

    if not all_auc_means:
        raise ValueError("No valid results data provided")

    # Calculate y-axis limits first (before plotting) to clip error bars
    y_min, y_max = min(all_auc_means), max(all_auc_means)
    y_range = y_max - y_min
    y_lim_low = max(0, y_min - 0.05 * y_range - 0.05)
    y_lim_high = min(1, y_max + 0.05 * y_range + 0.05)
    
    # Plot uncertainty-based active learning
    if al_results is not None:
        rounds = al_rounds_data['round'].values
        auc_means = al_rounds_data['AUC_Mean'].values
        
        if show_error_bars:
            if 'AUC_CI_LOW' in al_rounds_data.columns and 'AUC_CI_HIGH' in al_rounds_data.columns:
                lower_bounds = al_rounds_data['AUC_CI_LOW'].values
                upper_bounds = al_rounds_data['AUC_CI_HIGH'].values
                yerr_lower = np.clip(auc_means - lower_bounds, 0, auc_means - y_lim_low)
                yerr_upper = np.clip(upper_bounds - auc_means, 0, y_lim_high - auc_means)
            elif 'AUC_STD' in al_rounds_data.columns:
                auc_stds = al_rounds_data['AUC_STD'].values
                yerr_lower = np.minimum(auc_stds, auc_means - y_lim_low)
                yerr_upper = np.minimum(auc_stds, y_lim_high - auc_means)
            else:
                yerr_lower = yerr_upper = None
            if yerr_lower is not None:
                ax.errorbar(
                    rounds,
                    auc_means,
                    yerr=[yerr_lower, yerr_upper],
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=8,
                    capsize=5,
                    capthick=2,
                    label=al_label,
                    color='#2E86AB',  # Blue color
                    alpha=0.8,
                    errorevery=1
                )
                ax.relim()
                ax.autoscale_view()
            else:
                ax.plot(
                    rounds,
                    auc_means,
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=8,
                    label=al_label,
                    color='#2E86AB',  # Blue color
                    alpha=0.8
                )
                ax.relim()
                ax.autoscale_view()
        else:
            ax.plot(
                rounds,
                auc_means,
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=8,
                label=al_label,
                color='#2E86AB',  # Blue color
                alpha=0.8
            )
            ax.relim()
            ax.autoscale_view()
    # Plot random sampling
    if random_results is not None:
        rounds = random_rounds_data['round'].values
        auc_means = random_rounds_data['AUC_Mean'].values
        
        if show_error_bars:
            if 'AUC_CI_LOW' in random_rounds_data.columns and 'AUC_CI_HIGH' in random_rounds_data.columns:
                lower_bounds = random_rounds_data['AUC_CI_LOW'].values
                upper_bounds = random_rounds_data['AUC_CI_HIGH'].values
                yerr_lower = np.clip(auc_means - lower_bounds, 0, auc_means - y_lim_low)
                yerr_upper = np.clip(upper_bounds - auc_means, 0, y_lim_high - auc_means)
            elif 'AUC_STD' in random_rounds_data.columns:
                auc_stds = random_rounds_data['AUC_STD'].values
                auc_stds = np.clip(auc_stds, 0, 0.15)
                yerr_lower = np.minimum(auc_stds, auc_means - y_lim_low)
                yerr_upper = np.minimum(auc_stds, y_lim_high - auc_means)
            else:
                yerr_lower = yerr_upper = None
            if yerr_lower is not None:
                ax.errorbar(
                    rounds,
                    auc_means,
                    yerr=[yerr_lower, yerr_upper],
                    marker='s',
                    linestyle='--',
                    linewidth=2,
                    markersize=8,
                    capsize=5,
                    capthick=2,
                    label=random_label,
                    color='#A23B72',  # Purple/magenta color
                    alpha=0.8,
                    errorevery=1
                )
            else:
                ax.plot(
                    rounds,
                    auc_means,
                    marker='s',
                    linestyle='--',
                    linewidth=2,
                    markersize=8,
                    label=random_label,
                    color='#A23B72',  # Purple/magenta color
                    alpha=0.8
                )
        else:
            ax.plot(
                rounds,
                auc_means,
                marker='s',
                linestyle='--',
                linewidth=2,
                markersize=8,
                label=random_label,
                color='#A23B72',  # Purple/magenta color
                alpha=0.8
            )
    
    # --- Add upper bound line (black dashed) ---
    if upper_bound_auc is not None:
        ax.axhline(
            y=upper_bound_auc,
            color='black',
            linestyle='--',
            linewidth=2,
            label=upper_bound_label
        )
    # Formatting
    ax.set_xlabel('Active Learning Round', fontsize=12)
    ax.set_ylabel('AUC (Area Under ROC Curve)', fontsize=12)
    ax.set_title(f"T={T}, K={K}, Dropout Rate={dropout_rate}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='lower right')
    
    # Set x-axis to show integer rounds (use union of all rounds if both present)
    if al_results is not None and random_results is not None:
        all_rounds = sorted(
            set(al_rounds_data['round'].values) |
            set(random_rounds_data['round'].values)
        )
    elif al_results is not None:
        all_rounds = sorted(al_rounds_data['round'].values)
    else:
        all_rounds = sorted(random_rounds_data['round'].values)
    
    max_ticks = 12
    tick_rounds = all_rounds
    if len(all_rounds) > max_ticks:
        step = int(np.ceil(len(all_rounds) / max_ticks))
        tick_rounds = all_rounds[::step]
    ax.set_xticks(tick_rounds)
    ax.set_xticklabels([int(r) for r in tick_rounds])
    
    # Set y-axis limits (already calculated above)
    # ax.set_ylim(y_lim_low, y_lim_high)
    plt.autoscale(enable=True, axis='y', tight=False)

    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Convert Path object to string if needed
        save_path_str = str(save_path) if isinstance(save_path, Path) else save_path
        plt.savefig(save_path_str, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path_str}")

    
    return fig, ax

# def _build_XY_from_windows(df: pd.DataFrame):
#     """
#     Convert a windows DataFrame into X (N, WINDOW_SIZE, 2) and y (N,).
#     Assumes df has columns: 'hr_seq', 'st_seq', 'state_val'
#     """
#     X = np.stack([np.vstack([h, s]).T for h, s in zip(df["hr_seq"], df["st_seq"])])
#     y = df["state_val"].values.astype(int)
#     return X, y

import numpy as np
import pandas as pd

def _build_XY_from_windows(
    df: pd.DataFrame,
):
    """
    Build X (N, window_size, 2) and y (N,) from df with columns hr_seq, st_seq, state_val.
    """
    
    X = np.stack([
    np.vstack([h, s]).T
    for h, s in zip(df["hr_seq"], df["st_seq"])
])

    y = df["state_val"].values.astype(int)

    return X, y

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
    model_path = sdir / f'cnn_classifier.keras'
    if model_path.exists():
        m = load_model(model_path)
        # rep_model = tf.keras.Model(m.input, m.get_layer("combined").output)
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

    X_val_u, y_val_u = _build_XY_from_windows(df_val_u)

    # # 3) Build & compile the model
    inp = layers.Input(shape=(WINDOW_SIZE, 2))

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
    
#     m.compile(
#     optimizer=Adam(3e-4),
#     loss="binary_crossentropy",
#     metrics=[tf.keras.metrics.AUC(name="auc")]
# )

    # 4) Compute class weights
    classes = np.unique(y)
    if len(classes) == 2:
        cw_vals = compute_class_weight('balanced', classes=classes, y=y)
        class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    else:
        class_weight = {0:1, 1:1}

    # # 5) Callbacks
    es = EarlyStopping(monitor='val_loss', patience=GS_PATIENCE , restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    # es = EarlyStopping(
    #     monitor="val_loss",
    #     mode="min",
    #     patience=GS_PATIENCE,
    #     restore_best_weights=True,
    #     verbose=1
    # )

    # lr_cb = ReduceLROnPlateau(
    #     monitor="val_loss",
    #     mode="max",
    #     factor=0.5,
    #     patience=3,
    #     min_lr=1e-6,
    #     verbose=1
    # )
    # 6) SINGLE TRAINING pass
    hist = m.fit(
        X, y,
        validation_data=(X_val_u, y_val_u),
        batch_size=GS_BATCH,
        epochs=GS_EPOCHS,
        class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=2 ,
        shuffle=False,
    )

    # 7) Save & plot
    compare_pipelines.plot_clf_losses(hist.history['loss'], hist.history['val_loss'], sdir, 'global_cnn_lstm_loss')
    m.save(model_path)
    return m, sdir



# def train_and_eval(current_df_lab: pd.DataFrame, Z_val, y_val, Z_te, y_te, init_model: tf.keras.Model):

#     prob_te = init_model.predict(Z_te, verbose=0)[:, 0]
#     test_auc_mean, test_auc_std, valid_frac = utility.bootstrap_auc(y_te, prob_te)
#     return test_auc_mean, test_auc_std, init_model

def train_and_eval(current_df_lab: pd.DataFrame, Z_val, y_val, Z_te, y_te, init_model: tf.keras.Model):
    '''
    Helper function to train and evaluate a global supervised model on the current labeled set.
    used in run_al function
    '''
    ###set hyperparameters 
    GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 15
    WINDOW_LEN                              = WINDOW_SIZE


    X_lab, y_lab = _build_XY_from_windows(
        current_df_lab
    )

    # class weights (balanced)
    classes = np.unique(y_lab)

    cw_vals = compute_class_weight('balanced', classes=np.unique(y_lab), y=y_lab)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    ##Fit (fresh weights each round)
    m = tf.keras.Model.from_config(init_model.get_config())
    m.set_weights(init_model.get_weights())
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"])
                # metrics=[tf.keras.metrics.AUC(name="auc")])


    es = EarlyStopping(monitor='val_loss', patience=GS_PATIENCE, restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    hist = m.fit(
        X_lab, y_lab,
        validation_data=(Z_val, y_val),
        batch_size=GS_BATCH,
        epochs=GS_EPOCHS,
        class_weight=class_weight,
        verbose=2,
        callbacks=[es, lr_cb],

    )

    # Evaluate

    p_val = m.predict(Z_val, verbose=0).ravel()
    p_te = m.predict(Z_te, verbose=0).ravel()

    val_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) == 2 else np.nan
    test_auc_mean, test_auc_std, valid_frac = utility.bootstrap_auc(y_te, p_te)
    
    return test_auc_mean, test_auc_std

def reset_seeds():
    import random, numpy as np, tensorflow as tf
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def build_global_cnn_lstm():
    inp = layers.Input(shape=(WINDOW_SIZE, 2))

    x = layers.Conv1D(64, 8, padding='same', activation='relu',
                      kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(4, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(128, 5, padding='same', activation='relu',
                      kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(8, activation='relu')(se)
    se = layers.Dense(128, activation='sigmoid')(se)
    se = layers.Reshape((1, 128))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    cnn_out = layers.GlobalAveragePooling1D()(x)

    lstm_out = layers.LSTM(128)(inp)
    lstm_out = layers.Dropout(0.5)(lstm_out)

    combined = layers.concatenate([cnn_out, lstm_out])
    out = layers.Dense(1, activation='sigmoid')(combined)

    model = Model(inp, out)
    return model


from typing import Callable
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.class_weight import compute_class_weight
from typing import Callable, Optional
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.class_weight import compute_class_weight

def train_and_eval_v2(
    current_df_lab,
    Z_val, y_val,
    Z_te, y_te,
    model_builder: Callable[[], tf.keras.Model],
    init_weights: list,
    model: Optional[tf.keras.Model] = None,   # <-- NEW: previous-round model (warm start)
    epochs: int = None,
    batch_size: int = None,
    patience: int = None,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: int = 0,
):
    """
    Warm-start training + evaluation helper for AL.

    - If `model` is None: create a new model, load `init_weights` (Round 0)
    - If `model` is provided: continue training from it (Round 1+)

    Returns:
      model, auc_mean, auc_std, best_thr, hist_dict
    """

    # fallback to global constants if not given
    if epochs is None:
        epochs = globals().get("GS_EPOCHS", 200)
    if batch_size is None:
        batch_size = globals().get("GS_BATCH", 32)
    if patience is None:
        patience = globals().get("GS_PATIENCE", 15)

    # RNGs (for reproducible shuffling / bootstrapping)
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 0) build X_lab, y_lab from DataFrame
    X_lab, y_lab = _build_XY_from_windows(current_df_lab)
    if len(X_lab) == 0:
        raise RuntimeError("No labeled examples passed to train_and_eval_v2!")

    # 1) INIT or WARM-START
    if model is None:
        # Round 0: start from fixed initial weights (e.g., trained-on-10% weights)
        model = model_builder()
        model.set_weights(init_weights)

    # (Re)compile each round with a fresh optimizer (simple warm-start)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 2) class weights (balanced)
    classes = np.unique(y_lab)
    class_weight = None
    if len(classes) == 2:
        cw_vals = compute_class_weight("balanced", classes=classes, y=y_lab)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    # 3) callbacks
    es = EarlyStopping(monitor='val_loss', patience=GS_PATIENCE, restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)


    # 4) shuffle training (deterministic)
    # X_lab, y_lab = sk_shuffle(X_lab, y_lab, random_state=seed)

    # 5) fit (continues training if model was passed in)
    hist = model.fit(
        X_lab, y_lab,
        validation_data=(Z_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=verbose,
        shuffle=True,
    )

    # 6) select threshold on the labeled set
    best_thr = select_threshold_train(model, X_lab, y_lab)

    # 7) evaluate on test with bootstrap metrics
    probs_te = model.predict(Z_te, verbose=0).ravel()
    df_boot, auc_m, auc_s = bootstrap_threshold_metrics(
        y_te,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=seed
    )

    hist_dict = hist.history if hasattr(hist, "history") else {}
    return model, auc_m, auc_s, best_thr, hist_dict



def train_base_init_on_labeled(
    df_tr_labeled,
    Z_val, y_val,
    model_builder: Callable[[], tf.keras.Model],
    *,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 15,
    seed: int = 42,
    verbose: int = 2,
):
    """
    Train a base initialization model on the initial labeled set (e.g., 10%).
    Returns the trained weights to be used as init for every AL round.

    Output:
      init_weights_trained, base_hist_dict
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_init, y_init = _build_XY_from_windows(df_tr_labeled)
    if len(X_init) == 0:
        raise RuntimeError("df_tr_labeled produced 0 training windows")

    model = model_builder()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # class weights (balanced) for init set
    # 4) Compute class weights
    classes = np.unique(y_init)
    if len(classes) == 2:
        cw_vals = compute_class_weight('balanced', classes=classes, y=y_init)
        class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    else:
        class_weight = {0:1, 1:1}
        
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    # Re-seed right before training to keep runs deterministic
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    hist = model.fit(
        X_init, y_init,
        validation_data=(Z_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=verbose,
        # shuffle=False,
    )

    init_weights_trained = model.get_weights()
    hist_dict = hist.history if hasattr(hist, "history") else {}
    return init_weights_trained, hist_dict, model
