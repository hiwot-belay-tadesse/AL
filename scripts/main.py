import argparse
from pathlib import Path
import os, sys
import random

import numpy as np
import pandas as pd

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
sys.path.append(str(Path(__file__).resolve().parents[1]))

import uq_utility, utility 
from utility import stratified_split_min, run_al
from preprocess import prepare_data


# Hyperparameters (mirrors no_leak.py)
BATCH_SSL, SSL_EPOCHS = 32, 100
CLF_EPOCHS, CLF_PATIENCE = 500, 20
# UNLABELED_FRAC = 0.5
# # K = 20
# # K = 5
# # budget = 5
dropout_rate = 0.5

# T = 30
pool = "global"
uncertainity_measure = "mc"  # "mc" or "margin"


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    pa = argparse.ArgumentParser()
    pa.add_argument("--user", required=True)
    pa.add_argument("--fruit", required=True)
    pa.add_argument("--scenario", required=True)
    pa.add_argument("--output-dir", default="results")
    pa.add_argument(
        "--sample-mode",
        choices=["original", "undersample", "oversample"],
        default="original",
        help=(
            "How to balance classes in TRAIN/VAL: keep original, undersample "
            "negs, or oversample pos."
        ),
    )
    pa.add_argument(
        "--results-subdir",
        default="results",
        help=(
            "Name of the subdirectory under each pipeline where results "
            "go (CSVs, plots, split_details)."
        ),
    )
    pa.add_argument(
        "--dropout-rate",
        type=float,
        default=dropout_rate,
        help="Override the dropout rate used inside the AL classifier.",
    )
    pa.add_argument(
        "--mc-passes",
        type=int,
        default=T,
        help="Override the number of MC-dropout passes (T).",
    )
    args = pa.parse_args()
    # breakpoint()
    # Override global default
    global RESULTS_SUBDIR
    RESULTS_SUBDIR = args.results_subdir

    # Allow runtime overrides for dropout rate and MC-dropout passes
    # dropout_rate = args.dropout_rate
    # T = args.mc_passes

    # Top‑level paths
    top_out = Path(args.output_dir)
    shared_enc_root = top_out / "_global_encoders"

    # Seed everything
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------------
    # 1) Preprocess & get encoders
    # -------------------------------
    enc_hr, enc_st, df_tr, df_val, df_te, df_all_tr, user_root = prepare_data(
        args=args,
        pool=pool,
        top_out=top_out,
        shared_enc_root=shared_enc_root,
        batch_ssl=BATCH_SSL,
        ssl_epochs=SSL_EPOCHS,
    )
    
    # Filter out flatline signals (where either hr_raw or st_raw has std <= 0)
    df_tr = df_tr.reset_index(drop=True)
    df_tr['hr_std'] = df_tr['hr_raw'].apply(np.std)
    df_tr['st_std'] = df_tr['st_raw'].apply(np.std)
    df_tr['combined_std'] = df_tr[['hr_std', 'st_std']].min(axis=1)
    df_tr = df_tr[df_tr['combined_std'] > 0].reset_index(drop=True)

    # breakpoint()
    Z_val_hr,  Z_val_st  = utility.encode(df_val, enc_hr, enc_st)
    Z_val = np.concatenate([Z_val_hr,  Z_val_st],  axis=1).astype('float32')
    y_val = df_val["state_val"].values.astype("float32")
    # Drop temporary columns
    # df_tr = df_tr.drop(columns=['hr_std', 'st_std', 'combined_std'])
    from sklearn.model_selection import train_test_split
    df_tr_labeled, df_tr_unlabeled = train_test_split(df_tr, test_size=UNLABELED_FRAC, stratify=df_tr["state_val"], random_state=41)
    df_tr_unlabeled['true_label'] = df_tr_unlabeled['state_val']
    df_tr_unlabeled['state_val'] = -1 
    # Add this:
    print(f"\n=== main.py AL split DEBUG ===")
    print(f"Labeled set - Positive: {(df_tr_labeled['state_val'] == 1).sum()}, Negative: {(df_tr_labeled['state_val'] == 0).sum()}")
    print(f"Unlabeled set - Positive: {(df_tr_unlabeled['true_label'] == 1).sum()}, Negative: {(df_tr_unlabeled['true_label'] == 0).sum()}")
    # breakpoint()
    # # Initial labeled / unlabeled split (no‑leak AL)
    # df_tr_labeled, df_tr_unlabeled = stratified_split_min(
    #     df_tr, "state_val", UNLABELED_FRAC, min_per_class=5, random_state=42
    # )
    
    # df_tr_labeled, df_tr_unlabeled = utility.seed_variance_stratified(df_tr, UNLABELED_FRAC)

    # df_tr_labeled, df_tr_unlabeled = utility.seed_representative_raw(df_tr, unlabeled_frac=UNLABELED_FRAC, method="kmeans")
    # -------------------------------
    # 2) Run active learning (uncertainty + random)
    # -------------------------------
    fit_kwargs = dict(
        epochs=CLF_EPOCHS,
        batch_size=10,
        verbose=0,
        CLF_PATIENCE=CLF_PATIENCE,
        dropout_rate=dropout_rate,
        validation_data=(Z_val, y_val),  
    )
    # al_progress, _, _, _, queried_indices_hybrid = run_al(
    #     "hybrid",
    #     df_tr_labeled,
    #     df_tr_unlabeled,
    #     df_val,
    #     df_te,
    #     enc_hr,
    #     enc_st,
    #     utility.clf_builder,
    #     utility.mc_predict_last_dropout,
    #     K=K,
    #     budget=budget,
    #     T=T,
    #     fit_kwargs=fit_kwargs,
    # )
    al_progress, clf, _, _, queried_indices = run_al(
        "uncertainty",
        df_tr_labeled,
        df_tr_unlabeled,
        df_val,
        df_te,
        enc_hr,
        enc_st,
        utility.clf_builder,
        utility.mc_predict_last_dropout,
        K=K,
        budget=budget,
        T=T,
        fit_kwargs=fit_kwargs,
    )

    al_progress_random, _, _, _, queried_indices_random = run_al(
        "random",
        df_tr_labeled,
        df_tr_unlabeled,
        df_val,
        df_te,
        enc_hr,
        enc_st,
        utility.clf_builder,
        utility.mc_predict_last_dropout,
        K=K,
        budget=budget,
        T=T,
        fit_kwargs=fit_kwargs,
    )
    

    # -------------------------------
    # 3) Save AL progress & plot
    # -------------------------------
    al_progress_dir = user_root / f"{pool}" / f"{uncertainity_measure}"
    al_progress_random_dir = user_root / f"{pool}" / "random"
    al_progress_dir.mkdir(parents=True, exist_ok=True)
    al_progress_random_dir.mkdir(parents=True, exist_ok=True)

    al_progress.to_csv(
        al_progress_dir / "no_leak_al_results.csv",
        index=False,
        float_format="%.3f",
    )
    al_progress_random.to_csv(
        al_progress_random_dir / "no_leak_al_results.csv",
        index=False,
        float_format="%.3f",
    )

    print("Saved AL progress to:", al_progress_dir / f"T_{T}_uncertainty.csv")
    print("Saved random sampling progress to:", al_progress_dir / f"T_{T}_random.csv")

    # Plot comparison
    al_df = pd.read_csv(al_progress_dir / "no_leak_al_results.csv")
    al_df_random = pd.read_csv(al_progress_random_dir / "no_leak_al_results.csv")
    # breakpoint()
    ## model on full train set to get upper bound AUC
    H_tr_full,  S_tr_full  = utility.encode(df_tr, enc_hr, enc_st)
    Z_tr_full   = np.concatenate([H_tr_full,  S_tr_full],  axis=1).astype('float32')
    H_val,  S_val  = utility.encode(df_val, enc_hr, enc_st)
    Z_val = np.concatenate([H_val,  S_val],  axis=1).astype('float32')
    y_val = df_val["state_val"].values.astype("float32")
    H_test,  S_test  = utility.encode(df_te, enc_hr, enc_st)
    Z_test = np.concatenate([H_test,  S_test],  axis=1).astype('float32')
    y_test = df_te["state_val"].values.astype("float32")
    y_lab = df_tr["state_val"].values.astype("float32")
    unique_classes = np.unique(y_lab)
    from sklearn.utils.class_weight import compute_class_weight
    cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_lab)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    clf_full, es  = utility.clf_builder(input_dim=Z_tr_full.shape[1], CLF_PATIENCE=CLF_PATIENCE, dropout_rate=dropout_rate)
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
    clf_full.fit(Z_tr_full, y_lab, class_weight=class_weight, **fit_kwargs_with_callbacks)
    best_thr = utility.select_threshold_train(clf_full, Z_val, y_val)
    probs_te = clf_full.predict(Z_test, verbose=0).ravel()
 
    df_boot, upper_bound_auc_m, upper_bound_auc_s = utility.bootstrap_threshold_metrics(
        y_test,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )


    uq_utility.plot_active_learning_curve(
        unlabeled_frac=UNLABELED_FRAC,
        results_df={"uncertainty": al_df, "random": al_df_random},
        save_path=al_progress_dir / "comparison_mc.png",
        title=(
            f"Active Learning Comparison : Uncertainty({uncertainity_measure}) "
            "vs Random Sampling"
        ),
        upper_bound_auc=upper_bound_auc_m,
    )

    # -------------------------------
    # 4) Plot queried points under AL/main.py folder
    # -------------------------------
    queried_base = user_root / "AL" / "queried_points"
    queried_unc_dir = queried_base / "uncertainty"
    queried_rand_dir = queried_base / "random"
    # utility.plot_top_k_picks(queried_indices, df_tr, queried_unc_dir)
    # utility.plot_top_k_picks(queried_indices_random, df_tr, queried_rand_dir)

