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

# Hyperparameters (mirrors main.py)
BATCH_SSL, SSL_EPOCHS = 32, 100
CLF_EPOCHS, CLF_PATIENCE = 500, 20
UNLABELED_FRAC = 0.5
K = 20
budget = 5
dropout_rate = 0.3
T = 30
pool = "global"
uncertainity_measure = "mc"  # "mc" or "margin"

# Number of seeds to run
N_SEEDS = 4
SEEDS = [41, 42, 43, 44]  # Different seeds for train_test_split


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
    pa.add_argument(
        "--n-seeds",
        type=int,
        default=N_SEEDS,
        help="Number of different seeds to run (default: 4).",
    )
    args = pa.parse_args()
    
    # Override global default
    global RESULTS_SUBDIR
    RESULTS_SUBDIR = args.results_subdir
    n_seeds = args.n_seeds
    seeds = SEEDS[:n_seeds]  # Use first n_seeds seeds

    # Top‑level paths
    top_out = Path(args.output_dir)
    shared_enc_root = top_out / "_global_encoders"

    # Seed everything (for reproducibility of encoders, etc.)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------------
    # 1) Preprocess & get encoders (same for all seeds)
    # -------------------------------
    print("="*60)
    print("Preprocessing data and loading/training encoders...")
    print("="*60)
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

    # Encode validation and test sets (same for all seeds)
    Z_val_hr, Z_val_st = utility.encode(df_val, enc_hr, enc_st)
    Z_val = np.concatenate([Z_val_hr, Z_val_st], axis=1).astype('float32')
    y_val = df_val["state_val"].values.astype("float32")
    
    # Setup fit_kwargs
    fit_kwargs = dict(
        epochs=CLF_EPOCHS,
        batch_size=10,
        verbose=0,
        CLF_PATIENCE=CLF_PATIENCE,
        dropout_rate=dropout_rate,
        validation_data=(Z_val, y_val),  
    )

    # -------------------------------
    # 2) Run active learning with multiple seeds
    # -------------------------------
    print("\n" + "="*60)
    print(f"Running active learning with {n_seeds} different seeds: {seeds}")
    print("="*60)
    
    all_uncertainty_results = []
    all_random_results = []
    
    for seed_idx, split_seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx + 1}/{n_seeds}: split_seed={split_seed}")
        print(f"{'='*60}")
        
        # Reset seeds for model training (but use different split_seed for data split)
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Create labeled/unlabeled split with this seed
        from sklearn.model_selection import train_test_split
        df_tr_labeled, df_tr_unlabeled = train_test_split(
            df_tr, 
            test_size=UNLABELED_FRAC, 
            stratify=df_tr["state_val"], 
            random_state=split_seed
        )
        df_tr_unlabeled['true_label'] = df_tr_unlabeled['state_val']
        df_tr_unlabeled['state_val'] = -1
        
        print(f"Split (seed={split_seed}) - Labeled: {len(df_tr_labeled)}, Unlabeled: {len(df_tr_unlabeled)}")
        print(f"  Labeled - Positive: {(df_tr_labeled['state_val'] == 1).sum()}, Negative: {(df_tr_labeled['state_val'] == 0).sum()}")
        print(f"  Unlabeled - Positive: {(df_tr_unlabeled['true_label'] == 1).sum()}, Negative: {(df_tr_unlabeled['true_label'] == 0).sum()}")
        
        # Run uncertainty-based AL
        print(f"\n  Running uncertainty-based AL (seed {seed_idx + 1}/{n_seeds})...")
        al_progress_unc, _, _, _, _ = run_al(
            "uncertainty",
            df_tr_labeled.copy(),
            df_tr_unlabeled.copy(),
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
        al_progress_unc['seed'] = split_seed
        all_uncertainty_results.append(al_progress_unc)
        
        # Run random sampling AL
        print(f"\n  Running random sampling AL (seed {seed_idx + 1}/{n_seeds})...")
        al_progress_rand, _, _, _, _ = run_al(
            "random",
            df_tr_labeled.copy(),
            df_tr_unlabeled.copy(),
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
        al_progress_rand['seed'] = split_seed
        all_random_results.append(al_progress_rand)
        
        print(f"  Seed {seed_idx + 1} complete!")
    
    # -------------------------------
    # 3) Aggregate results across seeds
    # -------------------------------
    print("\n" + "="*60)
    print("Aggregating results across seeds...")
    print("="*60)
    
    # Combine all results
    combined_uncertainty = pd.concat(all_uncertainty_results, ignore_index=True)
    combined_random = pd.concat(all_random_results, ignore_index=True)
    
    # Compute mean and std across seeds for each round
    uncertainty_agg = combined_uncertainty.groupby('round').agg({
        'AUC_Mean': ['mean', 'std'],
        'AUC_STD': 'mean',  # Average the bootstrap std
    }).reset_index()
    uncertainty_agg.columns = ['round', 'AUC_Mean', 'AUC_Mean_Std', 'AUC_STD']
    
    random_agg = combined_random.groupby('round').agg({
        'AUC_Mean': ['mean', 'std'],
        'AUC_STD': 'mean',
    }).reset_index()
    random_agg.columns = ['round', 'AUC_Mean', 'AUC_Mean_Std', 'AUC_STD']
    
    # -------------------------------
    # 4) Save results
    # -------------------------------
    al_progress_dir = user_root / f"{pool}" / f"{uncertainity_measure}_seeded"
    al_progress_random_dir = user_root / f"{pool}" / "random_seeded"
    al_progress_dir.mkdir(parents=True, exist_ok=True)
    al_progress_random_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual seed results
    combined_uncertainty.to_csv(
        al_progress_dir / "all_seeds_uncertainty.csv",
        index=False,
        float_format="%.3f",
    )
    combined_random.to_csv(
        al_progress_random_dir / "all_seeds_random.csv",
        index=False,
        float_format="%.3f",
    )
    
    # Save aggregated results (mean across seeds)
    uncertainty_agg.to_csv(
        al_progress_dir / "averaged_uncertainty.csv",
        index=False,
        float_format="%.3f",
    )
    random_agg.to_csv(
        al_progress_random_dir / "averaged_random.csv",
        index=False,
        float_format="%.3f",
    )
    
    print(f"\nSaved results to:")
    print(f"  Uncertainty: {al_progress_dir}")
    print(f"  Random: {al_progress_random_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Average AUC across seeds")
    print("="*60)
    print("\nUncertainty Sampling:")
    print(uncertainty_agg[['round', 'AUC_Mean', 'AUC_Mean_Std']].to_string(index=False))
    print("\nRandom Sampling:")
    print(random_agg[['round', 'AUC_Mean', 'AUC_Mean_Std']].to_string(index=False))
    
    # -------------------------------
    # 5) Plot comparison with averaged results
    # -------------------------------
    # Compute upper bound AUC (same for all seeds)
    H_tr_full, S_tr_full = utility.encode(df_tr, enc_hr, enc_st)
    Z_tr_full = np.concatenate([H_tr_full, S_tr_full], axis=1).astype('float32')
    H_test, S_test = utility.encode(df_te, enc_hr, enc_st)
    Z_test = np.concatenate([H_test, S_test], axis=1).astype('float32')
    y_test = df_te["state_val"].values.astype("float32")
    y_lab = df_tr["state_val"].values.astype("float32")
    
    unique_classes = np.unique(y_lab)
    from sklearn.utils.class_weight import compute_class_weight
    cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_lab)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    
    clf_full, es = utility.clf_builder(input_dim=Z_tr_full.shape[1], CLF_PATIENCE=CLF_PATIENCE, dropout_rate=dropout_rate)
    fit_kwargs_with_callbacks = fit_kwargs.copy()
    fit_kwargs_with_callbacks.pop('CLF_PATIENCE', None)
    fit_kwargs_with_callbacks.pop('dropout_rate', None)
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
    
    # Plot with averaged results
    # Prepare DataFrames in the format expected by plot_active_learning_curve
    uncertainty_plot_df = uncertainty_agg[['round', 'AUC_Mean', 'AUC_STD']].copy()
    uncertainty_plot_df.columns = ['round', 'AUC_Mean', 'AUC_STD']
    random_plot_df = random_agg[['round', 'AUC_Mean', 'AUC_STD']].copy()
    random_plot_df.columns = ['round', 'AUC_Mean', 'AUC_STD']
    
    uq_utility.plot_active_learning_curve(
        unlabeled_frac=UNLABELED_FRAC,
        dropout_rate=dropout_rate,
        K=K,
        T=T,
        results_df={"uncertainty": uncertainty_plot_df, "random": random_plot_df},
        save_path=str(al_progress_dir / "comparison_averaged.png"),
        title=(
            f"Active Learning Comparison (Averaged over {n_seeds} seeds): "
            f"Uncertainty({uncertainity_measure}) vs Random Sampling"
        ),
        upper_bound_auc=upper_bound_auc_m,
    )
    
    print(f"\nSaved averaged comparison plot to: {al_progress_dir / 'comparison_averaged.png'}")
    print("\nExperiment complete!")

