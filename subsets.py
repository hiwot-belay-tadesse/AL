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

import utility
import uq_utility
from preprocess import prepare_data
from src.chart_utils import bootstrap_threshold_metrics

# Hyperparameters (mirrors no_leak.py)
BATCH_SSL, SSL_EPOCHS = 32, 100
CLF_EPOCHS, CLF_PATIENCE = 500, 20
UNLABELED_FRAC = 0.7
dropout_rate = 0.2
T = 50

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
        "--budget",
        type=int,
        default=5,
        help="Number of active learning rounds to run.",
    )
    pa.add_argument(
        "--K",
        type=int,
        default=3,
        help="Number of points to select from each day per round.",
    )
    pa.add_argument(
        "--dropout-rate",
        type=float,
        default=dropout_rate,
        help="Dropout rate for the classifier.",
    )
    pa.add_argument(
        "--mc-passes",
        type=int,
        default=T,
        help="Number of MC-dropout passes (T).",
    )
    args = pa.parse_args()
    
    budget = args.budget
    K = args.K
    dropout_rate = args.dropout_rate
    T = args.mc_passes

    # Override global default
    global RESULTS_SUBDIR
    RESULTS_SUBDIR = args.results_subdir

    # Allow runtime overrides for dropout rate and MC-dropout passes



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

    Z_val_hr,  Z_val_st  = utility.encode(df_val, enc_hr, enc_st)
    Z_val = np.concatenate([Z_val_hr,  Z_val_st],  axis=1).astype('float32')
    y_val = df_val["state_val"].values.astype("float32")
    
    # Drop temporary columns
    # df_tr = df_tr.drop(columns=['hr_std', 'st_std', 'combined_std'])

    # df_tr_labeled, df_tr_unlabeled = utility.seed_variance_stratified(
    #     df_tr, UNLABELED_FRAC
    # )
    

    def make_day_seed_balanced(df, day_col="day", label_col="state_val", random_state=42):
        """
        Build a seed set with:
        - at least one sample from each day
        - as close to class balance (0/1) as possible

        Assumes binary labels in `label_col` with values {0, 1}.
        """
        rng = np.random.default_rng(random_state)
        seed_indices = []
        n_pos = 0
        n_neg = 0

        # ensure deterministic day order
        for d in sorted(df[day_col].unique()):
            day_df = df[df[day_col] == d]

            pos_idx = day_df[day_df[label_col] == 1].index.to_list()
            neg_idx = day_df[day_df[label_col] == 0].index.to_list()

            if not pos_idx and not neg_idx:
                # no labeled data for this day (shouldn't happen usually)
                continue

            # Decide which class to favor to keep global balance
            if n_pos > n_neg:
                # currently too many positives → prefer negative if available
                if neg_idx:
                    chosen = rng.choice(neg_idx)
                    n_neg += 1
                else:
                    chosen = rng.choice(pos_idx)
                    n_pos += 1
            elif n_neg > n_pos:
                # currently too many negatives → prefer positive if available
                if pos_idx:
                    chosen = rng.choice(pos_idx)
                    n_pos += 1
                else:
                    chosen = rng.choice(neg_idx)
                    n_neg += 1
            else:
                # currently balanced → pick randomly among available classes
                if pos_idx and neg_idx:
                    if rng.random() < 0.5:
                        chosen = rng.choice(pos_idx)
                        n_pos += 1
                    else:
                        chosen = rng.choice(neg_idx)
                        n_neg += 1
                elif pos_idx:  # only positives that day
                    chosen = rng.choice(pos_idx)
                    n_pos += 1
                else:          # only negatives that day
                    chosen = rng.choice(neg_idx)
                    n_neg += 1

            seed_indices.append(chosen)

        print(f"Seed class counts: pos={n_pos}, neg={n_neg}")
        return seed_indices
    
    seed_indices = make_day_seed_balanced(df_tr)
    df_labeled = df_tr.loc[seed_indices]
    df_unlabeled = df_tr.drop(seed_indices)
    

    
    day_groups = {
        day: df_unlabeled[df_unlabeled['day'] == day].index.tolist()
        for day in df_unlabeled['day'].unique()
    }
    
    Z_labeled_hr, Z_labeled_st = utility.encode(df_labeled, enc_hr, enc_st)
    Z_L = np.concatenate([Z_labeled_hr, Z_labeled_st], axis=1).astype('float32')
    y_L = df_labeled['state_val'].values.astype('float32')
    
    # Encode test set once
    Z_test_hr, Z_test_st = utility.encode(df_te, enc_hr, enc_st)
    Z_test = np.concatenate([Z_test_hr, Z_test_st], axis=1).astype('float32')
    y_test = df_te["state_val"].values.astype("float32")
    
    # Setup fit_kwargs for model training
    fit_kwargs = dict(
        epochs=CLF_EPOCHS,
        batch_size=10,
        verbose=0,
        CLF_PATIENCE=CLF_PATIENCE,
        dropout_rate=dropout_rate,
        validation_data=(Z_val, y_val),
    )
    
    # Round 0: Evaluate on initial seed set
    print("\n[Round 0] Evaluating on initial seed set...")
    auc_m_0, auc_s_0= utility.fit_and_eval(
        fit_kwargs=fit_kwargs,
        clf_builder=utility.clf_builder,
        input_dim=Z_L.shape[1],
        X_lab=Z_L,
        y_lab=y_L,
        X_val=Z_val,
        y_val=y_val,
        X_test=Z_test,
        y_test=y_test,
        CLF_PATIENCE=CLF_PATIENCE,
        dropout_rate=dropout_rate,
    )
    print(f"  Round 0: n_labeled={len(Z_L)}, AUC={auc_m_0:.4f} ± {auc_s_0:.4f}")
            
    auc_results = [{
        "round": 0,
        "n_labeled": len(Z_L),
        "AUC_Mean": auc_m_0,
        "AUC_STD": auc_s_0,
    }]
    
    # -------------------------------
    # Day-based selection AL loop
    # -------------------------------
    # Active learning rounds: select K points from each day per round
    for round_num in range(1, budget + 1):
        # Get all unique days remaining in unlabeled set
        remaining_days = sorted(df_unlabeled["day"].unique().tolist())
        
        if len(remaining_days) == 0:
            print(f"\n[Round {round_num}] No more days available, stopping.")
            break
        
        print(f"\n[Round {round_num}] Selecting {K} window(s) from each of {len(remaining_days)} days...")
        
        # Select K windows from each day

        selected_idxs_all = []
        for d in remaining_days:
            day_idxs = day_groups[d]
            
            if len(day_idxs) == 0:
                continue
            
            # Select K windows from this day (or all if fewer than K remain)
            n_select = min(K, len(day_idxs))
            sampled = df_unlabeled.loc[day_idxs].sample(
                n=n_select, random_state=42 + round_num
            )
            selected_indices = sampled.index.tolist()
            selected_idxs_all.extend(selected_indices)
        
        if len(selected_idxs_all) == 0:
            print("  No more windows available, stopping.")
            break
        
        # Reveal labels for all selected windows
        selected_df = df_unlabeled.loc[selected_idxs_all]
        Z_new_hr, Z_new_st = utility.encode(selected_df, enc_hr, enc_st)
        Z_new = np.concatenate([Z_new_hr, Z_new_st], axis=1).astype('float32')
        y_new = df_unlabeled.loc[selected_idxs_all]['state_val'].values.astype('float32')
        
        # Add all selected windows to labeled pool
        Z_L = np.concatenate([Z_L, Z_new], axis=0)
        y_L = np.concatenate([y_L, y_new], axis=0)
        
        # Update unlabeled set and day_groups
        df_unlabeled = df_unlabeled.drop(index=selected_idxs_all)
        for d in remaining_days:
            if d in day_groups:
                day_groups[d] = [idx for idx in day_groups[d] if idx not in selected_idxs_all]
        
        # Train model and evaluate on test set
        
        auc_m, auc_s = utility.fit_and_eval(
            fit_kwargs=fit_kwargs,
            clf_builder=utility.clf_builder,
            input_dim=Z_L.shape[1],
            X_lab=Z_L,
            y_lab=y_L,
            X_val=Z_val,
            y_val=y_val,
            X_test=Z_test,
            y_test=y_test,
            CLF_PATIENCE=CLF_PATIENCE,
            dropout_rate=dropout_rate,
        )
        
        n_selected = len(selected_idxs_all)
        n_labeled = len(Z_L)
        print(
            f"  Round {round_num}: selected {n_selected} windows, "
            f"n_labeled={n_labeled}, AUC={auc_m:.4f} ± {auc_s:.4f}"
        )
        
        auc_results.append({
            "round": round_num,
            "n_labeled": len(Z_L),
            "AUC_Mean": auc_m, 
            "AUC_STD": auc_s,
        })
    
    # -------------------------------
    # Random sampling AL loop
    # -------------------------------
    # Reset to initial state for random sampling
    print("\n" + "="*60)
    print("Starting Random Sampling Active Learning")
    print("="*60)
    
    seed_indices_rand = make_day_seed_balanced(df_tr)
    df_labeled_rand = df_tr.loc[seed_indices_rand].copy()
    df_unlabeled_rand = df_tr.drop(seed_indices_rand).copy()
    
    Z_labeled_rand_hr, Z_labeled_rand_st = utility.encode(df_labeled_rand, enc_hr, enc_st)
    Z_L_rand = np.concatenate([Z_labeled_rand_hr, Z_labeled_rand_st], axis=1).astype('float32')
    y_L_rand = df_labeled_rand['state_val'].values.astype('float32')
    
    # Round 0 for random (same as day-based)
    print("\n[Random Round 0] Evaluating on initial seed set...")
    auc_m_0_rand, auc_s_0_rand = utility.fit_and_eval(
        fit_kwargs=fit_kwargs,
        clf_builder=utility.clf_builder,
        input_dim=Z_L_rand.shape[1],
        X_lab=Z_L_rand,
        y_lab=y_L_rand,
        X_val=Z_val,
        y_val=y_val,
        X_test=Z_test,
        y_test=y_test,
        CLF_PATIENCE=CLF_PATIENCE,
        dropout_rate=dropout_rate,
    )
    print(f"  Random Round 0: n_labeled={len(Z_L_rand)}, AUC={auc_m_0_rand:.4f} ± {auc_s_0_rand:.4f}")
    
    auc_results_random = [{
        "round": 0,
        "n_labeled": len(Z_L_rand),
        "AUC_Mean": auc_m_0_rand,
        "AUC_STD": auc_s_0_rand,
    }]
    
    # Random sampling rounds
    for round_num in range(1, budget + 1):
        if len(df_unlabeled_rand) == 0:
            print(f"\n[Random Round {round_num}] No more data available, stopping.")
            break
        
        # Calculate how many points to select (K per day, but for random we select total)
        # To match day-based selection, we select the same total number
        remaining_days_rand = sorted(df_unlabeled_rand["day"].unique().tolist())
        n_days = len(remaining_days_rand)
        total_to_select = min(K * n_days, len(df_unlabeled_rand))
        
        if total_to_select == 0:
            print(f"\n[Random Round {round_num}] No more data available, stopping.")
            break
        
        print(f"\n[Random Round {round_num}] Randomly selecting {total_to_select} windows...")
        
        # Randomly select points
        selected_idxs_rand = df_unlabeled_rand.sample(
            n=total_to_select, random_state=42 + round_num
        ).index.tolist()
        
        # Reveal labels for selected windows
        selected_df_rand = df_unlabeled_rand.loc[selected_idxs_rand]
        Z_new_rand_hr, Z_new_rand_st = utility.encode(selected_df_rand, enc_hr, enc_st)
        Z_new_rand = np.concatenate([Z_new_rand_hr, Z_new_rand_st], axis=1).astype('float32')
        y_new_rand = df_unlabeled_rand.loc[selected_idxs_rand]['state_val'].values.astype('float32')
        
        # Add to labeled pool
        Z_L_rand = np.concatenate([Z_L_rand, Z_new_rand], axis=0)
        y_L_rand = np.concatenate([y_L_rand, y_new_rand], axis=0)
        
        # Update unlabeled set
        df_unlabeled_rand = df_unlabeled_rand.drop(index=selected_idxs_rand)
        
        # Train model and evaluate
        auc_m_rand, auc_s_rand = utility.fit_and_eval(
            fit_kwargs=fit_kwargs,
            clf_builder=utility.clf_builder,
            input_dim=Z_L_rand.shape[1],
            X_lab=Z_L_rand,
            y_lab=y_L_rand,
            X_val=Z_val,
            y_val=y_val,
            X_test=Z_test,
            y_test=y_test,
            CLF_PATIENCE=CLF_PATIENCE,
            dropout_rate=dropout_rate,
        )
        
        n_selected_rand = len(selected_idxs_rand)
        n_labeled_rand = len(Z_L_rand)
        print(
            f"  Random Round {round_num}: selected {n_selected_rand} windows, "
            f"n_labeled={n_labeled_rand}, AUC={auc_m_rand:.4f} ± {auc_s_rand:.4f}"
        )
        
        auc_results_random.append({
            "round": round_num,
            "n_labeled": len(Z_L_rand),
            "AUC_Mean": auc_m_rand,
            "AUC_STD": auc_s_rand,
        })
    
    # -------------------------------
    # Save results
    # -------------------------------
    results_df = pd.DataFrame(auc_results)
    results_df_random = pd.DataFrame(auc_results_random)
    
    results_dir = user_root / f"{pool}" / "day_subsets"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(
        results_dir / "day_subsets_results.csv",
        index=False,
        float_format="%.3f",
    )
    results_df_random.to_csv(
        results_dir / "random_results.csv",
        index=False,
        float_format="%.3f",
    )
    
    print(f"\nSaved day subsets results to: {results_dir / 'day_subsets_results.csv'}")
    print(f"\nSaved random sampling results to: {results_dir / 'random_results.csv'}")
    print(f"\nDay-based selection results:")
    print(results_df.to_string(index=False))
    print(f"\nRandom sampling results:")
    print(results_df_random.to_string(index=False))
    
    # -------------------------------
    # Compute upper bound AUC (train on full training set)
    # -------------------------------
    print("\n[Upper Bound] Training on full training set...")
    Z_tr_full_hr, Z_tr_full_st = utility.encode(df_tr, enc_hr, enc_st)
    Z_tr_full = np.concatenate([Z_tr_full_hr, Z_tr_full_st], axis=1).astype('float32')
    y_tr_full = df_tr["state_val"].values.astype("float32")
    
    unique_classes = np.unique(y_tr_full)
    from sklearn.utils.class_weight import compute_class_weight
    cw_vals = compute_class_weight(
        'balanced', classes=unique_classes, y=y_tr_full
    )
    class_weight = {
        int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)
    }
    clf_full, es = utility.clf_builder(
        input_dim=Z_tr_full.shape[1],
        CLF_PATIENCE=CLF_PATIENCE,
        dropout_rate=dropout_rate
    )
    
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
    
    clf_full.fit(
        Z_tr_full, y_tr_full, class_weight=class_weight,
        **fit_kwargs_with_callbacks
    )
    best_thr = utility.select_threshold_train(clf_full, Z_val, y_val)
    probs_te = clf_full.predict(Z_test, verbose=0).ravel()
    
    upper_bound_auc_m, upper_bound_auc_s = utility.bootstrap_auc(
        y_test,
        probs_te,)
    print(f"  Upper bound AUC: {upper_bound_auc_m:.4f} ± {upper_bound_auc_s:.4f}")
    
    # -------------------------------
    # Plot AUC per round (similar to main.py)
    # -------------------------------
    uq_utility.plot_active_learning_curve(
        unlabeled_frac=None,
        results_df={"uncertainty": results_df, "random": results_df_random},
        save_path=results_dir / "day_subsets_vs_random_auc_curve.png",
        title=f"Day Subsets vs Random Sampling: AUC per Round (K={K} points/day)",
        upper_bound_auc=upper_bound_auc_m,
    )
        