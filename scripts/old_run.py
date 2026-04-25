import json
import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import pickle

from sklearn.utils import compute_class_weight

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import tensorflow as tf


# from utility import stratified_split_min, run_al
import preprocess
from preprocess import prepare_data

import uq_utility
import utility
import random
import numpy as np
import tensorflow as tf  
import pandas as pd 

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from src import compare_pipelines as compare_pipelines

DRYRUN = True


# Default; overridden after parsing args.pool
OUTPUT_DIR = 'Refactor'
# dropout_rate = 0.3  
# UNLABELED_FRAC = 0.9 

# pool="personal"
breakpoint()
pa = argparse.ArgumentParser()
pa.add_argument("--user", default= "ID10" )
pa.add_argument("--pool", default="global")
pa.add_argument("--fruit", default="Nectarine")
pa.add_argument("--scenario", default="Crave")
pa.add_argument("--sample_mode", default="original")
pa.add_argument("--unlabeled_frac", default=0.7)
pa.add_argument("--dropout_rate", default=0.1)
pa.add_argument("--warm_start", default=1, help="1 = warm-start between rounds, 0 = retrain each round")
# pa.add_argument("--T", default=30)
# pa.add_argument("--K", default=10)
# pa.add_argument("--Budget", default=10)
pa.add_argument("--results_subdir", default="results")

# pa.add_argument("--aq_f", default="eu_sampling")


args, _ = pa.parse_known_args()

# Override OUTPUT_DIR based on pool
if args.pool == "personal":
    OUTPUT_DIR = "Refactor/P_SSL"
elif args.pool == "global":
    OUTPUT_DIR = "Refactor/global_SSL"
elif args.pool == "global_supervised":
    OUTPUT_DIR = "Refactor/GS"

# user = args.user
# fruit = args.fruit
# scenario = args.scenario
unlabeled_frac = [float(args.unlabeled_frac)]
# unlabeled_frac = [0.7]

dropout_rate = [float(args.dropout_rate)]
# Warm-start flag
warm_start = bool(int(args.warm_start))
# T = [30]
T = [30]
K = [40]
Budget = [None]

# Budget = args.Budget
RESULTS_SUBDIR = args.results_subdir
# aq_f = args.aq_f
aq_f = ["uncertainty", "random"]

BATCH_SSL, SSL_EPOCHS = 32, 100
CLF_EPOCHS, CLF_PATIENCE = 500, 20
GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 15


# Top‑level paths
# Use OUTPUT_DIR to match submit_batch.py
top_out = Path(OUTPUT_DIR) 
shared_enc_root = top_out / "_global_encoders"
shared_cnn_root = top_out / "global_cnns"



QUEUE = [
    ('uncertainty', dict(user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario], T=T, K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate, warm_start=[warm_start])),
    ('random', dict(user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario], K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate, warm_start=[warm_start])),
]

def reset_seeds(seed=42):
    import random, numpy as np, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

def run(exp_dir, exp_name, exp_kwargs):
    '''
    This is the function that will actually execute the job.
    To use it, here's what you need to do:
    1. Create directory 'exp_dir' as a function of 'exp_kwarg'.
       This is so that each set of experiment+hyperparameters get their own directory.
    2. Get your experiment's parameters from 'exp_kwargs'
    3. Run your experiment
    4. Store the results however you see fit in 'exp_dir'
    '''
    
    print('Running experiment {}:'.format(exp_name))
    
    print('with hyperparameters', exp_kwargs)
    print('\n')
    
    user = args.user
    pool = args.pool
    fruit = args.fruit
    scenario = args.scenario
    dropout_rate = float(args.dropout_rate)
    unlabeled_frac = float(args.unlabeled_frac)
    

    # 
    # -------------------------------
    # 1) Preprocess & get encoders
    # -------------------------------


    try:
        prep = prepare_data(
        args=args,
        top_out=top_out,
        shared_enc_root=shared_enc_root,
        shared_cnn_root=shared_cnn_root,
        batch_ssl=BATCH_SSL,
        ssl_epochs=SSL_EPOCHS, 
        pool=pool,
    )
    except (RuntimeError, ValueError, SystemExit) as e:
        print(f"Skipping user {args.user}: {e}")
        return
    if prep is None:
        print(f"Skipping user {args.user}: prepare_data returned no data.")
        return
    df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d, all_negatives = prep
    
    # Guard against empty splits (insufficient data for this participant)
    if df_tr is None or df_val is None or df_te is None:
        print(f"Skipping user {args.user}: missing train/val/test splits.")
        return
    if len(df_tr) == 0 or len(df_val) == 0 or len(df_te) == 0:
        print(f"Skipping user {args.user}: empty train/val/test split(s).")
        return
    if pool in ["global", "global_supervised"] and (df_all_tr is None or len(df_all_tr) == 0):
        print(f"Skipping user {args.user}: empty pooled training set.")
        return
    
    
    # Compute budget dynamically from dataset size.
    
    if pool == "personal":
        N = len(df_tr)
    elif pool == "global" or pool == "global_supervised":
        N = len(df_all_tr)
    else:
        print("Error: unknown pool type:", pool)
    k_val = exp_kwargs.get("K", K[0])
    if isinstance(k_val, (list, tuple)):
        k_val = k_val[0]
    k_val = int(k_val)
    uf_val = exp_kwargs.get("unlabeled_frac", unlabeled_frac)
    if isinstance(uf_val, (list, tuple)):
        uf_val = uf_val[0]
    uf_val = float(uf_val)
    budget = int(np.ceil(uf_val * N / k_val))
    budget = max(1, budget)
    exp_kwargs["Budget"] = budget

    t_val = exp_kwargs.get("T", None)
    if isinstance(t_val, (list, tuple)):
        t_val = t_val[0] if t_val else None
    dr_val = exp_kwargs.get("dropout_rate", dropout_rate)
    if isinstance(dr_val, (list, tuple)):
        dr_val = dr_val[0]
    dr_val = float(dr_val)
    # Build hyperparameter folder name here so Budget matches exp_summary.json
    hp = []
    hp.append(f"UF{int(uf_val*100)}")
    hp.append(f"K{k_val}")
    hp.append(f"B{budget}")
    if t_val is not None:
        hp.append(f"T{int(t_val)}")
    hp.append(f"DR{int(dr_val*100)}")
    hp_folder = "_".join(hp)

    exp_dir_path = Path(exp_dir)
    if exp_dir_path.name != hp_folder:
        exp_dir_path = exp_dir_path / hp_folder
    exp_dir = str(exp_dir_path)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    print('Results are stored in:', exp_dir)
    # 
    summary = {
        "user": user,
        "pool": pool,
        "fruit": fruit,
        "scenario": scenario,
        "dropout_rate": dropout_rate,
        "unlabeled_frac": uf_val,
        "T": exp_kwargs.get("T", None),
        "K": k_val,
        "Budget": budget,
    }
    summary_path = os.path.join(exp_dir, "exp_summary.json")
  
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved exp_summary.json to:", summary_path)
    
    # Filter out flatline signals (where either hr_raw or st_raw has std <= 0)
    if pool == "personal":
        df_tr = df_tr.reset_index(drop=True)
    elif pool == "global" or pool == "global_supervised":
        df_all_tr = df_all_tr.reset_index(drop=True)

    
    if pool == "personal" or pool == "global":
        ## Encode validation set 
        Z_val_hr, Z_val_st = utility.encode(df_val, enc_hr, enc_st)
        Z_val = np.concatenate([Z_val_hr,  Z_val_st],  axis=1).astype('float32')
        y_val = df_val["state_val"].values.astype("float32")
        
        ## Encode test set
        Z_test_hr, Z_test_st = utility.encode(df_te, enc_hr, enc_st)
        Z_test = np.concatenate([Z_test_hr,  Z_test_st],  axis=1).astype('float32')
        y_test = df_te["state_val"].values.astype("float32") 
    
    elif pool == "global_supervised":
        ## Encode validation set 
        Z_val, y_val = uq_utility._build_XY_from_windows(df_val)
        ## Encode test set
        Z_test, y_test = uq_utility._build_XY_from_windows(df_te)

    else: 
        print("Error: unknown pool type:", pool)
    
    fit_kwargs = dict(
    epochs=CLF_EPOCHS,
    batch_size=32,
    verbose=0,
)
    if len(Z_val) > 0:
        fit_kwargs["validation_data"] = (Z_val, y_val)

    fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}

    

    # df_tr_labeled, df_tr_unlabeled = train_test_split(df_all_tr, test_size=unlabeled_frac, stratify=df_all_tr["state_val"], random_state=48)
    # df_tr_labeled, df_tr_unlabeled = utility.make_labeled_unlabeled_with_target_quota(df_all_tr, target_uid=args.user, unlabeled_frac=unlabeled_frac)


    # ##Note the following setup is said for SAIL submission
    if pool == "personal":
        df_tr_labeled, df_tr_unlabeled = train_test_split(df_tr, test_size=unlabeled_frac, stratify=df_tr["state_val"], random_state=41)
    #     df_tr_labeled, df_tr_unlabeled = utility.seed_representative_raw(df_tr, unlabeled_frac=unlabeled_frac, method="kmeans")
    elif pool == "global" or pool == "global_supervised":
    #     # df_tr_labeled, df_tr_unlabeled = utility.seed_representative_raw(df_all_tr, unlabeled_frac=unlabeled_frac, method="kmeans")  
    #     df_tr_labeled, df_tr_unlabeled = utility.split_labeled_unlabeled_balanced(df_all_tr, "user_id", labeled_frac=1-unlabeled_frac, min_per_participant=4, random_state=42)
        # df_tr_labeled, df_tr_unlabeled = utility.make_labeled_unlabeled_with_target_quota(df_all_tr, target_uid=args.user, unlabeled_frac=unlabeled_frac)
        df_tr_labeled, df_tr_unlabeled = train_test_split(df_all_tr, test_size=unlabeled_frac, stratify=df_all_tr["state_val"], random_state=39)

    if df_tr_labeled is None or len(df_tr_labeled) == 0:
        print(f"Skipping user {args.user}: no labeled samples available.")
        return
    if df_tr_unlabeled is None:
        print(f"Skipping user {args.user}: unlabeled pool unavailable.")
        return
    
    # y_pred
    df_queried = pd.DataFrame()
    df_queried = df_tr_labeled.copy()

    # all_splits_labeled = uq_utility.build_all_splits_labeled(df_tr_labeled, all_splits)
    


    if pool == "global_supervised":
        init_weights_trained, base_hist, active_model = uq_utility.train_base_init_on_labeled(df_tr_labeled, Z_val, y_val, uq_utility.build_global_cnn_lstm)
        
    else:
        init_weights_trained=None
        
    
    warm_start_val = exp_kwargs.get("warm_start", warm_start)
    if isinstance(warm_start_val, (list, tuple)):
        warm_start_val = warm_start_val[0]
    warm_start_val = bool(int(warm_start_val)) if isinstance(warm_start_val, (int, str)) else bool(warm_start_val)

    seed_val = exp_kwargs.get("seed", 42)
    if isinstance(seed_val, (list, tuple)):
        seed_val = seed_val[0]
    seed_val = int(seed_val)

    al_progress, active_model, df_tr_labeled_final, df_tr_unlabeled_final, queried_indices, count_df = utility.run_al(
        exp_name,
        df_tr_labeled,
        df_tr_unlabeled,
        df_val,
        df_te,
        enc_hr,
        enc_st,
        utility.clf_builder,
        utility.mc_predict,
        K=exp_kwargs['K'],
        budget=budget,
        T=exp_kwargs['T'] if 'T' in exp_kwargs else None,
        CLF_PATIENCE=CLF_PATIENCE, 
        dropout_rate=dropout_rate,
        fit_kwargs=fit_kwargs,
        pool=pool,
        models_d=models_d, results_d=results_d,
        active_model=active_model if pool == "global_supervised" else None,
        init_weights_trained=init_weights_trained,
        warm_start=warm_start_val,
        seed=seed_val,
        
    )

    al_progress.to_csv( os.path.join(exp_dir, "al_progress.csv"), index=False)
    # Save upper bound based on warm-start mode
    upper_bound_path = Path(exp_dir) / "upper_bound_auc.npy"
    if not upper_bound_path.exists():
        if warm_start_val:
            # For warm-start AL, compute a from-scratch ceiling on full train data
            df_full = pd.concat(
                [df_tr_labeled_final, df_tr_unlabeled_final],
                axis=0,
            ) if df_tr_unlabeled_final is not None else df_tr_labeled_final

            if pool in ["personal", "global"]:
                Z_full_hr, Z_full_st = utility.encode(df_full, enc_hr, enc_st)
                Z_full = np.concatenate([Z_full_hr, Z_full_st], axis=1).astype("float32")
                y_full = df_full["state_val"].values.astype("float32")
                Z_val_hr, Z_val_st = utility.encode(df_val, enc_hr, enc_st)
                Z_val = np.concatenate([Z_val_hr, Z_val_st], axis=1).astype("float32")
                y_val = df_val["state_val"].values.astype("float32")
                Z_te_hr, Z_te_st = utility.encode(df_te, enc_hr, enc_st)
                Z_te = np.concatenate([Z_te_hr, Z_te_st], axis=1).astype("float32")
                y_te = df_te["state_val"].values.astype("float32")
                auc_m, auc_s, _, _, _, _, _, _ = utility.fit_and_eval(
                    fit_kwargs, utility.clf_builder, Z_full.shape[1],
                    Z_full, y_full, Z_val, y_val, Z_te, y_te,
                    CLF_PATIENCE, dropout_rate,
                    clf=None,
                )
                upper_bound_auc = auc_m
            else:
                Z_val, y_val = uq_utility._build_XY_from_windows(df_val)
                Z_te, y_te = uq_utility._build_XY_from_windows(df_te)
                _, _, ceiling_model = uq_utility.train_base_init_on_labeled(
                    df_full, Z_val, y_val, uq_utility.build_global_cnn_lstm, verbose=0
                )
                upper_bound_auc, _, _ = utility.bootstrap_auc(
                    y_te, ceiling_model.predict(Z_te, verbose=0).ravel()
                )
        else:
            # Retrain-from-scratch AL already matches the ceiling style
            upper_bound_auc = al_progress.iloc[-1]["AUC_Mean"]
        np.save(upper_bound_path, upper_bound_auc)
    
    # Use the model returned from run_al as the final model
   
    print("AL progress saved.")
    print("saved in  ", exp_dir)

    # Save final active model if available
    if active_model is not None:
        model_path = Path(models_d) / "al_final.keras"
        try:
            active_model.save(model_path)
            print("Saved final model to:", model_path)
        except Exception as e:
            print(f"Warning: failed to save final model: {e}")

    ### Save the queried indices to a file 
    print("queried_participants", count_df)
    with open( os.path.join(exp_dir, "queried_participant_counts_{}_{}.pkl".format(args.user, args.pool)), "wb") as f:
        pickle.dump(count_df, f)
    print("Saved queried participant counts.")
    # 

def main():
    assert(len(sys.argv) > 2)
    
    exp_dir = sys.argv[1]      # ignore later
    exp_name = sys.argv[2]     # ignore later
    exp_kwargs = json.loads(sys.argv[3])
    
    run(exp_dir, exp_name, exp_kwargs)  
if __name__ == '__main__':
    main()
    
    
