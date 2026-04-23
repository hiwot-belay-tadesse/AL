


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

pa = argparse.ArgumentParser()
pa.add_argument("--user", default= "ID21" )
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

OUTPUT_DIR = "TEST_AL/global_SSL"

unlabeled_frac = [float(args.unlabeled_frac)]
# unlabeled_frac = [0.7]

dropout_rate = [float(args.dropout_rate)]


T = 30
K = 40
CLF_EPOCHS, CLF_PATIENCE = 500, 20
GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 15
BATCH_SSL, SSL_EPOCHS = 32, 100


# Top‑level paths
# Use OUTPUT_DIR to match submit_batch.py
top_out = Path(OUTPUT_DIR) 
shared_enc_root = top_out / "_global_encoders"
shared_cnn_root = top_out / "global_cnns"

 
user = args.user
pool = args.pool
fruit = args.fruit
scenario = args.scenario
dropout_rate = float(args.dropout_rate)
unlabeled_frac = float(args.unlabeled_frac)

prep = prepare_data(
args=args,
top_out=top_out,
shared_enc_root=shared_enc_root,
shared_cnn_root=shared_cnn_root,
batch_ssl=BATCH_SSL,
ssl_epochs=SSL_EPOCHS, 
pool=pool,
)

df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d, all_negatives = prep
breakpoint()
  # Compute budget dynamically from dataset size.

if pool == "personal":
    N = len(df_tr)
elif pool == "global" or pool == "global_supervised":
    N = len(df_all_tr)
else:
    print("Error: unknown pool type:", pool)

k_val = K

uf_val = unlabeled_frac

budget = int(np.ceil(uf_val * N / k_val))

# budget = max(1, budget)


t_val = T
if isinstance(t_val, (list, tuple)):
    t_val = t_val if t_val else None
dr_val = dropout_rate
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

exp_dir_path = Path(OUTPUT_DIR)
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
    "T":T,
    "K": k_val,
    "Budget": budget,
}
summary_path = os.path.join(exp_dir, "exp_summary.json")

random_al_results = {}
uncertainty_results = {}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print("Saved exp_summary.json to:", summary_path)

if pool == "global" or "global_supervised":
    df_tr_labeled, df_tr_unlabeled = train_test_split(df_all_tr, test_size=unlabeled_frac, stratify=df_all_tr["state_val"], random_state=48)
elif pool == "personal":
    df_tr_labeled, df_tr_unlabeled = train_test_split(df_tr, test_size=unlabeled_frac, stratify=df_tr["state_val"], random_state=48)


## for finetuning
if pool == "global_supervised":

    init_weights_trained, base_hist, active_model = uq_utility.train_base_init_on_labeled(df_tr_labeled, Z_val, y_val, uq_utility.build_global_cnn_lstm)
    
else:
    init_weights_trained=None
    
fit_kwargs = dict(
epochs=CLF_EPOCHS,
batch_size=32,
verbose=0,
)
if pool in ["global", "personal"]:
    H_val, S_val = utility.encode(df_val, enc_hr, enc_st)
    Z_val = np.concatenate([H_val, S_val], axis=1).astype('float32')
    y_val = df_val['state_val'].values.astype('float32')
else:
    Z_val, y_val = uq_utility._build_XY_from_windows(df_val)

fit_kwargs["validation_data"] = (Z_val, y_val)



if len(Z_val) > 0:
    fit_kwargs["validation_data"] = (Z_val, y_val)

fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}

    
seed_val = 42
exp_names = ["random", "uncertainty"]
for exp_name in exp_names:
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
        K,
        budget,
        T if exp_name!='random' else None,
        CLF_PATIENCE=CLF_PATIENCE, 
        dropout_rate=dropout_rate,
        fit_kwargs=fit_kwargs,
        pool=pool,
        models_d=models_d, results_d=results_d,
        active_model=active_model if pool == "global_supervised" else None,
        init_weights_trained=init_weights_trained,
        warm_start=0,
        seed=seed_val,
        
    )
    breakpoint()
    from matplotlib import pyplot as plt

    al_progress.to_csv( os.path.join(exp_dir, f"{exp_name}_al_progress.csv"), index=False)
    print(f"unlabeled pool is {len(df_tr_unlabeled)} for {exp_name}")

breakpoint() 
