import numpy as np
import pandas as pd

import utility, uq_utility

import os, sys 
from preprocess import prepare_data
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

def pick_random(K, df_tr_unlabeled):
    """Return K random indices from the unlabeled pool."""
    if len(df_tr_unlabeled) < K:
        K = len(df_tr_unlabeled)
    import random
    queried_indices = random.sample(list(df_tr_unlabeled.index), K)
    df_queried = df_tr_unlabeled.loc[queried_indices]
    return queried_indices, df_queried
    
def train_single_model(seed, Z_tr_labeled, y_tr_labeled,
                       Z_val, y_val,
                       Z_te, y_te,
                       clf_builder,
                       CLF_PATIENCE,
                       dropout_rate,
                       fit_kwargs):   
    import random, numpy as np, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    input_dim = Z_tr_labeled.shape[1] 
    # fit_kwargs should already have callbacks set up, but we need to replace
    # the EarlyStopping callback with a new one for this specific model
    fit_kwargs_callbacks = fit_kwargs.copy() if fit_kwargs else {}
    
    model, es  = clf_builder(input_dim, CLF_PATIENCE, dropout_rate)
    
    # Ensure shuffle is set
    if 'shuffle' not in fit_kwargs_callbacks:
        fit_kwargs_callbacks['shuffle'] = False
    
    # Replace EarlyStopping callback with new one for this model
    # (each model needs its own EarlyStopping instance)
    # Preserve other callbacks if any exist
    if 'callbacks' in fit_kwargs_callbacks:
        callbacks = fit_kwargs_callbacks['callbacks']
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        # Remove any existing EarlyStopping callbacks and add new one
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks = [cb for cb in callbacks if not isinstance(cb, EarlyStopping)]
        callbacks.append(es)
        fit_kwargs_callbacks['callbacks'] = callbacks
    else:
        fit_kwargs_callbacks['callbacks'] = [es]
    
    # Compute class weights for consistency
    unique_classes = np.unique(y_tr_labeled)
    cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_tr_labeled)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    
    history = model.fit(Z_tr_labeled, 
                      y_tr_labeled,
                      class_weight=class_weight,
                      **fit_kwargs_callbacks)
    
    return model, history

def train_ensemble_models( M, 
    Z_tr_labeled, y_tr_labeled,
                          Z_val, y_val,
                          Z_te, y_te,
                          clf_builder,
                          CLF_PATIENCE,
                          dropout_rate,
                          fit_kwargs_with_callbacks):   
    input_dim = Z_tr_labeled.shape[1] 
    
    print(f"\n=== Training Ensemble of {M} Models ===")
    models = []
    histories = []
    for i in range(M):
        print(f"Training model {i+1}/{M} (seed={42+i})...")
        model, history = train_single_model(
            seed=42 + i,
            Z_tr_labeled=Z_tr_labeled, y_tr_labeled=y_tr_labeled,
            Z_val=Z_val, y_val=y_val,
            Z_te=Z_te, y_te=y_te,
            clf_builder=clf_builder,
            CLF_PATIENCE=CLF_PATIENCE,
            dropout_rate=dropout_rate,
            fit_kwargs=fit_kwargs_with_callbacks
        )
        models.append(model)
        histories.append(history)
        print(f"  ✓ Finished training model {i+1}/{M}")
    
    print(f"=== Ensemble Training Complete: {M} models trained ===\n")
    return models, histories

def ensemble_epistemic_uncertainty(models, Z_unlabeled):
    preds = np.stack([m.predict(Z_unlabeled, verbose=0).ravel() for m in models])
    mean_pred = preds.mean(axis=0)
    var_pred  = preds.var(axis=0)        # THIS IS EPISTEMIC UNCERTAINTY
    return mean_pred, var_pred

def acquire_uncertain_samples(models, Z_unlabeled, df_unlabeled, K):
    """
    models: list of trained models in the ensemble
    Z_unlabeled: np.array representation of unlabeled samples
    df_unlabeled: corresponding dataframe rows
    K: number of samples to query
    """

    preds = np.stack([m.predict(Z_unlabeled, verbose=0).ravel() for m in models])
    var_pred = preds.var(axis=0)

    # top-K uncertain points (positional indices)
    queried_positions = np.argsort(var_pred)[-K:]

    # Get DataFrame rows using positional indexing
    df_queried = df_unlabeled.iloc[queried_positions]
    
    # Return DataFrame index labels (not positional indices) for proper DataFrame operations
    queried_indices = df_queried.index.tolist()

    return queried_indices, df_queried, var_pred

def run_al(Aq, df_tr_labeled, df_tr_unlabeled, df_val, df_te, enc_hr, enc_st, clf_builder, K, budget, CLF_PATIENCE, dropout_rate, fit_kwargs):
    '''Active learning loop'''
    ## unpack from fit_kwargs
    
    print(f"\n=== run_al INPUT DEBUG ===")
    print(f"df_tr_labeled - Positive: {(df_tr_labeled['state_val'] == 1).sum()}, Negative: {(df_tr_labeled['state_val'] == 0).sum()}")
    print(f"df_val - Positive: {(df_val['state_val'] == 1).sum()}, Negative: {(df_val['state_val'] == 0).sum()}")
    print(f"df_te - Positive: {(df_te['state_val'] == 1).sum()}, Negative: {(df_te['state_val'] == 0).sum()}")

    
    H_tr_labeled,  S_tr_labeled  = utility.encode(df_tr_labeled, enc_hr, enc_st)
    H_tr_unlabeled, S_tr_unlabeled = utility.encode(df_tr_unlabeled, enc_hr, enc_st)

    H_val, S_val = utility.encode(df_val, enc_hr, enc_st)
    H_te,  S_te  = utility.encode(df_te, enc_hr, enc_st)

    ##utility.encoded representations
    Z_tr_labeled   = np.concatenate([H_tr_labeled,  S_tr_labeled],  axis=1).astype('float32')
    y_tr_labeled   = df_tr_labeled['state_val'].values.astype('float32')
    Z_tr_unlabeled = np.concatenate([H_tr_unlabeled, S_tr_unlabeled],axis=1).astype('float32')
    y_tr_unlabeled = df_tr_unlabeled['state_val'].values.astype('float32')

    
    # Fixed validation and test sets (do NOT change across rounds)
    Z_val  = np.concatenate([H_val, S_val], axis=1).astype('float32')
    y_val  = df_val['state_val'].values.astype('float32')
    Z_te   = np.concatenate([H_te,  S_te],  axis=1).astype('float32')
    y_te   = df_te['state_val'].values.astype('float32')

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
    
    # Compute class weights for consistency with utility.fit_and_eval and run.py
    unique_classes = np.unique(y_tr_labeled)
    cw_vals = compute_class_weight('balanced', classes=unique_classes, y=y_tr_labeled)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    
    history = clf.fit(Z_tr_labeled, 
                      y_tr_labeled,
                      class_weight=class_weight,
                      **fit_kwargs_with_callbacks)
    # breakpoint()
    auc_m_pre, auc_s_pre = utility.bootstrap_auc(y_te, clf.predict(Z_te, verbose=0).ravel())

    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    
    results = []
    
    # metrics_0 = df_boot_pre.copy()
    metrics_0 = pd.DataFrame(index=[0])
    metrics_0["round"] = 0
    metrics_0["AUC_Mean"] = auc_m_pre
    metrics_0["AUC_STD"] = auc_s_pre
    results.append(metrics_0)

    queried_all = []

    # 4. Active learning loop
    for b in range(1, budget + 1):
        if len(df_tr_unlabeled) == 0:
            break

        k_actual = min(K, len(df_tr_unlabeled))
        print(f"AL round {b}/{budget}  (labeled={len(df_tr_labeled)}, unlabeled={len(df_tr_unlabeled)}, K={k_actual})")

        # utility.encode current unlabeled pool
        H_tr_unlabeled, S_tr_unlabeled = utility.encode(df_tr_unlabeled, enc_hr, enc_st)
        Z_tr_unlabeled = np.concatenate([H_tr_unlabeled, S_tr_unlabeled], axis=1).astype("float32")
        # 
        # Acquisition: random or most‑uncertain K windows
        if Aq == "random":
            queried_indices, df_queried = pick_random(k_actual, df_tr_unlabeled)
        elif Aq == "uncertainty":
            models, histories = train_ensemble_models(
                M=5,
                Z_tr_labeled=Z_tr_labeled, y_tr_labeled=y_tr_labeled,
                Z_val=Z_val, y_val=y_val,
                Z_te=Z_te, y_te=y_te,
                clf_builder=clf_builder,
                CLF_PATIENCE=CLF_PATIENCE,
                dropout_rate=dropout_rate,
                fit_kwargs_with_callbacks=fit_kwargs_with_callbacks,
            )
            queried_indices, df_queried, var_pred = acquire_uncertain_samples(
                models=models,
                Z_unlabeled=Z_tr_unlabeled,
                df_unlabeled=df_tr_unlabeled,
                K=k_actual
            )
        else:
            raise ValueError(f"Unknown acquisition function: {Aq}. Must be 'random' or 'uncertainty'.")
        queried_all.extend(queried_indices)

        # Move queried from UNLABELED → LABELED

        # df_queried['state_val'] = df_queried['true_label']  # restore true labels

        df_tr_labeled = pd.concat([df_tr_labeled, df_queried], axis=0)
        df_tr_unlabeled = df_tr_unlabeled.drop(index=queried_indices)

        # Re‑utility.encode labeled set
        H_tr_labeled, S_tr_labeled = utility.encode(df_tr_labeled, enc_hr, enc_st)
        Z_tr_labeled = np.concatenate([H_tr_labeled, S_tr_labeled], axis=1).astype("float32")
        y_tr_labeled = df_tr_labeled["state_val"].values.astype("float32")

        

        # Retrain / fine‑tune classifier on expanded labeled pool
        # df_boot_post, auc_m_post, auc_s_post, clf = utility.fit_and_eval(
        #     fit_kwargs, clf_builder, input_dim,
        #     Z_tr_labeled, y_tr_labeled,
        #     Z_val, y_val, Z_te, y_te,
        #     CLF_PATIENCE, dropout_rate,
        #     clf=clf,
        # )
        
        auc_m_post, auc_s_post, clf = utility.fit_and_eval(
            fit_kwargs, clf_builder, input_dim,
            Z_tr_labeled, y_tr_labeled,
            Z_val, y_val, Z_te, y_te,
            CLF_PATIENCE, dropout_rate,
            clf=clf,
        )
        
        # metrics = df_boot_post.copy()
        metrics = pd.DataFrame(index=[0]) 
        metrics["round"] = b
        metrics["AUC_Mean"] = auc_m_post
        metrics["AUC_STD"] = auc_s_post
        
        results.append(metrics) 
            ## model on full train set to get upper bound AUC
        ### auc for full train set



        print(f"  → Round {b}: AUC={auc_m_post:.3f} ± {auc_s_post:.3f}")

    al_progress = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return al_progress, clf, df_tr_labeled, df_tr_unlabeled, queried_all


def main():
    import argparse
    
    pa = argparse.ArgumentParser()
    pa.add_argument('--user', type=str, required=True)
    pa.add_argument('--fruit', type=str, required=True)
    pa.add_argument('--scenario', type=str, required=True)
    # pa.add_argument('--output_dir', type=str, required=True)
    pa.add_argument("--results_subdir", default="results")
    pa.add_argument('--unlabeled_frac', type=float, default=0.8)
    args = pa.parse_args()
    fruit = args.fruit
    scenario = args.scenario
    unlabeled_frac = args.unlabeled_frac    
    
    # Top‑level paths
    RESULTS_SUBDIR = args.results_subdir
    top_out = Path(RESULTS_SUBDIR)
    shared_enc_root = top_out / "_global_encoders"
    pool ="global"
    BATCH_SSL, SSL_EPOCHS = 32, 100
    CLF_EPOCHS, CLF_PATIENCE = 500, 20
    
    budget = 10 
    K = 10 
    Aq = "uncertainty"  # "random" or "uncertainty"
    dropout_rate = 0.1
    

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
    
    df_tr = df_tr.reset_index(drop=True)
    df_tr['hr_std'] = df_tr['hr_raw'].apply(np.std)
    df_tr['st_std'] = df_tr['st_raw'].apply(np.std)
    df_tr['combined_std'] = df_tr[['hr_std', 'st_std']].min(axis=1)
    df_tr = df_tr[df_tr['combined_std'] > 0].reset_index(drop=True)

    Z_val_hr,  Z_val_st  = utility.encode(df_val, enc_hr, enc_st)
    Z_val = np.concatenate([Z_val_hr,  Z_val_st],  axis=1).astype('float32')
    y_val = df_val["state_val"].values.astype("float32")
    Z_te_hr,  Z_te_st  = utility.encode(df_te, enc_hr, enc_st)
    Z_te = np.concatenate([Z_te_hr,  Z_te_st],  axis=1).astype('float32')
    y_te = df_te["state_val"].values.astype("float32")      
    
    # Drop temporary columns
    # df_tr = df_tr.drop(columns=['hr_std', 'st_std', 'combined_std'])
    from sklearn.model_selection import train_test_split
    df_tr_labeled, df_tr_unlabeled = train_test_split(df_tr, test_size=unlabeled_frac, stratify=df_tr["state_val"], random_state=41)

    Z_tr_labeled_hr,  Z_tr_labeled_st  = utility.encode(df_tr_labeled, enc_hr, enc_st)
    Z_tr_labeled = np.concatenate([Z_tr_labeled_hr,  Z_tr_labeled_st],  axis=1).astype('float32')
    y_tr_labeled = df_tr_labeled["state_val"].values.astype("float32")  
    Z_tr_unlabeled_hr,  Z_tr_unlabeled_st  = utility.encode(df_tr_unlabeled, enc_hr, enc_st)
    Z_tr_unlabeled = np.concatenate([Z_tr_unlabeled_hr,  Z_tr_unlabeled_st],  axis=1).astype('float32')
    y_tr_unlabeled = df_tr_unlabeled["state_val"].values.astype("float32")      
    
    fit_kwargs = dict(
        epochs=CLF_EPOCHS,
        batch_size=16,
        verbose=0,
        validation_data=(Z_val, y_val),  
    )
    al_progress, clf, df_tr_labeled, df_tr_unlabeled, queried_all = run_al(Aq, df_tr_labeled, df_tr_unlabeled, df_val, df_te, enc_hr, enc_st, uq_utility.clf_builder, K, budget, CLF_PATIENCE, dropout_rate, fit_kwargs)    
    
    # ---- Save AL results ----
    al_progress.to_csv(top_out / "al_progress.csv", index=False)
    
    
if __name__ == "__main__":
    main()
    