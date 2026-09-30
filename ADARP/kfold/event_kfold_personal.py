#!/usr/bin/env python
"""
Event-level k-fold CV for personal pool without data leakage.

Stratifies by event (button press) to ensure no event appears in both train and test.
Evaluates at window level to compute AUC.

Structure:
  ADARP/kfold/
    ├── encoders/         (saved SSL encoders per participant/fold)
    ├── results/          (AUC results per participant/fold)
    └── summary.csv       (final results table)
"""

import os
import sys
from pathlib import Path
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_ADARP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ADARP_DIR.parent
for _path in (str(_REPO_ROOT), str(_ADARP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import utility
from new_helper import (
    set_classifier, reset_seeds, build_classifier,
    predict_positive_scores, compute_class_weight
)
from preprocess_adarp_data import prepare_data, load_windows, windows_to_frame


# Configuration
GRID_SEARCH_DIR = _ADARP_DIR / "kfold"
ENCODERS_DIR = GRID_SEARCH_DIR / "encoders"
RESULTS_DIR = GRID_SEARCH_DIR / "results"
BATCH_SSL = 32
SSL_EPOCHS = 100


def auto_choose_k(n_events):
    """Choose k-fold based on number of events."""
    if n_events < 3:
        return None  # Can't do CV
    elif n_events < 5:
        return 2
    elif n_events < 10:
        return 3
    elif n_events < 20:
        return 4
    else:
        return 5


def get_events_per_participant():
    """Count stress events per participant."""
    print("Loading ADARP data...")
    try:
        eda, hr, meta = load_windows(_ADARP_DIR / "processed")
        frame = windows_to_frame(eda, hr, meta)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using pre-computed counts from notebook...")
        return {
            "101": 4, "102": 47, "104": 2, "105": 8,
            "106": 2, "107": 1, "108": 10, "109": 15,
            "110": 14, "111": 24, "112": 56
        }

    # Count unique stress events per participant
    stress_events = {}
    for pid in frame['user_id'].unique():
        user_frame = frame[frame['user_id'] == str(pid)]
        stress_frame = user_frame[user_frame['state_val'] == 1]
        n_events = stress_frame['event_id'].nunique()
        stress_events[str(pid)] = n_events

    return stress_events


def run_event_kfold_personal(participant_id, seed=42, pool="personal"):
    """Run k-fold CV at event level for one participant.

    Returns:
        dict with fold results and overall AUC ± std
    """

    set_classifier("lr")
    reset_seeds(seed)

    # Participant setup
    pid = str(participant_id)
    participant_dir = ENCODERS_DIR / f"participant_{pid}" / f"seed_{seed}"
    participant_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESULTS_DIR / f"participant_{pid}" / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data using standard pipeline
    args_ns = SimpleNamespace(
        user=pid,
        pool=pool,
        fruit="ADARP",
        scenario="stress",
        task="adarp",
        participant_id=pid,
        unlabeled_frac=0.0,
        dropout_rate=0.5,
        warm_start=1,
        results_subdir="results",
        input_df="raw",
        classifier="lr",
    )

    print(f"\nProcessing {pid}...")
    try:
        prep = prepare_data(
            args=args_ns,
            top_out=participant_dir,
            shared_enc_root=participant_dir / "_global_encoders",
            shared_cnn_root=participant_dir / "global_cnns",
            batch_ssl=BATCH_SSL,
            ssl_epochs=SSL_EPOCHS,
            pool=pool,
            task="adarp",
            input_df="raw",
            seed=seed,
            processed_dir=_ADARP_DIR / "processed",
        )
    except Exception as e:
        print(f"  ✗ prepare_data failed: {e}")
        return None

    if prep is None:
        print(f"  ✗ No data for {pid}")
        return None

    df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, *_ = prep

    # Use personal pool data (df_tr)
    if df_tr is None or len(df_tr) == 0:
        print(f"  ✗ Empty training data")
        return None

    # Get unique events
    events = df_tr['event_id'].unique()
    n_events = len(events)
    k = auto_choose_k(n_events)

    if k is None:
        print(f"  ✗ Too few events ({n_events}) for k-fold")
        return None

    print(f"  ✓ {n_events} stress events → {k}-fold CV")

    # Stratified k-fold at event level
    fold_results = []
    aucs = []

    # Create stratified folds on events (not windows)
    df_events = pd.DataFrame({'event_id': events})
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # Dummy y for stratification (all same class, just for balance)
    y_dummy = np.zeros(len(events))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_events, y_dummy)):
        print(f"    Fold {fold_idx + 1}/{k}...")

        train_events = events[train_idx]
        test_events = events[test_idx]

        # Split data by events
        df_fold_train = df_tr[df_tr['event_id'].isin(train_events)]
        df_fold_test = df_tr[df_tr['event_id'].isin(test_events)]

        if len(df_fold_test) == 0 or df_fold_test['state_val'].nunique() < 2:
            print(f"      Skipping: test fold has no positive examples")
            continue

        # Encode
        Z_train = utility.encode_single_df(df_fold_train, enc_hr, enc_st, pool)
        Z_test = utility.encode_single_df(df_fold_test, enc_hr, enc_st, pool)

        y_train = df_fold_train['state_val'].values.astype("float32")
        y_test = df_fold_test['state_val'].values.astype("float32")

        # Standardize
        scaler = StandardScaler().fit(Z_train)
        Z_train = scaler.transform(Z_train).astype("float32")
        Z_test = scaler.transform(Z_test).astype("float32")

        # Train LR with C=0.01 (from grid search)
        model, _ = build_classifier(Z_train.shape[1], patience=15, dropout_rate=0.5, seed=seed)

        # Apply LR hyperparameters
        if hasattr(model, 'set_params'):
            model.set_params(
                estimator__solver="lbfgs",
                estimator__max_iter=1000,
                estimator__C=0.01,
                estimator__tol=0.0001,
            )

        # Class weight
        classes = np.unique(y_train)
        cw_vals = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

        # Fit
        model.fit(Z_train, y_train, class_weight=class_weight)

        # Evaluate
        scores = predict_positive_scores(model, Z_test)
        auc = float(roc_auc_score(y_test, scores))
        aucs.append(auc)

        fold_results.append({
            'fold': fold_idx,
            'n_train': len(df_fold_train),
            'n_test': len(df_fold_test),
            'train_events': len(train_events),
            'test_events': len(test_events),
            'auc': auc,
        })

        print(f"      AUC: {auc:.4f} (test: {len(df_fold_test)} windows / {len(test_events)} events)")

    if not aucs:
        print(f"  ✗ No folds completed")
        return None

    # Summary
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=0)) if len(aucs) > 1 else 0.0

    result = {
        'participant': pid,
        'seed': seed,
        'pool': pool,
        'n_events': n_events,
        'k_folds': k,
        'folds': fold_results,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
    }

    # Save results
    with open(results_dir / "kfold_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  ✓ Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    return result


def plot_results(all_results):
    """Plot personal model results: test AUC ± std per participant."""
    import matplotlib.pyplot as plt

    if not all_results:
        return

    # Prepare data
    participants = [r['participant'] for r in all_results]
    test_aucs = [r['mean_auc'] for r in all_results]
    test_stds = [r['std_auc'] for r in all_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(participants))
    width = 0.6

    # Plot test AUC with error bars
    bars = ax.bar(x, test_aucs, width, label='Test AUC', color='#3498db',
                  alpha=0.8, edgecolor='black', linewidth=1.5,
                  yerr=test_stds, capsize=5, error_kw={'elinewidth': 2, 'ecolor': 'darkblue'})

    # Add value labels on bars
    for bar, auc, std in zip(bars, test_aucs, test_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{auc:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Reference line at 0.5 (chance)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2.5,
               label='Chance (0.5)', alpha=0.7)

    # Formatting
    ax.set_xlabel('Participant', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Personal Pool Models: Event-Level K-Fold CV Results',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    plt.tight_layout()

    # Save
    plot_path = GRID_SEARCH_DIR / "personal_models_auc.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    plt.close()


def main():
    """Run event-level k-fold CV for all viable participants."""

    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Event-Level K-Fold CV for Personal Pool (No Leakage)")
    print("="*80)

    # Get events per participant
    events_per_pid = get_events_per_participant()

    # Determine viable participants
    viable = {}
    for pid, n_events in events_per_pid.items():
        k = auto_choose_k(n_events)
        viable[pid] = (n_events, k)

    print("\nParticipant Viability:")
    print("-" * 50)
    for pid in sorted(viable.keys()):
        n_events, k = viable[pid]
        if k is None:
            print(f"  {pid}: {n_events} events → ✗ Not viable (<3 events)")
        else:
            print(f"  {pid}: {n_events} events → ✓ {k}-fold CV")

    # Run for viable participants
    all_results = []
    for pid, (n_events, k) in sorted(viable.items()):
        if k is None:
            continue

        result = run_event_kfold_personal(pid, seed=42, pool="personal")
        if result is not None:
            all_results.append(result)

    # Summary table
    if all_results:
        summary_rows = [
            {
                'participant': r['participant'],
                'events': r['n_events'],
                'k': r['k_folds'],
                'mean_auc': r['mean_auc'],
                'std_auc': r['std_auc'],
            }
            for r in all_results
        ]

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(GRID_SEARCH_DIR / "summary_event_kfold.csv", index=False)

        print("\n" + "="*80)
        print("SUMMARY: Event-Level K-Fold Results")
        print("="*80)
        print(df_summary.to_string(index=False))
        print(f"\n✓ Saved to: {GRID_SEARCH_DIR / 'summary_event_kfold.csv'}")
        print(f"✓ Full results in: {RESULTS_DIR}")
        print(f"✓ Encoders in: {ENCODERS_DIR}")

        # Generate plot
        plot_results(all_results)


if __name__ == "__main__":
    main()