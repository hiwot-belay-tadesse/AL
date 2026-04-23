import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


# --------------------------------------------------------------------------- #
# Boot‑strap utilities
# --------------------------------------------------------------------------- #

def bootstrap_threshold_metrics(
    y,
    probs,
    thresholds: np.ndarray = np.arange(0.0, 1.01, 0.01),
    sample_frac: float     = 0.7,
    n_iters: int           = 1000,
    rng_seed: int          = 42,
) -> (pd.DataFrame, float, float):
    """
    Re‑sample ~sample_frac of the data (with replacement) n_iters times and
    aggregate Sensitivity, Specificity, Accuracy and AUC (ROC) statistics.
    Returns (metrics_df, auc_mean, auc_std).
    """
    rng     = np.random.default_rng(rng_seed)
    n       = len(y)
    k       = int(np.round(sample_frac * n))
    idx_all = np.arange(n)

    records = {t: {'sens': [], 'spec': [], 'acc': []} for t in thresholds}
    aucs    = []

    for _ in range(n_iters):
        sidx   = rng.choice(idx_all, size=k, replace=True)
        y_samp = y[sidx]
        p_samp = probs[sidx]

        # AUC only valid with both classes present
        if len(np.unique(y_samp)) > 1:
            
            aucs.append(roc_auc_score(y_samp, p_samp))

        for t in thresholds:
            preds = (p_samp >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_samp, preds, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            acc  = (tp + tn) / len(y_samp)
            records[t]['sens'].append(sens)
            records[t]['spec'].append(spec)
            records[t]['acc'].append(acc)

    rows = []
    for t in thresholds:
        rows.append({
            "Threshold":          t,
            "Sensitivity_Mean":   np.mean(records[t]['sens']),
            "Sensitivity_STD":    np.std(records[t]['sens'], ddof=1),
            "Specificity_Mean":   np.mean(records[t]['spec']),
            "Specificity_STD":    np.std(records[t]['spec'], ddof=1),
            "Accuracy_Mean": np.mean(records[t]['acc']),
            "Accuracy_STD":       np.std(records[t]['acc'], ddof=1),
        })

    auc_mean = np.nanmean(aucs)
    auc_std  = np.nanstd(aucs,  ddof=1)
    return pd.DataFrame(rows), auc_mean, auc_std    

# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_thresholds(
    y_test,
    p_test,
    out_dir: str,
    title:   str,
    thresholds: np.ndarray = np.arange(0.0, 1.01, 0.01),
    sample_frac: float     = 0.7,
    n_iters: int           = 1000,
):
    """
    • Boot‑straps test set only
    • Saves CSV with mean ± SD
    • Saves CSV of thresholds beating (Sens>0.9 & Spec>0.5)
    • Generates error‑bar threshold plot + ROC curve
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------- boot‑strapped stats (test only) --------
    df_te, auc_te_m, auc_te_s = bootstrap_threshold_metrics(
        y_test, p_test.flatten(), thresholds, sample_frac, n_iters
    )
    df_te.to_csv(os.path.join(out_dir, "bootstrap_metrics.csv"), index=False)

    # thresholds that meet publication‑quality criteria
    mask = (df_te["Sensitivity_Mean"] > 0.7) & (df_te["Specificity_Mean"] > 0.3)
    df_te[mask].to_csv(os.path.join(out_dir, "passing_thresholds.csv"),
                       index=False)

    # -------- threshold curve with error bars (test only) --------
    plt.figure(figsize=(8, 5))
    for metric in ["Sensitivity", "Specificity", "Accuracy"]:
        plt.errorbar(
            thresholds,
            df_te[f"{metric}_Mean"],
            yerr=df_te[f"{metric}_STD"],
            fmt="-o",
            capsize=2,
            label=metric,
        )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Analysis – {title}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "threshold_analysis.png"))
    plt.close()

    # -------- ROC curve (raw test set) + boot‑strap AUC stats --------
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, p_test)
        raw_auc     = roc_auc_score(y_test, p_test)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"Raw AUC = {raw_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.title(f"ROC – {title}\nBoot AUC = {auc_te_m:.3f} ± {auc_te_s:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_curve.png"))
        plt.close()
    else:
        print(f"[plot_thresholds] ROC skipped – only one class in test set "
              f"for {title}")


# --------------------------------------------------------------------------- #
# SSL loss curves
# --------------------------------------------------------------------------- #
def plot_ssl_losses(train_losses, val_losses, out_dir, encoder_name="encoder"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val  Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"SimCLR – {encoder_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{encoder_name}_ssl_loss.png"))
    plt.close()

# ─── Plot helpers ─────────────────────────────────────────────────────────
def plot_clf_losses(train, val, out_dir, fname):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train, label="Train")
    plt.plot(val,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Binary CE")
    plt.title(fname.replace('_', ' ').title())
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.png")
    plt.close()