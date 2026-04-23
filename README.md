# Active Learning Pipeline for Personalized Craving/Use Prediction

This repository contains code for active learning experiments on:
- Banaware fruit craving/use tasks
- Cardiomate BP spike prediction

## Run Active Learning

```bash
make run_all
```

For Cardiomate runs:

```bash
make run_bp_submit
```

## Aggregate AUC Results

Use [`scripts/aggregate_comparison.py`](/Users/hiwotbelaytadesse/Desktop/Banaware_AL/scripts/aggregate_comparison.py) to build a single CSV from all `comparison_summary.csv` files.

Cardiomate example:

```bash
python scripts/aggregate_comparison.py \
  --root BP_SPIKE_PRED \
  --mode new \
  --out BP_SPIKE_PRED/user_pipeline_auc_wide.csv
```

Banaware example:

```bash
python scripts/aggregate_comparison.py \
  --root BANAWARE_PRED \
  --mode new \
  --out BANAWARE_PRED/user_pipeline_auc_wide.csv
```

## Plot Aggregated AUC

1. Train-AUC distribution plot (boxplot):
   script: [`scripts/plot_train_auc.py`](/Users/hiwotbelaytadesse/Desktop/Banaware_AL/scripts/plot_train_auc.py)

Cardiomate command:

```bash
python scripts/plot_train_auc.py
```

This reads:
- `BP_SPIKE_PRED/user_pipeline_auc_wide.csv`

and writes:
- `supp_plots/train_auc_boxplot.png`

2. Aggregated global-supervised ROC/AUC plot:
   script: [`scripts/plot_global_auc.py`](/Users/hiwotbelaytadesse/Desktop/Banaware_AL/scripts/plot_global_auc.py)

```bash
python scripts/plot_global_auc.py --data cardiomate
```

Output:
- `supp_plots/global_supervised_roc_curve_aggregated.png`

## Move All Global-Supervised Loss Curves Into One Folder

Use [`scripts/scratch.py`](/Users/hiwotbelaytadesse/Desktop/Banaware_AL/scripts/scratch.py).

Banaware:

```bash
python scripts/scratch.py --data banaware
```

Cardiomate:

```bash
python scripts/scratch.py --data cardiomate
```

Preview only (no file moves):

```bash
python scripts/scratch.py --data banaware --dry-run
```

Default destination folders:
- `BANAWARE_PRED/loss_curves/global_supervised`
- `BP_SPIKE_PRED/loss_curves/global_supervised`
