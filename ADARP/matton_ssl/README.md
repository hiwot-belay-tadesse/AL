# ADARP SimCLR with Matton et al. EDA augmentations

Self-contained re-run of the ADARP encoder pre-training where the **EDA channel**
uses the augmentation strategy of

> Matton, Lewis, Guttag, Picard. *Contrastive Learning of Electrodermal Activity
> Representations for Stress Detection.* CHIL 2023, PMLR 209:410-426.
> Code: https://github.com/kmatton/contrastive-learning-for-eda

No existing file in the repo is modified. The data loading, the window pool, the
session split, the encoder architecture, the projection head, the NT-Xent loss and
the HR augmentations are imported from `ADARP/preprocess_adarp_data.py` and
`src/signal_utils.py` as they are.

## Files

| File | Purpose |
|---|---|
| `eda_augmentations.py` | The paper's 17 transforms and the "All DAs" sampler (`MattonAugmenter`). Run it directly to draw an example figure. |
| `data4hz.py` | Reads the E4's native 4 Hz `EDA.csv` with the pipeline's own `read_signal`, caches it per participant, and cuts the labelled windows out of it by timestamp. |
| `train_encoders.py` | Trains the EDA encoder (4 Hz, 240-sample windows, Matton augs) and the HR encoder (1 Hz, 60-sample windows, pipeline augs) on the training sessions of the pipeline's split. |
| `score_one_class_svm.py` | One-class SVM on the new encoders, reusing the protocol functions of `ADARP/one_class_svm.py`; needed because the pipeline's 30-point scoring path cannot feed a 240-sample encoder. |
| `submit_train_encoders.sh` | SLURM job: train, then score. |
| `tests/` | Augmentation, window-bank and timestamp-slicing checks. |
| `encoders/seed_<seed>/...` | Output, in the same layout the pipeline writes. |

## What "the paper's strategy" means here

* Two views per window, each made by **one** transform sampled uniformly from
  the 17 (their pretraining config: `n_transforms=1`, `stochastic_choice=true`).
* Parameter ranges are the ones in that config. Where the paper text and the
  config disagree (time shift, cutout, loose sensor), the config wins because it
  is what their runs used.
* Transforms act on **raw microsiemens** at the E4's native **4 Hz**, in 60 s
  windows of 240 samples, exactly the paper's setting. Each view is z-scored per
  window afterwards.

## What differs from the existing ADARP encoder

* The pipeline averages EDA to 1 Hz and bins windows to 30 points before the
  encoder. Here the EDA encoder sees the raw 4 Hz signal, unfiltered, so the
  paper's parameter table applies with no rescaling. The HR encoder keeps the
  pipeline's 1 Hz export, in 60-sample windows covering the same seconds.
* Both views are augmented; the pipeline pairs the raw window with one augmented copy.
* The SSL pool is 60 s windows at a 30 s stride (the labelled pool's stride),
  not the paper's 0.25 s shift, which would be 24 million windows here.
* The time shift needs signal outside the window; 60 s of context either side is
  sliced from the session stream on the fly.
* The SSL validation loss is computed on 20% of training sessions held out by
  session, not on a random 20% of overlapping windows.

`encoders/seed_<seed>/encoder_config.json` records the exact settings of a run.

`eda_augmentations.py` still takes an `fs` argument and rescales the
above-Nyquist frequency ranges if you ever run it at a lower rate; at 4 Hz it is
the authors' config unchanged.

## Run

```bash
python -m pytest ADARP/matton_ssl/tests -q                     # checks
python ADARP/matton_ssl/eda_augmentations.py                    # example figure (4 Hz)
python ADARP/matton_ssl/train_encoders.py                       # global encoders, eda + hr
python ADARP/matton_ssl/train_encoders.py --pool personal       # one pair per participant
sbatch ADARP/matton_ssl/submit_train_encoders.sh                # on the cluster

# score the new encoders with the one-class SVM protocol
python ADARP/matton_ssl/score_one_class_svm.py --channels eda
python ADARP/matton_ssl/score_one_class_svm.py --channels hr_eda
```
