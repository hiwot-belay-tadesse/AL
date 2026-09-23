32# ADARP Dataset Validation + Experiment Guide

## Objective

Determine whether the **ADARP dataset** is suitable as a second wearable
dataset for the existing active-learning study.

The intended downstream experiment should match the current BP-spike
study as closely as is scientifically reasonable:

1.  Learn representations from wearable biosignals with a
    **self-supervised encoder**.
2.  Freeze the encoder.
3.  Train a downstream **binary classifier** on the learned
    representations.
4.  Simulate pool-based active learning with **coreset vs. random
    acquisition**.
5.  Analyze **which participants the acquired labeled windows come
    from** and whether acquisition concentrates on a subset of
    participants.

Do **not** redesign the research question. The immediate goal is to
validate ADARP, reproduce its label construction carefully, run the
analogous experiment, and generate diagnostics/plots that reveal whether
the dataset is appropriate.

------------------------------------------------------------------------

## 1. Dataset structure already established

### Participants

The sensor-data folders correspond to participants:

``` python
PARTICIPANTS = [
    "101C", "102C", "104C", "105C", "106C",
    "107C", "108C", "109C", "110C", "111C", "112C"
]
```

Folders are named:

``` text
Part 101C
Part 102C
...
Part 112C
```

Each participant folder contains multiple E4 recording
subfolders/sessions.

### Sensor files

Each recording subfolder can contain Empatica E4 files including:

-   `EDA.csv` --- EDA, typically 4 Hz
-   `HR.csv` --- derived heart rate, typically 1 Hz
-   `BVP.csv`
-   `ACC.csv`
-   `TEMP.csv`
-   `IBI.csv`
-   `tags.csv`

For standard signal CSVs:

-   row 0 = Unix start timestamp in UTC
-   row 1 = sampling rate in Hz
-   remaining rows = signal values

`tags.csv` contains Unix UTC timestamps of physical E4 button presses.

For the first experiment, use **EDA + HR only**.

------------------------------------------------------------------------

## 2. Load EDA and HR without downsampling

Preserve each modality at its native rate.

Do **not** forward-fill HR to the EDA rate and do not downsample EDA to
1 Hz merely to make shapes match.

A loader can follow:

``` python
def read_signal(path, column):
    raw = pd.read_csv(path, header=None).iloc[:, 0].astype(float)
    start, rate, values = raw.iloc[0], raw.iloc[1], raw.iloc[2:].to_numpy()

    index = pd.to_datetime(
        start + np.arange(len(values)) / rate,
        unit="s",
        utc=True
    )

    return pd.Series(values, index=index, name=column)
```

When combining signals for inspection, an outer timestamp join is
acceptable:

``` python
session_df = pd.concat([eda, hr], axis=1, join="outer")
```

The HR NaNs at EDA-only timestamps are **structural**, not missing
measurements.

HR often begins approximately 10 seconds after EDA because it is derived
from BVP. For multimodal windows, use only time intervals where both
modalities have actual coverage.

Never create a window across two recording sessions.

------------------------------------------------------------------------

## 3. Human stress labels: use E4 button tags first

The original ADARP processing is centered on participant-generated E4
button presses.

Treat a button press as the **human annotation anchor**.

Load every `tags.csv` for each participant and retain:

``` text
participant
session
tag_id
tag_time
```

Use all tags when constructing exclusion regions, even tags that later
fail the positive-segment coverage requirement.

The raw dataset contains **409 button tags** across the 11 participants.

------------------------------------------------------------------------

## 4. Reproduce ADARP positive/stress construction

For each tag at time `t`, the ADARP code uses a **40-minute segment
centered on the tag**:

``` text
t - 20 min ---------------- t ---------------- t + 20 min
                              ^
                         button press
```

Require complete EDA + HR coverage for the full +/-20-minute segment.

We already obtained the following usable-tag counts under this rule:

  Participant     Total tags   Usable +/-20 min tags
  ------------- ------------ -----------------------
  101C                    19                       9
  102C                    86                      61
  104C                    22                       4
  105C                    32                      18
  106C                    28                      14
  107C                    22                       4
  108C                    24                      14
  109C                    40                      23
  110C                    26                      17
  111C                    49                      30
  112C                    61                      59
  **Total**          **409**                 **253**

Note: a later event-validation analysis may yield a smaller set of
validated stress events. Keep the exact filtering rule explicit in every
result. Do not silently mix "all usable tags" with a stricter
validated-event subset.

------------------------------------------------------------------------

## 5. Reproduce ADARP non-stress construction

Non-stress is **not a separate button response**.

Use the ADARP rule that candidate non-stress physiology must lie outside
a **+/-60-minute exclusion buffer around every stress tag**.

Conceptually:

``` text
non-stress        excluded/uncertain         non-stress
---------|-----------------------------------|---------
       t-60m               tag             t+60m
```

The +/-20-minute positive segment sits inside this larger +/-60-minute
exclusion zone.

Important:

-   Use **all button tags** for the +/-60-minute exclusion rule.
-   Extract non-stress intervals separately within each recording
    session.
-   Do not bridge gaps between sessions.
-   Non-stress segments can have variable duration.
-   Keep a unique `segment_id` for every extracted non-stress segment.

------------------------------------------------------------------------

## 6. Window construction

Follow the original ADARP windowing baseline:

``` python
WINDOW_SEC = 60
STEP_SEC = 30
```

Thus:

-   window length = 60 seconds
-   stride = 30 seconds
-   overlap = 50%

Window each stress/non-stress segment **independently**.

### Stress windows

Each valid stress event produces one fixed 40-minute segment.

A 40-minute segment produces:

``` text
floor((2400 - 60) / 30) + 1 = 79 windows
```

Do not let windows cross from one event segment into another.

### Non-stress windows

For a non-stress segment with duration `D` seconds:

``` text
floor((D - 60) / 30) + 1
```

when `D >= 60`.

Do not infer non-stress window counts from the number of non-stress
segments because their durations vary.

------------------------------------------------------------------------

## 7. Preserve provenance

Every final window must retain enough metadata to map it back to its
source.

Recommended metadata:

``` text
participant
session
segment_type        # stress / nonstress
segment_id
event_id            # originating tag for stress; NA for nonstress
window_id
window_start
window_end
label                # 1 stress, 0 nonstress
```

For stress examples:

``` text
tag_001
  -> 40-minute segment
  -> 79 overlapping windows
```

These are **79 labeled model examples derived from one human
annotation**.

For the primary active-learning analysis, the key quantity is the
**number/share of acquired labeled windows contributed by each
participant**. Preserve `event_id` so event-level sensitivity analyses
remain possible.

Use careful terminology:

-   "acquired labeled windows/examples" for AL query counts
-   "participant contribution" for which participant those queried
    windows came from
-   do not call 79 derived windows "79 separate button presses"

------------------------------------------------------------------------

## 8. First validation table: window counts

Before training any model, generate a table with:

``` text
participant
total_tags
usable_stress_events
stress_windows
nonstress_segments
nonstress_windows
total_windows
stress_fraction
```

We have already observed that the resulting class balance can be highly
participant-dependent.

Example already observed for 101C under one event-filtering version:

``` text
stress windows:      316
non-stress windows:  12,646
```

Do not assume this is the final 101C count if the validated-event filter
changes. Recompute all counts from the exact chosen rule.

### Required plot A: per-participant event counts

Grouped bar plot:

``` text
x = participant
bars = stress-event count, non-stress-segment count
```

### Required plot B: per-participant window counts

Grouped bar plot:

``` text
x = participant
bars = stress windows, non-stress windows
```

Because counts differ greatly, also produce either:

-   a log-scale version, or
-   a second plot of stress fraction per participant.

### Required plot C: class balance

Plot:

``` text
stress_fraction = stress_windows / total_windows
```

for every participant.

This is important for interpreting AL acquisition behavior.

------------------------------------------------------------------------

## 9. Signal/tag diagnostic plots

Before modeling, visually verify the tag-centered construction.

For several participants and several usable tags, create aligned
two-panel plots:

### Panel 1

EDA at native sampling rate.

### Panel 2

HR at native sampling rate.

Overlay:

-   vertical line at the button press
-   shaded region from `tag - 20 min` to `tag + 20 min`

Only plot tags with complete +/-20-minute multimodal coverage when
validating the stress-segment construction.

Also inspect a few rejected tags to verify that rejection is due to
insufficient recording coverage rather than a bug.

### Required plot D: representative valid tag examples

At minimum show:

-   one participant with few usable tags
-   one participant with many usable tags
-   several events with visibly different physiology

------------------------------------------------------------------------

## 10. SSL dataset construction

The SSL encoder should operate on the fixed window pool.

Use **all eligible windows** for representation learning according to
the train/test split rules.

Do not use labels in the SSL loss.

Because EDA and HR have different native sampling rates, use
**modality-specific encoders** rather than manufacturing a common
sampling rate.

For each 60-second window:

``` text
EDA @ 4 Hz -> approximately 240 samples -> EDA encoder
HR  @ 1 Hz -> approximately 60 samples  -> HR encoder
```

Concatenate the modality representations:

``` text
EDA encoder ----\
                 -> fused representation z
HR encoder -----/
```

Match the BP-spike architecture as closely as practical:

-   contrastive/self-supervised pretraining
-   separate modality encoders
-   concatenate embeddings
-   frozen encoder during downstream AL
-   logistic regression classifier on representations

The BP experiment used a 32-dimensional representation. Preserve that
dimensionality unless there is an implementation reason not to.

------------------------------------------------------------------------

## 11. Avoid leakage

This is critical because ADARP creates many overlapping windows from the
same human event.

Never randomly split overlapping windows from the same event across
train and test.

Prefer a **participant-level split** when evaluating a global model, or
at minimum ensure all windows derived from the same `event_id` remain in
the same split.

For personalized models, split at the event/segment level rather than
individual overlapping windows.

Record the split assignment in metadata.

------------------------------------------------------------------------

## 12. Downstream classifier

After SSL:

1.  Freeze encoder weights.
2.  Encode every labeled window.
3.  Train logistic regression on the learned representations.
4.  Evaluate binary stress vs. non-stress classification.

Because class imbalance is severe for some participants:

-   report AUROC as in the BP study
-   also report AUPRC
-   report per-participant positive prevalence
-   do not rely on accuracy alone

Do not rebalance/downsample silently. First run the natural
distribution. If balancing is later introduced, treat it as an explicit
experimental condition.

------------------------------------------------------------------------

## 13. Active-learning experiment

Match the BP experiment:

### Pool

The AL pool contains fixed encoded windows from the training data.

The windows do **not** change between AL rounds.

### Encoder

Frozen throughout AL.

### Classifier

Logistic regression retrained after each acquisition round.

### Acquisition functions

At minimum:

1.  **Coreset** using cosine distance in frozen representation space.
2.  **Uniform random** baseline.

Use the same initialization fraction, acquisition batch size, query
budget, number of seeds, and evaluation protocol as the BP experiment
unless dataset size makes an exact value impossible. If a value must
change, document the reason.

The current BP study uses:

``` text
initial labeled fraction = 0.18%
K = 3 acquisitions per round
4 random seeds
```

Verify these against the existing BP experiment code before execution.

------------------------------------------------------------------------

## 14. Main AL question

The central analysis is:

> When a global active-learning model acquires labeled windows, which
> participants do those acquired examples come from?

For each round `r` and participant `p`, compute:

``` text
n_p(r) = cumulative acquired windows from participant p
```

and participant share:

``` text
s_p(r) = n_p(r) / sum_i n_i(r)
```

Compare against even allocation:

``` text
1 / N
```

and compute:

``` text
delta_p(r) = s_p(r) - 1/N
```

This should directly mirror the BP-spike analysis.

------------------------------------------------------------------------

## 15. Required AL plots

### Plot E: per-participant learning curves

For the global model:

-   one panel per participant
-   x = cumulative acquired labels/windows
-   y = AUROC
-   coreset vs. random
-   mean over seeds
-   uncertainty/min-max band
-   fully supervised upper bound

### Plot F: acquisition burden by participant

At selected AL rounds or over the full budget:

``` text
participant -> cumulative number of queried windows
```

Compare coreset vs. random.

### Plot G: participant share over AL rounds

For every participant:

``` text
x = AL round
y = share of cumulative acquired windows
```

Separate coreset and random if necessary for readability.

### Plot H: deviation from even allocation

``` text
delta_p(r) = s_p(r) - 1/N
```

This is the clearest diagnostic for whether coreset systematically
concentrates acquisition on particular participants.

### Plot I: final acquisition share vs. available data

This is important for ADARP because participants have very different
numbers of windows.

Plot or correlate:

``` text
participant's fraction of total available pool
vs.
participant's fraction of AL acquisitions
```

This distinguishes "coreset selects participant X heavily because X
simply has more data" from genuine over-selection relative to
availability.

Also compute a normalized acquisition ratio:

``` text
acquisition_share / pool_share
```

Values:

-   1 = participant is over-selected relative to available windows

-   \<1 = participant is under-selected

This is especially important given the large per-participant class/data
imbalance already observed.

------------------------------------------------------------------------

## 16. Event-provenance sensitivity analysis

The primary result should remain window-level participant contribution
because that matches the research question.

However, add a sensitivity analysis using `event_id`.

For stress windows selected by AL, compute:

``` text
number of unique originating stress tags represented
```

per participant.

Compare:

``` text
window-level acquisition share
vs.
unique-event-level share
```

This tests whether a participant appears dominant merely because AL
repeatedly selects overlapping windows from the same small number of
stress events.

Do not replace the primary window-level analysis with this; use it as a
robustness check.

------------------------------------------------------------------------

## 17. Compare ADARP with BP-spike dataset

Produce a concise comparison table:

  -----------------------------------------------------------------------
  Property                BP Spike                ADARP
  ----------------------- ----------------------- -----------------------
  Participants            20                      11

  Continuous signals      HR + steps              EDA + HR

  Human label source      self-initiated BP       self-initiated E4
                          measurement             stress button

  Positive label          timestamped measurement timestamped subjective
  precision                                       stress anchor

  Post-processing         context/window          +/-20-min stress
                          construction            segment, +/-60-min
                                                  negative exclusion

  SSL                     modality-specific       modality-specific
                          encoders                encoders

  Downstream classifier   logistic regression     logistic regression

  AL unit                 encoded window/example  encoded window/example

  Main analysis           participant source of   participant source of
                          queries                 queries
  -----------------------------------------------------------------------

Clearly state that ADARP stress windows are derived from human
button-press annotations through a predefined temporal labeling rule,
whereas non-stress is rule-derived from physiology sufficiently far from
stress tags.

------------------------------------------------------------------------

## 18. Dataset-validity criteria

After preprocessing and baseline modeling, decide whether ADARP is
appropriate using these criteria.

### A. Sufficient labeled examples

Check:

-   number of usable stress events per participant
-   number of stress/non-stress windows per participant
-   whether any participants have too few positive events for meaningful
    evaluation

Do not judge sufficiency from the total window count alone; retain event
counts.

### B. Class imbalance

Determine whether severe imbalance makes participant-level evaluation
unstable.

### C. Cross-participant heterogeneity

Check whether baseline classifier performance varies substantially
across participants.

### D. SSL usefulness

Compare downstream performance using:

1.  SSL representations
2.  simple/raw or handcrafted baseline if available

The representation should provide a meaningful prediction signal before
investing in AL.

### E. AL behavior

Determine whether:

-   performance improves with additional queried labels
-   coreset differs meaningfully from random
-   participant acquisition is concentrated
-   concentration remains after normalizing for each participant's pool
    size

### F. Comparability with BP

ADARP does not need identical sensor preprocessing, but it should
support the same conceptual experiment:

``` text
continuous wearable signals
-> SSL representation
-> sparse/derived labeled examples
-> global vs. personalized classifier
-> active learning
-> participant-level acquisition analysis
```

------------------------------------------------------------------------

## 19. Sanity checks that must run before the full experiment

Implement assertions/checks for:

``` text
1. No window crosses a recording-session boundary.
2. Every stress window maps to exactly one retained stress event.
3. Stress windows contain complete EDA + HR coverage.
4. Non-stress windows do not overlap any +/-60-min tag exclusion zone.
5. No train/test leakage of overlapping windows from the same event.
6. Participant IDs are preserved after encoding.
7. Event IDs are preserved after encoding.
8. Label distribution is reported before any balancing.
9. Duplicate windows caused by overlapping +/-20-min regions around nearby tags are detected.
```

For item 9, explicitly check whether two stress tags are less than 40
minutes apart. Their +/-20-minute segments overlap and can generate
duplicate/near-duplicate physiological windows. Decide on a
deterministic handling rule and report it.

Possible handling options to compare:

-   preserve both event-derived segments but track duplicate timestamps
-   merge overlapping positive intervals before windowing
-   deduplicate exact `(participant, session, window_start, window_end)`
    windows

Do not silently choose one without reporting its effect on counts.

------------------------------------------------------------------------

## 20. Recommended execution order

Run the work in this order:

``` text
1. Load all participants/sessions
2. Validate EDA + HR timestamps and coverage
3. Load all tags
4. Compute usable +/-20-min stress events
5. Extract +/-20-min stress segments
6. Extract non-stress segments outside +/-60-min buffers
7. Check overlapping tag-centered stress segments
8. Window each segment: 60 s / 30 s stride
9. Preserve participant/event/session provenance
10. Produce event-count + window-count + class-balance plots
11. Build leakage-safe train/test splits
12. Pretrain SSL encoder
13. Freeze encoder and encode labeled windows
14. Train fully supervised logistic-regression baseline
15. Produce per-participant AUROC + AUPRC
16. Decide whether predictive signal is adequate
17. Run global AL: coreset vs. random
18. Run personalized AL reference
19. Produce acquisition-distribution plots
20. Normalize acquisition share by pool share
21. Run event-provenance sensitivity analysis
22. Compare results with BP-spike experiment
23. Decide whether ADARP is a valid second dataset
```

------------------------------------------------------------------------

## 21. Deliverables

Save:

``` text
outputs/
├── tables/
│   ├── participant_event_counts.csv
│   ├── participant_window_counts.csv
│   ├── participant_class_balance.csv
│   ├── baseline_metrics.csv
│   ├── al_metrics_by_round.csv
│   └── acquisition_by_participant.csv
│
├── figures/
│   ├── event_counts.png
│   ├── window_counts.png
│   ├── stress_fraction.png
│   ├── tag_context_examples/
│   ├── baseline_per_participant.png
│   ├── global_learning_curves.png
│   ├── personalized_learning_curves.png
│   ├── acquisition_counts.png
│   ├── acquisition_share.png
│   ├── acquisition_deviation.png
│   ├── acquisition_vs_pool_share.png
│   └── event_provenance_sensitivity.png
│
└── processed/
    ├── window_metadata.parquet
    ├── ssl_representations.parquet
    └── split_assignments.csv
```

Also produce a short Markdown summary containing:

1.  exact preprocessing rules used
2.  final event/window counts
3.  class balance
4.  leakage-prevention strategy
5.  baseline prediction performance
6.  AL performance
7.  participant acquisition distribution
8.  pool-size-normalized acquisition distribution
9.  event-level sensitivity result
10. final assessment of whether ADARP supports the same research
    question as BP

------------------------------------------------------------------------

## 22. Important interpretation rules

Do not make these claims:

-   "Each stress window is a separate human button press."
-   "Non-stress windows are directly human-labeled."
-   "A +/-20-minute segment means the participant was continuously
    stressed for 40 minutes."
-   "More acquired windows automatically means more participant
    interactions."

Use these instead:

-   "Stress windows are derived from participant-generated stress-event
    annotations."
-   "Non-stress windows are rule-derived from periods sufficiently
    distant from stress tags."
-   "AL acquisition counts measure which participants contribute
    selected labeled windows."
-   "Event provenance is retained to quantify how many unique human
    annotations underlie selected stress windows."

The main research question remains:

> **Does active learning on a global wearable model concentrate its
> selected labeled examples on a subset of participants, and does that
> concentration differ from random acquisition?**
