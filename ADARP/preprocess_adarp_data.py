"""Turn loaded ADARP recordings into labelled 60 s windows with provenance.

The label construction follows the original ADARP work (rameshKrSah/ADARP_Dataset):

  stress      -- a 40 min interval centred on each usable button press
                 (20 min either side). A press repeated within a minute is the
                 same moment hit twice, so only the first is kept. Presses
                 further apart keep their own intervals even where those
                 overlap: the intervals are never fused, and a window covered by
                 two of them enters the dataset once, annotated with both.
  not-stress  -- the spans of a session that open only once a button press's
                 whole +/- 60 min window has gone by, and that still sit 60 min
                 clear of *every* other press; a session with no press at all
                 contributes whole.

Windows are 60 s with a 30 s stride (50% overlap) on one grid per session, so
the same seconds always produce the same window whichever interval asked for it;
a window is kept only where it lies wholly inside an interval.
EDA is low-pass filtered at its native 4 Hz and then averaged onto HR's 1 Hz grid
before anything is segmented, so both modalities are windowed once on one grid and a
window is 60 samples either way. It is min-max normalized per segment as in the
reference; HR is left raw.

Every window keeps the metadata needed to map it back to its source -- the
participant, session, segment and originating tag -- because the downstream
active-learning analysis asks *whose* windows get acquired, and because 79
windows from one button press are 79 model examples from a single human
annotation, not 79 annotations.

Input frames come from `load_adarp`.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from load_adarp import EDA_HZ, HR_HZ

# windowing baseline, unchanged from ADARP
WINDOW_SEC = 60
STEP_SEC = 30

# a stress event is the tag +/- CONTEXT_MIN; non-stress must clear every tag
# by BUFFER_MIN
CONTEXT_MIN = 20
BUFFER_MIN = 60

# presses this close together are one moment hit twice, not two events
REFRACTORY_S = 60

# non-stress windows kept per stress window; None keeps the whole pool
NONSTRESS_RATIO = 2.0

# EDA is averaged down onto HR's grid before segmenting, so both are carried at
# this rate and a window holds 60 samples of each
PIPELINE_HZ = HR_HZ

EDA_LEN = WINDOW_SEC * PIPELINE_HZ   # 60 samples, post-downsample
HR_LEN = WINDOW_SEC * HR_HZ          # 60 samples

# 2nd-order Butterworth at 1.25 Hz, as in the reference pipeline
EDA_CUTOFF_HZ = 5.0 / EDA_HZ

NS = 1_000_000_000

# ---- hand-off to the active-learning pipeline -------------------------------

# The shared pipeline reads a fixed 30 bins per channel (src.signal_utils.WINDOW_SIZE,
# src.classifier_utils.FEATURE_POINTS), so both modalities are harmonised to 1 Hz and
# then binned to 30. EDA's 240 samples and HR's 60 both reduce by averaging, and the
# averaging is what keeps the reduction honest: taking every 8th EDA sample instead
# would fold everything above 0.25 Hz back into the band being modelled, and the
# reference low-pass sits at 1.25 Hz -- five times too high to prevent that. A boxcar
# mean is its own anti-alias filter.
#
# EDA reaches 1 Hz in `downsample_eda`, before segmenting; this is the second and
# last reduction, 60 samples to 30 bins, applied identically to both modalities.
FEATURE_POINTS = 30

# where export_signals writes, relative to the processed directory
SIGNAL_SUBDIR = "signals"

# whole sessions per split, in time order
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2


# ---------------------------------------------------------------- tag validity

def session_spans(df):
    """First/last real EDA and HR sample per session, cached on `df.attrs`.

    Scanning all ~24M rows once per tag is far too slow, and every coverage
    question reduces to these spans.
    """
    if "session_spans" in df.attrs:
        return df.attrs["session_spans"]

    spans = {}
    for name, session in df.groupby("session", observed=True):
        eda = session["eda"].dropna()
        hr = session["hr"].dropna()
        spans[name] = {
            "eda": (eda.index.min(), eda.index.max()) if len(eda) else None,
            "hr": (hr.index.min(), hr.index.max()) if len(hr) else None,
        }

    df.attrs["session_spans"] = spans
    return spans


def check_tag_coverage(tags, df, context_min=CONTEXT_MIN):
    """Flag which button presses actually have signal around them.

    usable        -> the tag instant falls inside both EDA and HR
    full_context  -> the whole +/- context_min segment does too
    """
    context = pd.Timedelta(minutes=context_min)
    spans = session_spans(df)

    out = tags.copy()
    for column in ("has_eda", "has_hr", "full_eda", "full_hr"):
        out[column] = False

    for i, row in out.iterrows():
        span = spans.get(row["session"])
        if span is None:
            continue  # session never made it into df (e.g. empty HR file)

        t = row["tag_time"]

        for signal in ("eda", "hr"):
            if span[signal] is None:
                continue
            first, last = span[signal]
            out.loc[i, f"has_{signal}"] = first <= t <= last
            out.loc[i, f"full_{signal}"] = (first <= t - context
                                            and t + context <= last)

    out["usable"] = out["has_eda"] & out["has_hr"]
    out["full_context"] = out["full_eda"] & out["full_hr"]
    return out


def valid_tags(tags):
    """The presses that can carry a full stress segment, renumbered as events."""
    valid = tags[tags["usable"] & tags["full_context"]].reset_index(drop=True)
    valid.insert(0, "event_id", [f"event_{i:03d}" for i in range(len(valid))])
    return valid


def collapse_repeats(events, refractory_s=REFRACTORY_S):
    """Keep the first press of each burst and drop the rest.

    Presses a few seconds apart are the button hit twice for one moment, not two
    events: their segments would be near-identical, so the same signal would
    enter the pool twice under two event ids. Merging them would keep both ids
    on one span; dropping is the sharper statement -- there was one annotation.

    Event ids are left as they were, so a dropped press stays identifiable; the
    dropped rows ride along in `.attrs["dropped"]`.
    """
    if not len(events) or not refractory_s:
        out = events.copy()
        out.attrs["dropped"] = events.iloc[:0]
        return out

    gap = pd.Timedelta(seconds=refractory_s)
    keep = []

    for _, session_events in events.groupby("session", observed=True):
        last = None
        for i, row in session_events.sort_values("tag_time").iterrows():
            if last is not None and row["tag_time"] - last < gap:
                continue
            keep.append(i)
            last = row["tag_time"]

    kept = events.loc[sorted(keep)].reset_index(drop=True)
    kept.attrs["dropped"] = events.drop(index=keep)
    return kept


def overlapping_events(tags, context_min=CONTEXT_MIN):
    """Pairs of stress events whose +/- context_min segments overlap.

    Overlap means the same seconds of signal become windows twice, under the
    same label, so it inflates the positive class and couples two "independent"
    events. Worth knowing before counting anything.
    """
    context = pd.Timedelta(minutes=context_min)
    pairs = []

    for session, group in tags.groupby("session", observed=True):
        group = group.sort_values("tag_time")
        times = group["tag_time"].tolist()
        ids = group.get("event_id", pd.Series(group.index)).tolist()

        for (a, ta), (b, tb) in zip(zip(ids, times), zip(ids[1:], times[1:])):
            if tb - ta < 2 * context:
                pairs.append({
                    "session": session,
                    "first": a,
                    "second": b,
                    "gap_min": (tb - ta).total_seconds() / 60,
                })

    return pd.DataFrame(pairs, columns=["session", "first", "second", "gap_min"])


# -------------------------------------------------------------------- segments

def _one_participant(df, tags):
    """Guard against mixing people: their sessions would pollute each other."""
    people = set(df["participant"].unique()) | set(tags["participant"].unique())
    if len(people) > 1:
        raise ValueError(f"df and tags mix participants: {sorted(people)}")


def event_groups(tags, context_min=CONTEXT_MIN, merge_overlapping=False):
    """Presses grouped into the spans that will become stress segments.

    One span per press by default: two presses half an hour apart are two
    annotations, and collapsing them would lose that. The signal they share is
    handled after windowing instead, by `drop_duplicate_windows`.

    `merge_overlapping` is the alternative -- presses whose +/- context_min
    spans touch become one span covering all of them. The presses behind a span
    are kept in `event_ids` either way, so an event-level sensitivity analysis
    stays possible.
    """
    context = pd.Timedelta(minutes=context_min)
    groups = []

    for session, session_tags in tags.groupby("session", observed=True):
        current = None

        for _, row in session_tags.sort_values("tag_time").iterrows():
            start = row["tag_time"] - context
            end = row["tag_time"] + context

            if merge_overlapping and current is not None and start <= current["end"]:
                current["end"] = max(current["end"], end)
                current["event_ids"].append(row["event_id"])
                current["tag_times"].append(row["tag_time"])
                continue

            current = {
                "participant": row["participant"],
                "session": session,
                "start": start,
                "end": end,
                "event_ids": [row["event_id"]],
                "tag_times": [row["tag_time"]],
            }
            groups.append(current)

    return sorted(groups, key=lambda g: g["start"])


def stress_segments(df, tags, context_min=CONTEXT_MIN, merge_overlapping=False):
    """One segment per event span: the tag +/- context_min.

    Overlapping presses each keep their own 40 min segment; the windows they
    both claim are dropped later rather than the segments being fused.
    """
    _one_participant(df, tags)

    segments = []

    for i, group in enumerate(event_groups(tags, context_min, merge_overlapping)):
        session = df[df["session"] == group["session"]]
        frame = session.loc[group["start"]:group["end"]]

        if frame.empty:
            continue

        segments.append({
            "frame": frame,
            "start": group["start"],
            "end": group["end"],
            "participant": group["participant"],
            "session": group["session"],
            "segment_type": "stress",
            "segment_id": f"stress_{i:03d}",
            "event_id": "+".join(group["event_ids"]),
            "n_events": len(group["event_ids"]),
            "tag_time": group["tag_times"][0],
            "label": 1,
        })

    return segments


def nonstress_segments(df, tags, buffer_min=BUFFER_MIN, side="after"):
    """Continuous spans sitting at least `buffer_min` clear of every press.

    `tags` should be *all* of the participant's presses, not only the ones that
    passed coverage validation: a press that fails validation still marks a
    reported stress moment, so the signal around it is not clean non-stress.

    `side="after"` only opens a non-stress span once a press's whole +/-
    `buffer_min` window has gone by, so the run-up to the first press of a
    session is discarded rather than counted as calm. `side="both"` also keeps
    that leading span, ending it `buffer_min` before the press. A session with
    no press at all contributes whole either way.
    """
    if side not in ("after", "both"):
        raise ValueError(f'side must be "after" or "both", got {side!r}')

    _one_participant(df, tags)

    buffer = pd.Timedelta(minutes=buffer_min)
    segments = []

    for session_name, session in df.groupby("session", observed=True):
        session = session.sort_index()

        eda = session["eda"].dropna()
        hr = session["hr"].dropna()
        if eda.empty or hr.empty:
            continue

        session_start = max(eda.index.min(), hr.index.min())
        session_end = min(eda.index.max(), hr.index.max())

        session_tags = (
            tags.loc[tags["session"] == session_name, "tag_time"]
            .sort_values()
            .tolist()
        )

        # tag1+60m ---- tag2-60m | tag2+60m ---- session end, and with
        # side="both" the session start ---- tag1-60m run-up as well
        candidate_start = session_start
        past_a_window = side == "both"

        for tag_time in session_tags:
            candidate_end = tag_time - buffer

            if past_a_window and candidate_end > candidate_start:
                frame = session.loc[candidate_start:candidate_end]
                if not frame.empty:
                    segments.append(frame)

            candidate_start = max(candidate_start, tag_time + buffer)
            past_a_window = True

        if candidate_start < session_end:
            frame = session.loc[candidate_start:session_end]
            if not frame.empty:
                segments.append(frame)

    participant = str(df["participant"].iloc[0])

    return [
        {
            "frame": frame,
            "participant": participant,
            "session": str(frame["session"].iloc[0]),
            "segment_type": "nonstress",
            "segment_id": f"nonstress_{i:03d}",
            "event_id": pd.NA,
            "n_events": 0,
            "tag_time": pd.NaT,
            "label": 0,
        }
        for i, frame in enumerate(segments)
    ]


# --------------------------------------------------------------- EDA transforms

def butter_lowpass_filter(values, cutoff=EDA_CUTOFF_HZ, sample_rate=EDA_HZ, order=2):
    """Zero-phase Butterworth low-pass filter."""
    b, a = butter(order, cutoff / (0.5 * sample_rate), btype="lowpass")
    return filtfilt(b, a, values)


def minmax(values):
    """Min-max scale into [0, 1]; a flat stretch becomes zeros."""
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def downsample_eda(df, hz=PIPELINE_HZ, filter_eda=True):
    """Average EDA onto HR's grid, before anything is segmented.

    The order matters twice over. The low-pass runs first, at the native 4 Hz,
    because the reference cutoff of 1.25 Hz sits above the 0.5 Hz Nyquist of a
    1 Hz grid and `butter` would refuse it afterwards. The reduction is then a
    boxcar mean, which is its own anti-alias filter -- taking every fourth sample
    instead would fold everything above 0.5 Hz back into the band being modelled.

    Doing it here rather than after windowing means segments and windows are cut
    once, on a single grid, with both modalities already sitting on it.

    Resampling runs inside each session: sessions sit hours or days apart, and a
    resample across the whole frame would manufacture bins spanning those gaps.
    `origin="start"` keeps each session's bins in phase with its own first sample,
    which `load_session` has already trimmed to the instant both modalities cover.
    """
    rule = pd.Timedelta(seconds=1 / hz)
    parts = []

    for session, block in df.groupby("session", observed=True):
        eda = block["eda"].dropna()

        if eda.empty:
            parts.append(block)
            continue

        values = eda.to_numpy(dtype=float)
        # filtfilt pads by 3 * max(len(a), len(b)) = 9 samples and refuses less
        if filter_eda and len(values) > 9:
            values = butter_lowpass_filter(values)

        binned = pd.Series(values, index=eda.index).resample(rule, origin="start").mean()

        merged = block.drop(columns=["eda"]).join(binned.rename("eda"), how="outer")
        merged["participant"] = block["participant"].iloc[0]
        merged["session"] = session
        parts.append(merged)

    if not parts:
        return df

    out = pd.concat(parts).sort_index()
    out.index.name = df.index.name or "timestamp"
    return out


def prepare_eda(series, normalize="segment"):
    """Scale one segment's EDA.

    Filtering is not done here: it belongs at the native 4 Hz, which is upstream
    of `downsample_eda`, and running it again on the 1 Hz series would be both a
    second pass and an invalid one.

    `normalize="segment"` reproduces the reference, but note what it implies:
    stress segments are a fixed 40 min while non-stress segments run for hours,
    so the scaling statistics differ systematically by class. `"window"` scales
    each window instead, and None leaves the microsiemens alone; both exist so
    the choice can be tested rather than assumed.
    """
    values = series.to_numpy(dtype=float)

    if normalize == "segment":
        values = minmax(values)

    return pd.Series(values, index=series.index, name="eda")


# --------------------------------------------------------------------- windows

def _window_starts(start, end, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                   origin=None):
    """Window start instants, every `step_sec`, fully inside [start, end].

    Both bounds are the interval's *nominal* edges, not its first and last
    sample: anchoring on the first sample would lose the trailing window to a
    fraction of a sampling period.

    `origin` is the session's first instant, and the grid runs from there
    through the whole recording; this function just returns the stretch of it
    that lands inside [start, end]. Restarting the grid at each interval instead
    would give two overlapping tags interleaved windows over the seconds they
    share, so the same signal would enter the pool twice under two slightly
    different timestamps and no duplicate would ever be detectable.
    """
    step_ns = step_sec * NS

    if origin is not None:
        # advance to the first grid point of the session's own timeline
        offset = (start - origin) % step_ns
        if offset:
            start += step_ns - offset

    span = end - start
    n = int((span - window_sec * NS) // (step_sec * NS)) + 1

    if n <= 0:
        return np.empty(0, dtype=np.int64)

    return start + np.arange(n, dtype=np.int64) * step_sec * NS


def _gather(series, starts_ns, n_samples, rate_hz):
    """Fixed-length slices of `series` at each start; NaN rows where impossible.

    A window only counts if the samples are actually there and contiguous: a
    recording gap inside the window would otherwise be silently stitched shut
    and become a fabricated example.
    """
    times = series.index.asi8
    values = series.to_numpy(dtype=float)

    out = np.full((len(starts_ns), n_samples), np.nan)
    if len(times) < n_samples:
        return out

    period_ns = NS / rate_hz
    pos = np.searchsorted(times, starts_ns, side="left")

    keep = pos + n_samples <= len(times)
    if not keep.any():
        return out

    idx = np.where(keep)[0]
    first = pos[idx]
    last = first + n_samples - 1

    # the slice must begin within one sample of the window, and span no more
    # than the nominal duration (allowing a little clock jitter)
    aligned = times[first] - starts_ns[idx] <= period_ns
    contiguous = (times[last] - times[first]) <= (n_samples - 1) * period_ns * 1.01

    idx = idx[aligned & contiguous]
    if not len(idx):
        return out

    rows = pos[idx][:, None] + np.arange(n_samples)
    out[idx] = values[rows]
    return out


def window_segment(segment, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                   normalize="segment", origin=None):
    """Cut one segment into aligned EDA/HR windows plus their provenance."""
    frame = segment["frame"]

    eda = frame["eda"].dropna()
    hr = frame["hr"].dropna()

    if eda.empty or hr.empty:
        return None

    eda = prepare_eda(eda, normalize=normalize)

    start = segment.get("start") or frame.index[0]
    end = segment.get("end") or frame.index[-1]

    starts = _window_starts(start.value, end.value, window_sec, step_sec,
                            None if origin is None else origin.value)
    if not len(starts):
        return None

    # both modalities are on the same grid by now, so both gather 60 samples
    eda_windows = _gather(eda, starts, window_sec * PIPELINE_HZ, PIPELINE_HZ)
    hr_windows = _gather(hr, starts, window_sec * HR_HZ, HR_HZ)

    # keep only windows both modalities could fill
    complete = ~np.isnan(eda_windows).any(axis=1) & ~np.isnan(hr_windows).any(axis=1)
    if not complete.any():
        return None

    eda_windows = eda_windows[complete]
    hr_windows = hr_windows[complete]
    kept = starts[complete]

    if normalize == "window":
        eda_windows = np.apply_along_axis(minmax, 1, eda_windows)

    starts_utc = pd.to_datetime(kept, utc=True)

    meta = pd.DataFrame({
        "participant": segment["participant"],
        "session": segment["session"],
        "segment_type": segment["segment_type"],
        "segment_id": segment["segment_id"],
        "event_id": segment["event_id"],
        "n_events": segment["n_events"],
        "session_start": origin,
        "segment_start": start,
        "segment_end": end,
        "window_id": [
            f"{segment['segment_id']}_w{i:04d}" for i in range(len(kept))
        ],
        "window_start": starts_utc,
        "window_end": starts_utc + pd.Timedelta(seconds=window_sec),
        "label": segment["label"],
    })

    return eda_windows, hr_windows, meta


def window_segments(segments, origins=None, **kwargs):
    """Window a list of segments into (eda, hr, meta) stacked together.

    `origins` maps a session to its first instant, so every window of that
    session sits on one grid however many intervals it is cut from.
    """
    eda_parts, hr_parts, meta_parts = [], [], []

    for segment in segments:
        origin = None if origins is None else origins.get(segment["session"])
        windowed = window_segment(segment, origin=origin, **kwargs)
        if windowed is None:
            continue
        eda, hr, meta = windowed
        eda_parts.append(eda)
        hr_parts.append(hr)
        meta_parts.append(meta)

    if not eda_parts:
        return (np.empty((0, EDA_LEN)), np.empty((0, HR_LEN)), _empty_meta())

    return (
        np.concatenate(eda_parts),
        np.concatenate(hr_parts),
        pd.concat(meta_parts, ignore_index=True),
    )


DUPLICATE_KEY = ["participant", "session", "window_start", "window_end"]


def drop_duplicate_windows(eda, hr, meta):
    """Keep one copy of each window, identified by when and whose it is.

    Two presses less than 40 min apart claim overlapping +/- 20 min intervals,
    so the seconds they share are windowed twice, once under each tag. Both tags
    stay -- each is its own annotation -- but the window enters the dataset once,
    keeping the copy from the earlier tag.

    The key is `DUPLICATE_KEY`: the participant, the session and the window's
    own start and end. Sensor values are never compared, and could not settle it
    anyway, since EDA is normalized against the interval a copy came from.
    """
    keep = ~meta.duplicated(subset=DUPLICATE_KEY, keep="first")

    if keep.all():
        return eda, hr, meta, 0

    return (
        eda[keep.to_numpy()],
        hr[keep.to_numpy()],
        meta[keep].reset_index(drop=True),
        int((~keep).sum()),
    )


def label_covering_events(meta, events, context_min=CONTEXT_MIN):
    """Name every valid tag whose +/- context_min interval covers each window.

    A deduplicated window carries the id of the tag that emitted it, which is
    the earliest one covering it. The other tags are just as much annotations of
    that window, so they are listed too: `event_ids` holds all of them and
    `n_events` counts them. The intervals themselves are never merged.
    """
    out = meta.copy()
    covering = [[] for _ in range(len(out))]

    if len(events) and len(out):
        # stay in pandas: these instants are tz-aware, numpy's are not
        starts = out["window_start"]
        ends = out["window_end"]
        context = pd.Timedelta(minutes=context_min)

        for event_id, tag_time in zip(events["event_id"], events["tag_time"]):
            inside = ((starts >= tag_time - context)
                      & (ends <= tag_time + context)).to_numpy()
            for i in np.flatnonzero(inside):
                covering[i].append(event_id)

    out["event_ids"] = ["+".join(ids) if ids else pd.NA for ids in covering]
    out["n_events"] = [len(ids) for ids in covering]
    return out


def _empty_meta():
    return pd.DataFrame(columns=[
        "participant", "session", "segment_type", "segment_id", "event_id",
        "event_ids", "n_events", "session_start", "segment_start",
        "segment_end", "window_id", "window_start", "window_end", "label",
    ])


# ------------------------------------------------------------------- pipelines

def build_participant(df, tags, context_min=CONTEXT_MIN, buffer_min=BUFFER_MIN,
                      side="after", merge_overlapping=False,
                      refractory_s=REFRACTORY_S, dedupe=True,
                      filter_eda=True, hz=PIPELINE_HZ, **window_kwargs):
    """Every labelled window for one participant, with its provenance.

    Repeat presses are dropped. Presses that merely overlap stay separate, since
    each is its own annotation, and the windows they both claim are deduplicated
    afterwards instead. Stress segments come from the validated presses only;
    the non-stress buffer is applied against all of them.

    EDA is brought onto HR's grid first, so segmenting, windowing and the window
    grid itself all happen once, at one rate, for both modalities.
    """
    participant_id = str(df["participant"].iloc[0])

    # tag coverage is asked of the recording as it was sampled, before the rate
    # is touched, so a press is judged against the real span either way
    tags = check_tag_coverage(tags, df, context_min=context_min)

    df = downsample_eda(df, hz=hz, filter_eda=filter_eda)
    events = collapse_repeats(valid_tags(tags), refractory_s=refractory_s)

    segments = (stress_segments(df, events, context_min=context_min,
                                merge_overlapping=merge_overlapping)
                + nonstress_segments(df, tags, buffer_min=buffer_min, side=side))

    # one window grid per session, shared by every interval cut from it
    origins = df.reset_index().groupby("session", observed=True)["timestamp"].min()

    eda, hr, meta = window_segments(segments, origins=origins.to_dict(),
                                    **window_kwargs)

    duplicates = 0
    if dedupe:
        eda, hr, meta, duplicates = drop_duplicate_windows(eda, hr, meta)

    meta = label_covering_events(meta, events, context_min=context_min)

    return {
        "participant": participant_id,
        "eda": eda,
        "hr": hr,
        "meta": meta,
        "tags": tags,
        "events": events,
        "repeats_dropped": len(events.attrs["dropped"]),
        "duplicate_windows_dropped": duplicates,
    }


def subsample_nonstress(eda, hr, meta, ratio=NONSTRESS_RATIO, seed=0):
    """Keep every stress window and `ratio` non-stress windows for each.

    The non-stress pool is an order of magnitude larger than the stress one and
    most of it is idle recording, so it is sampled down rather than carried
    whole. Sampling is uniform over windows, which means a long session
    contributes in proportion to its length.

    Where the pool cannot cover the ratio the whole of it is kept, so the result
    is `ratio` or the best available, never padding. `seed` fixes the draw.
    """
    stress = np.flatnonzero(meta["label"].to_numpy() == 1)
    pool = np.flatnonzero(meta["label"].to_numpy() == 0)

    if ratio is None:
        return eda, hr, meta, len(pool)

    target = int(round(ratio * len(stress)))

    if len(pool) <= target:
        return eda, hr, meta, len(pool)

    drawn = np.random.default_rng(seed).choice(pool, size=target, replace=False)
    keep = np.sort(np.concatenate([stress, drawn]))

    return (
        eda[keep],
        hr[keep],
        meta.iloc[keep].reset_index(drop=True),
        len(pool),
    )


def pool_metadata(per_participant):
    """The provenance of every window built, before any subsampling."""
    return pd.concat(
        [built["meta"] for built in per_participant.values()], ignore_index=True
    )


def sampling_table(meta, per_participant):
    """What the subsample kept, against what each participant had to offer."""
    rows = {}

    for participant_id, built in per_participant.items():
        pool = built["meta"]
        kept = meta[meta["participant"] == participant_id]

        stress = int((kept["label"] == 1).sum())
        nonstress = int((kept["label"] == 0).sum())

        rows[participant_id] = {
            "stress": stress,
            "nonstress_available": int((pool["label"] == 0).sum()),
            "nonstress_kept": nonstress,
            "ratio": round(nonstress / stress, 2) if stress else np.nan,
            "total": stress + nonstress,
            "total_pool": stress + int((pool["label"] == 0).sum()),
        }

    table = pd.DataFrame(rows).T
    table.index.name = "participant"
    return table


def build_dataset(df, tags_by_participant, nonstress_ratio=NONSTRESS_RATIO,
                  seed=0, **kwargs):
    """Run `build_participant` across everyone and stack the results.

    The returned arrays are the model dataset, with the non-stress side sampled
    down to `nonstress_ratio` windows per stress window, participant by
    participant so nobody's balance is set by anybody else's pool.
    `per_participant` keeps each full pool, which is what the validation table
    and the segment diagnostics report on.

    Returns (eda, hr, meta, per_participant); `meta` is the provenance table
    that indexes both arrays row for row.
    """
    per_participant = {}
    eda_parts, hr_parts, meta_parts = [], [], []

    for participant_id, tags in tags_by_participant.items():
        built = build_participant(
            df[df["participant"] == participant_id], tags, **kwargs
        )
        per_participant[participant_id] = built

        sampled = subsample_nonstress(
            built["eda"], built["hr"], built["meta"],
            ratio=nonstress_ratio, seed=seed,
        )

        eda_parts.append(sampled[0])
        hr_parts.append(sampled[1])
        meta_parts.append(sampled[2])

    eda = np.concatenate(eda_parts) if eda_parts else np.empty((0, EDA_LEN))
    hr = np.concatenate(hr_parts) if hr_parts else np.empty((0, HR_LEN))
    meta = (pd.concat(meta_parts, ignore_index=True) if meta_parts
            else _empty_meta())

    return eda, hr, meta, per_participant


def validation_table(per_participant):
    """The section-8 table: counts per participant, straight from the windows."""
    rows = {}

    for participant_id, built in per_participant.items():
        meta = built["meta"]
        stress = meta[meta["label"] == 1]
        nonstress = meta[meta["label"] == 0]

        total = len(meta)
        rows[participant_id] = {
            "total_tags": len(built["tags"]),
            "repeat_presses_dropped": built["repeats_dropped"],
            "duplicate_windows_dropped": built["duplicate_windows_dropped"],
            "usable_stress_events": len(built["events"]),
            "stress_segments": stress["segment_id"].nunique(),
            "stress_windows": len(stress),
            "nonstress_segments": nonstress["segment_id"].nunique(),
            "nonstress_windows": len(nonstress),
            "total_windows": total,
            "stress_fraction": len(stress) / total if total else np.nan,
        }

    table = pd.DataFrame(rows).T
    table.index.name = "participant"

    counts = [c for c in table.columns if c != "stress_fraction"]
    table[counts] = table[counts].astype(int)
    return table


def segment_durations(meta):
    """One row per segment: how long it is, and how much of it survived.

    `expected_windows` is what the segment's span yields on its own, so
    `windows_lost` is what was taken off it afterwards: the windows an earlier
    overlapping segment already claimed, plus anything a recording gap cost.
    Comparing first to last window would miss both, since the edges survive.
    """
    rows = (
        meta.groupby(["participant", "segment_type", "segment_id"], observed=True)
        .agg(
            session=("session", "first"),
            session_start=("session_start", "first"),
            event_id=("event_id", "first"),
            n_events=("n_events", "first"),
            windows=("window_id", "size"),
            start=("segment_start", "first"),
            end=("segment_end", "first"),
            first_window=("window_start", "min"),
            last_window=("window_end", "max"),
        )
        .reset_index()
    )

    span_sec = (rows["end"] - rows["start"]).dt.total_seconds()

    rows["span_min"] = span_sec / 60
    rows["covered_min"] = (
        (rows["last_window"] - rows["first_window"]).dt.total_seconds() / 60
    )
    rows["expected_windows"] = [
        len(_window_starts(start.value, end.value,
                           origin=None if pd.isna(origin) else origin.value))
        for start, end, origin in zip(rows["start"], rows["end"],
                                      rows["session_start"])
    ]
    rows["windows_lost"] = rows["expected_windows"] - rows["windows"]

    return rows.drop(columns=["first_window", "last_window"])


def segment_gap_table(built):
    """One row per segment, with both gaps that matter.

    `press_gaps_min` are the gaps between the presses merged into a stress
    segment -- empty for an isolated press, and bounded by 2 * CONTEXT_MIN since
    anything wider would not have overlapped. `gap_to_next_min` is the silence
    until the next segment of the same session, which for non-stress is the
    exclusion zone a press carved out.
    """
    rows = segment_durations(built["meta"])
    tag_time = dict(zip(built["events"]["event_id"], built["events"]["tag_time"]))

    press_gaps = []
    for event_id in rows["event_id"]:
        if pd.isna(event_id):
            press_gaps.append([])
            continue
        times = sorted(tag_time[e] for e in str(event_id).split("+"))
        press_gaps.append([
            round((b - a).total_seconds() / 60, 2) for a, b in zip(times, times[1:])
        ])

    rows["press_gaps_min"] = press_gaps
    rows["max_press_gap_min"] = [max(g) if g else np.nan for g in press_gaps]

    rows = rows.sort_values(["session", "start"]).reset_index(drop=True)
    next_start = rows.groupby("session", observed=True)["start"].shift(-1)
    rows["gap_to_next_min"] = (next_start - rows["end"]).dt.total_seconds() / 60

    # a negative gap is an overlap: the next segment starts before this one ends
    rows["overlap_next_min"] = (-rows["gap_to_next_min"]).clip(lower=0)
    rows["windows_dropped"] = rows["windows_lost"]

    return rows[[
        "participant", "session", "segment_id", "segment_type", "event_id",
        "n_events", "span_min", "windows", "windows_dropped",
        "gap_to_next_min", "overlap_next_min", "press_gaps_min",
        "max_press_gap_min",
    ]]


def all_segment_gaps(per_participant):
    """`segment_gap_table` for everyone, stacked."""
    return pd.concat(
        [segment_gap_table(built) for built in per_participant.values()],
        ignore_index=True,
    )


def duration_table(meta, bins=None, labels=None):
    """How many segments fall in each length band, and what they contribute.

    Segment length is what sets how many windows an annotation turns into, so
    the shape of this table is the shape of the pool.
    """
    if bins is None:
        bins = [0, 10, 30, 45, 60, 120, 240, 480, np.inf]
        labels = ["<10 min", "10-30 min", "30-45 min", "45-60 min",
                  "1-2 h", "2-4 h", "4-8 h", "8 h+"]

    rows = segment_durations(meta)
    rows["band"] = pd.cut(rows["span_min"], bins=bins, labels=labels, right=False)

    table = (
        rows.pivot_table(
            index="band",
            columns="segment_type",
            values=["segment_id", "windows"],
            aggfunc={"segment_id": "size", "windows": "sum"},
            observed=False,
        )
        .fillna(0)
        .astype(int)
    )

    table.columns = [f"{kind}_{'segments' if name == 'segment_id' else name}"
                     for name, kind in table.columns]

    ordered = [c for c in ("stress_segments", "stress_windows",
                           "nonstress_segments", "nonstress_windows")
               if c in table.columns]

    table = table[ordered]
    table.loc["total"] = table.sum()
    table.index.name = "segment length"
    return table


def save_windows(path, eda, hr, meta):
    """Arrays to npz, provenance to a parquet/csv sidecar beside it."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path.with_suffix(".npz"), eda=eda, hr=hr)
    meta.to_csv(path.with_suffix(".csv"), index=False)
    return path.with_suffix(".npz"), path.with_suffix(".csv")


# --------------------------------------------- active-learning hand-off

PROCESSED_DIR = Path(__file__).parent / "processed"
WINDOWS_STEM = "adarp_windows"


def bin_mean(values, n_bins):
    """Average the last axis of `values` down to `n_bins` equal-width bins.

    The one downsampler in this module. The window arrays and the continuous
    export both go through it, so the encoder cannot end up trained at one
    temporal scale and applied at another -- a divergence nothing downstream
    would raise on, because the shapes would still line up.
    """
    values = np.asarray(values, dtype=float)
    length = values.shape[-1]

    if length % n_bins:
        raise ValueError(f"{length} samples do not divide into {n_bins} bins")

    return values.reshape(*values.shape[:-1], n_bins, length // n_bins).mean(axis=-1)


def zscore_rows(values):
    """Standardise each row to zero mean and unit variance.

    Per-window scaling, matching `process_label_window` in the BP pipeline and
    `_build_windows` in the WESAD one. The encoders train on standardised signal,
    so the windows they are applied to have to arrive on the same scale; HR in
    raw bpm would otherwise sit two orders of magnitude off.

    It also makes `prepare_eda`'s per-segment min-max irrelevant to the model. That
    normalisation is affine with coefficients fixed across a segment, a window lies
    wholly inside one segment, and a z-score is invariant to affine rescaling -- so
    the EDA windows carry the same information whichever way the segment was scaled.
    A flat row has no variance to divide by and is returned as zeros.
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)

    return np.divide(values - mean, std, out=np.zeros_like(values), where=std > 0)


def windows_to_frame(eda, hr, meta, feature_points=FEATURE_POINTS):
    """The saved windows as the frame the active-learning pipeline consumes.

    Columns are that pipeline's contract -- `hr_seq`/`st_seq` of `feature_points`
    floats, `state_val`, `user_id` -- plus the ADARP provenance, which rides along
    so acquisition can be counted in button presses as well as windows. 79 windows
    cut from one press are one human annotation, not 79, and only `event_id` can
    tell the difference afterwards.

    The window's time is `window_start`, ADARP's own name for it, and is repeated
    as `hawaii_createdat_time` purely to be found at the pipeline boundary:
    `utility.split_fingerprint` looks for that column or `datetime_local`, and
    with neither it silently drops time from the fingerprint that the driver
    writes to audit a split. Everything inside this module reads `window_start`.

    EDA takes the `st_seq` slot: the pipeline names its second channel after the
    BP study's step count, and everything downstream keys on the name alone.
    """
    # both arrive 60 wide -- EDA reached 1 Hz in downsample_eda, before windowing
    hr_bins = zscore_rows(bin_mean(hr, feature_points))
    eda_bins = zscore_rows(bin_mean(eda, feature_points))

    window_start = pd.to_datetime(meta["window_start"], utc=True)

    return pd.DataFrame({
        "window_start": window_start,
        "hawaii_createdat_time": window_start,  # pipeline alias; see the docstring
        "hr_seq": list(hr_bins),
        "st_seq": list(eda_bins),
        "state_val": meta["label"].to_numpy().astype(int),
        "user_id": meta["participant"].astype(str).to_numpy(),
        "session": meta["session"].astype(str).to_numpy(),
        "segment_id": meta["segment_id"].astype(str).to_numpy(),
        "event_id": meta["event_id"].fillna("none").astype(str).to_numpy(),
    })


def load_windows(processed_dir=PROCESSED_DIR, stem=WINDOWS_STEM):
    """Read back what `save_windows` wrote, as (eda, hr, meta)."""
    processed_dir = Path(processed_dir)
    arrays = np.load(processed_dir / f"{stem}.npz")
    meta = pd.read_csv(processed_dir / f"{stem}.csv")

    if len(meta) != len(arrays["eda"]):
        raise ValueError(
            f"{stem}.csv has {len(meta)} rows against {len(arrays['eda'])} windows; "
            "the pair is written together and must stay in step"
        )

    width = arrays["eda"].shape[1]
    if width != EDA_LEN:
        raise ValueError(
            f"{stem}.npz holds {width}-sample EDA windows against the expected "
            f"{EDA_LEN}. A {WINDOW_SEC * EDA_HZ}-wide file predates the downsample "
            "to HR's grid; rebuild it with build_dataset before using it."
        )

    return arrays["eda"], arrays["hr"], meta


# ------------------------------------------------------- continuous signal

def participant_signal_frame(df, hz=PIPELINE_HZ):
    """One participant's whole recording as a continuous `hz` stream.

    Resampling runs inside each session. A participant's sessions sit hours or
    days apart, and resampling across the whole frame would manufacture empty
    bins spanning those gaps.

    `load_session` outer-joins 4 Hz EDA against 1 Hz HR, so HR is NaN off its own
    grid; `resample().mean()` skips those rather than averaging them in.

    The frame keeps a session column so a caller can hold out whole sessions
    before the encoder ever sees them.
    """
    rule = pd.Timedelta(seconds=1 / hz)
    parts = []

    for session, block in df.groupby("session", observed=True):
        if block.empty:
            continue

        resampled = block[["eda", "hr"]].resample(rule).mean().dropna(how="all")
        if resampled.empty:
            continue

        resampled["session"] = str(session)
        parts.append(resampled)

    if not parts:
        return pd.DataFrame(
            columns=["eda", "hr", "session"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    out = pd.concat(parts).sort_index()
    out.index.name = "timestamp"
    return out


def export_signals(df, processed_dir=PROCESSED_DIR, hz=PIPELINE_HZ):
    """Write the continuous stream the self-supervised encoders train on.

    The windows in the npz are not a substitute for this. They overlap by half,
    they cover only the labelled intervals, and the non-stress side has been
    sampled down -- between them they hold a duplicated fraction of what was
    recorded. Contrastive pre-training wants all of the unlabelled signal, which
    is the whole reason it is worth doing.

    One pair of CSVs per participant, in the schema `_train_or_load_encoder`
    reads:

        <processed_dir>/signals/<pid>_hr.csv    timestamp, value, session
        <processed_dir>/signals/<pid>_eda.csv   same

    Written raw, at 1 Hz, for every session. Which sessions an encoder may see is
    a question about an experiment's splits, not about the recording, so it is
    settled in `prepare_data` instead of here.
    """
    out_dir = Path(processed_dir) / SIGNAL_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}

    for participant_id, block in df.groupby("participant", observed=True):
        frame = participant_signal_frame(block, hz=hz)
        if frame.empty:
            continue

        paths = {}
        for channel in ("hr", "eda"):
            series = frame[[channel, "session"]].dropna(subset=[channel])
            series = series.rename(columns={channel: "value"})
            series.index.name = "timestamp"

            path = out_dir / f"{participant_id}_{channel}.csv"
            series.to_csv(path)
            paths[channel] = path

        written[str(participant_id)] = paths

    return written


def load_signals(processed_dir=PROCESSED_DIR):
    """Read the exported streams back, as {participant: DataFrame}.

    Each frame is indexed by timestamp with `hr`, `eda` and `session` columns --
    the shape `participant_signal_frame` produced, reassembled from the two CSVs.
    """
    signal_dir = Path(processed_dir) / SIGNAL_SUBDIR
    if not signal_dir.exists():
        raise FileNotFoundError(
            f"No exported signals at {signal_dir}. Run export_signals first: the "
            "encoders train on the continuous stream, which the npz does not hold."
        )

    frames = {}

    hr_paths = sorted(signal_dir.glob("*_hr.csv"))
    print(f"[adarp] reading exported signals for {len(hr_paths)} participants from "
          f"{signal_dir}; this is CSV parsing, not training, and takes a few minutes",
          flush=True)

    for hr_path in hr_paths:
        participant_id = hr_path.name[: -len("_hr.csv")]
        eda_path = signal_dir / f"{participant_id}_eda.csv"
        if not eda_path.exists():
            continue

        def _read(path, column):
            frame = pd.read_csv(path, parse_dates=["timestamp"])
            frame = frame.set_index("timestamp").sort_index()
            return frame.rename(columns={"value": column})

        hr = _read(hr_path, "hr")
        eda = _read(eda_path, "eda")

        merged = hr.join(eda[["eda"]], how="outer")
        merged["session"] = merged["session"].ffill().bfill()
        merged.index.name = "timestamp"
        frames[participant_id] = merged

        span = f"{merged.index[0]} -> {merged.index[-1]}" if len(merged) else "empty"
        print(f"[adarp]   {participant_id}: {len(merged):,} rows  {span}", flush=True)

    return frames


# ------------------------------------------------------------------ splits

def _three_way(items, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Cut a time-ordered list into train/val/test, keeping test non-empty.

    With one item there is nothing to hold out and it all goes to train; with two,
    train and test take one each and val goes without, since a missing validation
    block degrades early stopping while a missing test block ends the run.
    """
    items = list(items)
    n = len(items)

    if n == 0:
        return [], [], []
    if n == 1:
        return items, [], []
    if n == 2:
        return items[:1], [], items[1:]

    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))

    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1

    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def _three_way_random(items, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, rng=None):
    """Random split into train/val/test (for cross-validation with different seeds).

    Uses rng to randomly select which items go to each split.
    """
    items = list(items)
    n = len(items)

    if n == 0:
        return [], [], []
    if n == 1:
        return items, [], []
    if n == 2:
        return items[:1], [], items[1:]

    # Randomly shuffle items
    perm = rng.permutation(n)
    shuffled = [items[i] for i in perm]

    # Split randomly
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))

    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1

    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def split_sessions(frame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=None):
    """Assign whole sessions to train/val/test, per participant.

    If seed is None: split in time order (original behavior).
    If seed is provided: split randomly using that seed (for cross-validation).

    Splitting anywhere finer would leak. Windows step 30 s across a 60 s span, so
    neighbours share half their samples; a cut inside a session puts near-copies
    of the same seconds on both sides of it and the test score stops meaning
    anything. A session boundary is the finest cut that is genuinely clean.

    Stress-bearing and pure non-stress sessions are split separately so both reach
    every block. Most participants record stress in a minority of their sessions
    -- 101 in two of eleven -- and one time-ordered cut across all of them would
    hand every positive to train and leave a test set with nothing to detect.

    Returns {participant: {"train": [...], "val": [...], "test": [...]}}.
    """
    splits = {}
    if seed is not None:
        rng = np.random.default_rng(seed)

    for participant_id, block in frame.groupby("user_id", observed=True):
        order = (block.groupby("session")["window_start"].min()
                 .sort_values().index.tolist())
        has_stress = block.groupby("session")["state_val"].max()

        stress_sessions = [s for s in order if has_stress.get(s, 0) == 1]
        plain_sessions = [s for s in order if has_stress.get(s, 0) != 1]

        if seed is None:
            # Time-ordered split (original)
            parts = [_three_way(stress_sessions, train_frac, val_frac),
                     _three_way(plain_sessions, train_frac, val_frac)]
        else:
            # Random split with given seed
            parts = [_three_way_random(stress_sessions, train_frac, val_frac, rng),
                     _three_way_random(plain_sessions, train_frac, val_frac, rng)]

        splits[str(participant_id)] = {
            name: [s for part in parts for s in part[i]]
            for i, name in enumerate(("train", "val", "test"))
        }

    return splits


def usable_targets(frame, splits=None):
    """Participants whose split leaves both classes in train and a positive in test.

    ADARP's usable button presses run from one (107) to fifty-one (112), and a
    participant with two stress events cannot put one in train, one in val and
    one in test. Those participants still belong in the global training pool --
    their windows are real signal -- but they cannot be evaluated against, so
    they are not valid `--participant_id` targets.
    """
    splits = splits if splits is not None else split_sessions(frame)
    usable = []

    for participant_id, split in splits.items():
        block = frame[frame["user_id"] == participant_id]
        train = block[block["session"].isin(split["train"])]
        test = block[block["session"].isin(split["test"])]

        if train["state_val"].nunique() < 2:
            continue
        if test["state_val"].sum() < 1 or (test["state_val"] == 0).sum() < 1:
            continue

        usable.append(participant_id)

    return sorted(usable)


# ------------------------------------------------------------ pipeline entry

def _train_sessions_stream(signals, splits, users, channel):
    """The training-session part of one channel, concatenated across `users`.

    A caveat worth knowing, because nothing raises on it: `_train_or_load_encoder`
    re-windows whatever it is handed with `create_windows`, which slides over the
    value array and never looks at the index. Sessions are concatenated here, so
    each join produces about two windows straddling a gap of hours or days. That
    is well under a percent of the segments a participant contributes and SimCLR
    is robust to a little noise, but it is the reason this stream is not simply
    every session glued together regardless of split.
    """
    parts = []

    for user in users:
        frame = signals.get(user)
        if frame is None:
            continue

        block = frame[frame["session"].isin(splits[user]["train"])]
        block = block[[channel]].rename(columns={channel: "value"}).dropna()

        if not block.empty:
            parts.append(block)

    if not parts:
        raise SystemExit(f"No {channel} signal in any training session.")

    stream = pd.concat(parts).sort_index()
    print(f"[adarp] {channel} training stream: {len(stream):,} samples across "
          f"{len(parts)} participants", flush=True)
    return stream


def prepare_data(args, top_out, shared_enc_root, shared_cnn_root,
                 batch_ssl=32, ssl_epochs=100, pool="global",
                 task="adarp", input_df=None, seed=None,
                 processed_dir=PROCESSED_DIR):
    """Build ADARP's train/val/test frames and its frozen encoders.

    Mirrors the return signature of `new_prep.prepare_data` so `run_adarp.py` can
    drive the shared active-learning code without touching it.

    `batch_ssl` and `ssl_epochs` are passed through to `_train_or_load_encoder`,
    which falls back to `src.compare_pipelines`' 32/100 when they are None. They
    travel no further than ADARP's own encoders, so raising them here leaves the
    BP and WESAD branches on their original values.

    Returns the eleven-tuple the pipeline unpacks: df_tr, df_all_tr, df_val,
    df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d,
    all_negatives.
    """
    if input_df is None:
        input_df = getattr(args, "input_df", "raw")
    if input_df != "raw":
        raise NotImplementedError("The ADARP branch only supports input_df='raw'.")

    # Imported here, not at module scope: everything above this section is plain
    # numpy/pandas/scipy, and building windows should not have to import TensorFlow.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.compare_pipelines import _train_or_load_encoder

    user_root = Path(top_out) / str(args.user) / f"{args.fruit}_{args.scenario}"
    out_dir = user_root / pool
    models_d = out_dir / "models_saved"
    results_d = out_dir / "results"
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    eda, hr, meta = load_windows(processed_dir)
    frame = windows_to_frame(eda, hr, meta)
    splits = split_sessions(frame, seed=seed)

    uid = str(args.user)
    if uid not in splits:
        raise SystemExit(
            f"Participant {uid} has no windows. Present: {sorted(splits)}"
        )

    usable = usable_targets(frame, splits)
    if uid not in usable:
        raise SystemExit(
            f"Participant {uid} has too few stress events to be a target: the split "
            f"cannot give train both classes and test a positive. Usable targets are "
            f"{usable}. {uid} still contributes to the global pool."
        )

    excluded = {
        u.strip() for u in os.environ.get("BAN_AL_EXCLUDE_USERS", "").split(",") if u.strip()
    }
    if excluded:
        print(f"[adarp-exclude] BAN_AL_EXCLUDE_USERS active; dropped: {sorted(excluded)}")

    contributors = [u for u in sorted(splits) if u not in excluded]
    if pool == "personal":
        contributors = [uid]
    elif uid not in contributors:
        raise SystemExit(f"Target {uid} is in BAN_AL_EXCLUDE_USERS; it cannot also be the target.")

    def _block(user, which):
        block = frame[frame["user_id"] == user]
        return block[block["session"].isin(splits[user][which])]

    df_tr = _block(uid, "train").reset_index(drop=True)
    df_te = _block(uid, "test").reset_index(drop=True)

    if pool == "personal":
        df_all_tr = None
        df_val = _block(uid, "val").reset_index(drop=True)
    elif pool == "global":
        df_all_tr = pd.concat(
            [_block(u, "train") for u in contributors], ignore_index=True
        )
        df_val = pd.concat(
            [_block(u, "val") for u in contributors], ignore_index=True
        )
    else:
        raise ValueError(f"Unknown pool: {pool!r}")

    if df_te.empty:
        raise SystemExit(f"Target {uid} has no test windows after the session split.")

    # ---- encoders, over training sessions only ------------------------------
    signals = load_signals(processed_dir)
    hr_stream = _train_sessions_stream(signals, splits, contributors, "hr")
    eda_stream = _train_sessions_stream(signals, splits, contributors, "eda")

    if pool == "personal":
        enc_hr_path = models_d / "hr_encoder.keras"
        enc_st_path = models_d / "steps_encoder.keras"
    else:
        shared_enc_root = Path(shared_enc_root)
        shared_enc_root.mkdir(parents=True, exist_ok=True)
        enc_hr_path = shared_enc_root / "adarp_hr_encoder.keras"
        enc_st_path = shared_enc_root / "adarp_steps_encoder.keras"

    # The stream is already restricted to training sessions, so passing every date
    # it holds makes _train_or_load_encoder's date mask a no-op. Masking on dates
    # would not work here anyway: participants record several sessions a day, and
    # 101 has a train session and a held-out one both on 2019-04-30.
    print(f"[adarp] hr encoder -> {enc_hr_path}", flush=True)
    enc_hr = _train_or_load_encoder(
        enc_hr_path, "hr", hr_stream, sorted(set(hr_stream.index.date)), results_d,
        batch_ssl=batch_ssl, ssl_epochs=ssl_epochs,
    )
    print(f"[adarp] eda encoder -> {enc_st_path}", flush=True)
    enc_st = _train_or_load_encoder(
        enc_st_path, "steps", eda_stream, sorted(set(eda_stream.index.date)), results_d,
        batch_ssl=batch_ssl, ssl_epochs=ssl_epochs,
    )

    if pool != "personal":
        import shutil
        for src, dst in [(enc_hr_path, models_d / "hr_encoder.keras"),
                         (enc_st_path, models_d / "steps_encoder.keras")]:
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    def _dates(user, which):
        block = _block(user, which)
        return sorted({t.date() for t in block["window_start"]})

    all_splits = {
        u: (_dates(u, "train"), _dates(u, "val"), _dates(u, "test"))
        for u in contributors
    }

    pool_rows = len(df_all_tr) if df_all_tr is not None else len(df_tr)
    print(
        f"[adarp] target={uid} pool={pool} "
        f"train={len(df_tr)} pool_train={pool_rows} val={len(df_val)} test={len(df_te)} "
        f"test_pos={int(df_te['state_val'].sum())} "
        f"events_in_pool={frame.loc[frame['event_id'] != 'none', 'event_id'].nunique()}"
    )

    return (
        df_tr,
        df_all_tr,
        df_val,
        df_te,
        enc_hr,
        enc_st,
        user_root,
        all_splits,
        models_d,
        results_d,
        {},
    )
