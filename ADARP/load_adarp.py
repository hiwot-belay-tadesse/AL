"""Load the ADARP Empatica E4 recordings from DATA/Sensor Data.

Layout: DATA/Sensor Data/Part <pid>C/<device>_<yymmdd-HHMMSS>/{EDA,HR,tags}.csv

The signal CSVs carry two header rows -- the session start time (unix seconds,
UTC) and the sample rate in Hz -- then one sample per row. Timestamps are
reconstructed from those two numbers rather than stored per row.

This module only reads from disk. Everything that derives labels, segments or
windows from what it returns lives in `preprocess_adarp_data`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_DIR = Path(__file__).parent / "DATA" / "Sensor Data"

EDA_HZ = 4
HR_HZ = 1


def participant_ids(sensor_dir=SENSOR_DIR):
    """The folder ids, "101C" .. "112C", in sorted order."""
    return [
        p.name.removeprefix("Part ").strip()
        for p in sorted(Path(sensor_dir).glob("Part *"))
        if p.is_dir()
    ]


def participant_dir(participant, sensor_dir=SENSOR_DIR):
    """Folder for one participant, id given either as "101" or "101C"."""
    return Path(sensor_dir) / f"Part {str(participant).rstrip('C')}C"


def session_dirs(participant, sensor_dir=SENSOR_DIR):
    """Every recording session folder for one participant, in time order."""
    return sorted(
        p for p in participant_dir(participant, sensor_dir).iterdir() if p.is_dir()
    )


def read_signal(path, column):
    """Raw E4 CSV -> Series indexed by UTC time (row 0 = start, row 1 = rate)."""
    raw = pd.read_csv(path, header=None).iloc[:, 0].astype(float)

    # a header-only file describes a session that recorded nothing
    if len(raw) < 3:
        return pd.Series(
            dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=column
        )

    start = raw.iloc[0]
    rate = raw.iloc[1]
    values = raw.iloc[2:].to_numpy()

    index = pd.to_datetime(start + np.arange(len(values)) / rate, unit="s", utc=True)

    return pd.Series(values, index=index, name=column)


def load_session(session_dir):
    """One session as a UTC-indexed eda/hr frame, trimmed to their overlap.

    EDA is 4 Hz and HR 1 Hz, so the outer join leaves `hr` NaN off its own grid;
    callers that need one modality drop the NaNs for it.
    """
    session_dir = Path(session_dir)

    eda = read_signal(session_dir / "EDA.csv", "eda")
    hr = read_signal(session_dir / "HR.csv", "hr")

    if eda.empty or hr.empty:
        return pd.DataFrame(
            columns=["eda", "hr"], index=pd.DatetimeIndex([], tz="UTC")
        )

    # HR is derived over a startup buffer, so it begins ~10 s after EDA
    common_start = max(eda.index[0], hr.index[0])
    common_end = min(eda.index[-1], hr.index[-1])

    eda = eda.loc[common_start:common_end]
    hr = hr.loc[common_start:common_end]

    return pd.concat([eda, hr], axis=1, join="outer")


def load_participant(participant, sensor_dir=SENSOR_DIR):
    """Every session for one participant as one UTC-indexed frame."""
    frames = []

    for path in session_dirs(participant, sensor_dir):
        session_df = load_session(path)

        if session_df.empty:
            continue  # header rows but no samples (e.g. 106C A02160_191205-233346)

        session_df["session"] = path.name
        frames.append(session_df)

    participant_df = pd.concat(frames).sort_index()
    participant_df.insert(0, "participant", str(participant).rstrip("C"))
    participant_df.index.name = "timestamp"
    return participant_df


def load_all_participants(participants=None, sensor_dir=SENSOR_DIR):
    """All participants in one frame (~24M rows, so the labels stay categorical)."""
    if participants is None:
        participants = participant_ids(sensor_dir)

    df = pd.concat(
        load_participant(p, sensor_dir) for p in participants
    ).sort_index()

    df.index.name = "timestamp"
    df["participant"] = df["participant"].astype("category")
    df["session"] = df["session"].astype("category")
    return df


def load_tags(participant, sensor_dir=SENSOR_DIR):
    """Every E4 button press for one participant, one row per press.

    The audit -- which sessions held no tag at all -- rides along in `.attrs`
    rather than being printed from a loader.
    """
    rows = []
    empty_sessions = []

    for path in session_dirs(participant, sensor_dir):
        tag_path = path / "tags.csv"

        # 49 of 237 sessions have a 0-byte tags.csv: the button was never
        # pressed. read_csv raises EmptyDataError on those, so check first.
        if not tag_path.exists() or tag_path.stat().st_size == 0:
            empty_sessions.append(path.name)
            continue

        tags = pd.read_csv(tag_path, header=None)

        # a row that is blank or non-numeric is not a timestamp
        times = pd.to_numeric(tags.iloc[:, 0], errors="coerce").dropna()

        if times.empty:
            empty_sessions.append(path.name)
            continue

        for ts in times:
            rows.append({
                "participant": str(participant).rstrip("C"),
                "session": path.name,
                "tag_time": pd.to_datetime(float(ts), unit="s", utc=True),
            })

    tags_df = pd.DataFrame(rows, columns=["participant", "session", "tag_time"])
    tags_df = tags_df.sort_values("tag_time").reset_index(drop=True)

    tags_df.attrs["empty_sessions"] = empty_sessions
    tags_df.attrs["sessions_with_tags"] = tags_df["session"].nunique()
    return tags_df


def load_all_tags(participants=None, sensor_dir=SENSOR_DIR):
    """Button presses for every participant in one frame."""
    if participants is None:
        participants = participant_ids(sensor_dir)

    return pd.concat(
        [load_tags(p, sensor_dir) for p in participants], ignore_index=True
    )
