"""Rebuild the ADARP window arrays and export the continuous signal.

Both outputs are prerequisites for an active-learning run, and they are not
interchangeable:

    processed/adarp_windows.{npz,csv}       the labelled windows the classifier reads
    processed/signals/<pid>_{hr,eda}.csv    the stream the SSL encoders train on

The windows are overlapping, cover only the labelled intervals, and have the
non-stress side sampled down. Contrastive pre-training wants all of the
unlabelled signal, so it reads the second set, not the first.

    python ADARP/build_adarp_dataset.py
    python ADARP/build_adarp_dataset.py --nonstress_ratio none   # keep the whole pool
    python ADARP/build_adarp_dataset.py --eda_normalize none --skip_signals \
        --processed_dir ADARP/processed_raw_eda   # raw microsiemens, windows only
"""

import argparse
import sys
from pathlib import Path

_ADARP_DIR = Path(__file__).resolve().parent
if str(_ADARP_DIR) not in sys.path:
    sys.path.insert(0, str(_ADARP_DIR))

import load_adarp as adarp
import preprocess_adarp_data as prep


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--nonstress_ratio", default=str(prep.NONSTRESS_RATIO),
        help="Non-stress windows kept per stress window, or 'none' for the whole pool. "
             "The AL study is about finding rare positives in a large pool, so a ratio "
             "here pre-removes the difficulty being measured.",
    )
    parser.add_argument(
        "--eda_normalize", default="segment", choices=["segment", "window", "none"],
        help="How prepare_eda scales EDA. 'segment' is the reference, but a stress "
             "segment is a fixed 40 min while a non-stress one runs for hours, so "
             "its min/max differ systematically by class; 'none' keeps raw "
             "microsiemens and removes that label-dependent scaling entirely.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Fixes the subsample draw.")
    parser.add_argument("--processed_dir", type=Path, default=prep.PROCESSED_DIR)
    parser.add_argument("--skip_windows", action="store_true")
    parser.add_argument("--skip_signals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ratio = None if str(args.nonstress_ratio).lower() in ("none", "null", "") \
        else float(args.nonstress_ratio)

    print("loading every participant (this reads the whole recording)...")
    df = adarp.load_all_participants()
    print(f"  {len(df):,} rows, {df['participant'].nunique()} participants, "
          f"{df['session'].nunique()} sessions")

    if not args.skip_windows:
        tags_by_participant = {
            participant.rstrip("C"): adarp.load_tags(participant)
            for participant in adarp.participant_ids()
        }

        normalize = None if args.eda_normalize == "none" else args.eda_normalize
        print(f"  eda normalize={normalize!r}")
        eda, hr, meta, per_participant = prep.build_dataset(
            df, tags_by_participant, nonstress_ratio=ratio, seed=args.seed,
            normalize=normalize,
        )

        stem = args.processed_dir / prep.WINDOWS_STEM
        npz_path, csv_path = prep.save_windows(stem, eda, hr, meta)

        prep.validation_table(per_participant).to_csv(
            args.processed_dir / "validation_table.csv")
        prep.sampling_table(meta, per_participant).to_csv(
            args.processed_dir / "sampling_table.csv")

        stress = int((meta["label"] == 1).sum())
        print(f"windows: {len(meta):,} ({stress:,} stress, {len(meta) - stress:,} non-stress) "
              f"ratio={ratio}")
        print(f"  {npz_path}  eda={eda.shape} hr={hr.shape}")
        print(f"  {csv_path}")

    if not args.skip_signals:
        written = prep.export_signals(df, processed_dir=args.processed_dir)
        print(f"signals: {len(written)} participants -> "
              f"{args.processed_dir / prep.SIGNAL_SUBDIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
