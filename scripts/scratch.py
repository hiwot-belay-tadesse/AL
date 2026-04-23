"""Collect and move pool-specific loss plots into one folder.

Examples:
    python scripts/scratch.py --data banaware --pool global_supervised
    python scripts/scratch.py --data cardiomate --pool personal_ssl --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

POOL_TO_FILENAME = {
    "global_supervised": "global_supervised_loss.png",
    "personal_ssl": "personal_ssl_clf_loss.png",
    "global_ssl": "global_ssl_clf_loss.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move pool-specific loss curves into one destination folder.",
    )
    parser.add_argument(
        "--data",
        required=True,
        choices=("banaware", "cardiomate"),
        help="Dataset family used to resolve source and destination paths.",
    )
    parser.add_argument(
        "--pool",
        required=True,
        choices=tuple(POOL_TO_FILENAME.keys()),
        help="Pool to collect loss plots from.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root containing BANAWARE_PRED and BP_SPIKE_PRED.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Optional destination folder. Defaults to dataset-specific loss_curves/<pool>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    return parser.parse_args()


def default_paths(base_dir: Path, data: str, pool: str) -> tuple[Path, Path]:
    if data == "banaware":
        root = base_dir / "BANAWARE_PRED"
    else:
        root = base_dir / "BP_SPIKE_PRED"
    dest = root / "loss_curves" / pool
    return root, dest


def discover_source_files(root: Path, pool: str) -> list[Path]:
    filename = POOL_TO_FILENAME[pool]
    files = sorted(root.glob(f"**/{pool}/results/{filename}"))

    # Global CNN loss is stored without the intermediate results folder.
    if pool == "global_supervised":
        files.extend(sorted(root.glob("**/global_cnns/*/global_supervised_loss.png")))

    return sorted(set(files))


def unique_dest_path(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while True:
        alt = dest_dir / f"{stem}_{idx}{suffix}"
        if not alt.exists():
            return alt
        idx += 1


def infer_name(src: Path, dataset: str, pool: str) -> str:
    base_name = POOL_TO_FILENAME[pool]
    parts = src.parts

    if "global_cnns" in parts:
        i = parts.index("global_cnns")
        if i + 1 < len(parts):
            return f"global_cnns_{parts[i + 1]}_{base_name}"
        return f"global_cnns_{base_name}"

    if dataset == "cardiomate":
        for p in parts:
            if p.isdigit():
                return f"{p}_{base_name}"
        return f"{src.parent.name}_{base_name}"

    uid = None
    task = None
    for i, p in enumerate(parts):
        if re.fullmatch(r"ID\d+", p):
            uid = p
            if i + 1 < len(parts):
                task = parts[i + 1]
            break

    if uid and task:
        return f"{uid}_{task}_{base_name}"
    if uid:
        return f"{uid}_{base_name}"
    return f"{src.parent.name}_{base_name}"


def main() -> int:
    args = parse_args()
    root, default_dest = default_paths(args.base_dir, args.data, args.pool)
    dest_dir = args.dest if args.dest is not None else default_dest

    root = root.resolve()
    dest_dir = dest_dir.resolve()

    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    source_files = discover_source_files(root, args.pool)

    moved = 0
    skipped = 0
    for src in source_files:
        if dest_dir in src.parents:
            skipped += 1
            continue

        target_name = infer_name(src, args.data, args.pool)
        dst = unique_dest_path(dest_dir, target_name)
        print(f"{src} -> {dst}")

        if not args.dry_run:
            shutil.move(str(src), str(dst))
        moved += 1

    action = "Would move" if args.dry_run else "Moved"
    print(f"{action} {moved} file(s). Skipped {skipped} file(s) already in destination tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
