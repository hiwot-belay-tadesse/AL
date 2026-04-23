#!/usr/bin/env python3
"""Plot aggregated final ROC/AUC for predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utility import bootstrap_auc  # noqa: E402

BANAWARE_SCENARIOS = [
    "Melon_Crave",
    "Melon_Use",
    "Nectarine_Crave",
    "Nectarine_Use",
    "Almond_Crave",
    "Almond_Use",
    "Carrot_Crave",
    "Carrot_Use",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregated ROC/AUC for Cardiomate or Banaware. "
            "Banaware mode writes one plot per scenario."
        ),
    )
    parser.add_argument(
        "--data",
        choices=("cardiomate", "banaware"),
        default="cardiomate",
        help="Dataset family to aggregate.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional dataset root directory override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Cardiomate: output PNG path. "
            "Banaware: output directory for scenario-level PNGs."
        ),
    )
    parser.add_argument(
        "--pool",
        default="global_supervised",
    )
    return parser.parse_args()


def default_root(data: str) -> Path:
    return Path("BP_SPIKE_PRED") if data == "cardiomate" else Path("BANAWARE_PRED")


def discover_cardiomate_prob_files(root: Path, pool: str) -> list[Path]:
    return sorted(root.glob(f"**/BP_spike/{pool}/results/test_probs_*.npy"))


def discover_banaware_prob_files(root: Path, scenario: str, pool: str) -> list[Path]:
    return sorted(root.glob(f"**/{scenario}/{pool}/results/test_probs_*.npy"))


def labels_path_for(probs_path: Path) -> Path:
    suffix = probs_path.stem.replace("test_probs_", "", 1)
    return probs_path.with_name(f"test_labels_{suffix}.npy")


def load_aggregated_arrays(prob_files: list[Path]) -> tuple[np.ndarray, np.ndarray, int, int]:
    y_all = []
    p_all = []
    used = 0
    skipped = 0

    for probs_path in prob_files:
        labels_path = labels_path_for(probs_path)
        if not labels_path.exists():
            skipped += 1
            print(f"[WARN] Missing labels for: {probs_path}")
            continue

        try:
            probs = np.load(probs_path).ravel()
            labels = np.load(labels_path).ravel()
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Failed to load {probs_path}: {exc}")
            continue

        if len(probs) == 0 or len(labels) == 0 or len(probs) != len(labels):
            skipped += 1
            print(f"[WARN] Bad arrays in {probs_path}: probs={len(probs)}, labels={len(labels)}")
            continue

        p_all.append(probs)
        y_all.append(labels)
        used += 1

    if not y_all:
        raise RuntimeError("No valid prediction/label pairs found.")

    return np.concatenate(y_all), np.concatenate(p_all), used, skipped


def plot_roc(y: np.ndarray, p: np.ndarray, title: str, legend: str, out_path: Path) -> tuple[float, float]:
    auc_mean, auc_std, _ = bootstrap_auc(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=legend)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return auc_mean, auc_std


def run_cardiomate(root: Path, output: Path | None, pool: str) -> int:
    prob_files = discover_cardiomate_prob_files(root, pool)

    if not prob_files:
        raise SystemExit(f"No probability files found under {root}")

    y, p, used, skipped = load_aggregated_arrays(prob_files)
    out_path = output.resolve() if output else Path(f"supp_plots/cardiomate/{pool}/cardiomate_{pool}_roc_curve_aggregated.png").resolve()

    auc_mean, auc_std, _ = bootstrap_auc(y, p)
    legend = f"BP_spike (AUC = {auc_mean:.3f} +/- {auc_std:.3f})"

    _, _ = plot_roc(
        y,
        p,
        title="Aggregated ROC Curve ({pool}) - cardiomate",
        legend=legend,
        out_path=out_path,
    )

    print(f"Saved ROC plot: {out_path}")
    print(f"Used {used} files, skipped {skipped} files.")
    print(f"Final AUC: {auc_mean:.3f} +/- {auc_std:.3f}")
    return 0


def run_banaware(root: Path, output: Path | None, pool: str) -> int:
    out_dir = output.resolve() if output else Path("supp_plots/banaware").resolve()

    for scenario in BANAWARE_SCENARIOS:
        prob_files = discover_banaware_prob_files(root, scenario, pool)
        if not prob_files:
            print(f"[WARN] No probability files found for {scenario} under {root}")
            continue

        y, p, used, skipped = load_aggregated_arrays(prob_files)
        out_path = out_dir / f"{pool}/{scenario.lower()}_{pool}_roc_curve_aggregated.png"
        auc_mean, auc_std, _ = bootstrap_auc(y, p)
        legend = f"{scenario} (AUC = {auc_mean:.3f} +/- {auc_std:.3f})"
        _, _ = plot_roc(
            y,
            p,
            title=f"Aggregated ROC Curve ({pool}) - {scenario}",
            legend=legend,
            out_path=out_path,
        )

        print(f"Saved ROC plot: {out_path}")
        print(f"{scenario}: used {used} files, skipped {skipped} files.")
        print(f"{scenario} Final AUC: {auc_mean:.3f} +/- {auc_std:.3f}")

    return 0


def main() -> int:
    args = parse_args()
    root = (args.root if args.root is not None else default_root(args.data)).resolve()

    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    if args.data == "cardiomate":
        return run_cardiomate(root, args.output, args.pool)
    return run_banaware(root, args.output, args.pool)


if __name__ == "__main__":
    raise SystemExit(main())
