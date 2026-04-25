
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def resolve_lr_check_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path("Cardiomate_AL/GS/LR_check"),
        Path("Banaware_AL/Cardiomate_AL/GS/"),
        here / "Cardiomate_AL/GS/LR_check",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    discovered = sorted(h.parent for h in here.rglob("aggregate_full_data_auc.png"))
    breakpoint()
    if discovered:
        return discovered[0]

    discovered = sorted(h.parent for h in here.rglob("aggregate_auc.png"))
    if discovered:
        return discovered[0]

    return candidates[0]


def plot_grid(
    root: Path,
    cols: int = 4,
    out_name: str = "GS_grid_seed_comparison.png",

):
    images: list[Path] = []

    id_dirs = sorted(
        [directory for directory in root.iterdir() if directory.is_dir() and directory.name.isdigit()],
        key=lambda directory: int(directory.name),
    )

    for id_dir in id_dirs:
        expected = id_dir / "BP_spike" / "fixed_seed_10pct" / "seed_comparison.png"

        if expected.exists():
            images.append(expected)
            continue

        fallback = sorted(id_dir.glob("**/seed_comparison.png"))
        if fallback:
            images.append(fallback[0])

    # images.extend(sorted(root.glob("aggregate_full_data_auc_*.png")))
    images.extend(sorted(root.glob("aggregate_full_data_auc.png")))  ## hardcoded for now
    images.extend(sorted(root.glob('aggregate_10pct_seed_42_BP_spike.png'))) ## hardcoded for now
    images = sorted(set(images))

    if not images:
        raise FileNotFoundError(f"No matching images under {root}")

    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]


    for ax, img_path in zip(axes, images):
        ax.imshow(mpimg.imread(img_path))
        if img_path.name == "seed_comparison.png":
            ax.set_title(img_path.parent.name)

        else:  
            ax.axis("off")
            # ax.set_title(img_path.stem)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle("Global Supervised Seed Comparison Grid", fontsize=14)
    fig.tight_layout()
    out_path = root / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    lr_check_root = resolve_lr_check_root()
    plot_grid(lr_check_root)
