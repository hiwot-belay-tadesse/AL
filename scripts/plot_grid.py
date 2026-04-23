#!/usr/bin/env python3
"""Build a single titled grid from plot images in a folder."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one grid figure from plot images (default: first 5 files)."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing plot images.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of images to include from sorted files. Default: 5",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of grid columns. Default: 3",
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help="Optional figure-level title (suptitle).",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("W", "H"),
        help="Optional full figure size in inches. Default scales like sample code.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI. Default: 150")
    parser.add_argument(
        "--titles",
        nargs="+",
        default=None,
        metavar="TITLE",
        help="Optional explicit subplot titles. Must match selected image count.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
        metavar="EXT",
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--filename-regex",
        default="^ID",
        help="Regex filter applied to filename stem. Default: ^ID",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Search input directory recursively (default).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Search only top-level input directory.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "matplotlib", "pillow"),
        default="auto",
        help="Rendering backend. Default: auto",
    )
    return parser.parse_args()


def normalize_extensions(extensions: list[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def discover_images(
    input_dir: Path,
    extensions: set[str],
    count: int,
    recursive: bool,
    filename_regex: str,
    output_name: str,
) -> list[Path]:
    pattern = re.compile(filename_regex) if filename_regex else None

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = []
    for p in iterator:
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        if p.name == output_name:
            continue
        stem = p.stem
        if pattern and not pattern.search(stem):
            continue
        images.append(p)

    images = sorted(images)
    if len(images) < count:
        raise ValueError(
            f"Expected at least {count} matching images in {input_dir}, found {len(images)}."
        )
    return images[:count]


def infer_titles(paths: list[Path]) -> list[str]:
    return [path.stem.split("_")[0] for path in paths]


def resolve_titles(paths: list[Path], titles: list[str] | None) -> list[str]:
    if titles is None:
        return infer_titles(paths)
    if len(titles) != len(paths):
        raise ValueError(
            f"--titles count ({len(titles)}) must match selected image count ({len(paths)})."
        )
    return titles


def compute_figsize(nrows: int, ncols: int, requested: tuple[float, float] | None) -> tuple[float, float]:
    if requested is not None:
        return requested
    return (4.2 * ncols, 3.2 * nrows)


def plot_with_matplotlib(
    image_paths: list[Path],
    titles: list[str],
    ncols: int,
    output_path: Path,
    scenario_name: str | None,
    figsize_arg: tuple[float, float] | None,
    dpi: int,
) -> None:
    import numpy as np
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = len(image_paths)
    nrows = int(math.ceil(n / ncols))
    figsize = compute_figsize(nrows, ncols, figsize_arg)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, ax in enumerate(axes.flatten()):
        if idx >= n:
            ax.axis("off")
            continue

        image = mpimg.imread(image_paths[idx])
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(titles[idx], fontsize=10)

    if scenario_name:
        fig.suptitle(scenario_name.replace("_", " "), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_with_pillow(
    image_paths: list[Path],
    titles: list[str],
    ncols: int,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    n = len(image_paths)
    nrows = int(math.ceil(n / ncols))

    images = [Image.open(p).convert("RGB") for p in image_paths]
    min_w = min(img.width for img in images)
    min_h = min(img.height for img in images)
    resized = [img.resize((min_w, min_h)) for img in images]

    title_space = 40
    cell_h = min_h + title_space
    canvas = Image.new("RGB", (min_w * ncols, cell_h * nrows), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, img in enumerate(resized):
        row = idx // ncols
        col = idx % ncols
        x = col * min_w
        y = row * cell_h
        canvas.paste(img, (x, y + title_space))
        draw.text((x + 8, y + 10), titles[idx], fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.ncols < 1:
        raise ValueError("--ncols must be >= 1")

    output_path = Path(args.output)

    image_paths = discover_images(
        input_dir=input_dir,
        extensions=normalize_extensions(args.extensions),
        count=args.count,
        recursive=args.recursive,
        filename_regex=args.filename_regex,
        output_name=output_path.name,
    )
    titles = resolve_titles(image_paths, args.titles)
    figsize_arg = tuple(args.figsize) if args.figsize else None

    if args.backend in ("auto", "matplotlib"):
        try:
            plot_with_matplotlib(
                image_paths=image_paths,
                titles=titles,
                ncols=args.ncols,
                output_path=output_path,
                scenario_name=args.scenario_name,
                figsize_arg=figsize_arg,
                dpi=args.dpi,
            )
            print(f"Saved grid to {output_path} (backend: matplotlib)")
            return
        except ModuleNotFoundError:
            if args.backend == "matplotlib":
                raise

    if args.backend in ("auto", "pillow"):
        try:
            plot_with_pillow(
                image_paths=image_paths,
                titles=titles,
                ncols=args.ncols,
                output_path=output_path,
            )
            print(f"Saved grid to {output_path} (backend: pillow)")
            return
        except ModuleNotFoundError:
            if args.backend == "pillow":
                raise

    raise RuntimeError(
        "No supported backend found. Install one of: `pip install matplotlib` or `pip install pillow`."
    )


if __name__ == "__main__":
    main()
