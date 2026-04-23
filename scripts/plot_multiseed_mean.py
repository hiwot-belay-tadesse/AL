from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_seed_curve(seed_dir: Path) -> pd.DataFrame | None:
    csvs = sorted(seed_dir.rglob("al_progress.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    if "round" not in df.columns:
        return None
    auc_col = "AUC_Mean" if "AUC_Mean" in df.columns else "auc_mean" if "auc_mean" in df.columns else None
    if auc_col is None:
        return None
    out = df[["round", auc_col]].copy().rename(columns={auc_col: "auc"})
    return out


def _read_upper_bound(seed_dir: Path) -> float | None:
    p = seed_dir.rglob("upper_bound_auc.npy")
    p = sorted(p)
    if not p:
        return None
    try:
        import numpy as np
        return float(np.load(p[0]))
    except Exception:
        return None


def plot_mean_across_seed_pairs(
    root_dir: str,
    pool: str,
    user: str,
    fruit: str,
    scenario: str,
    methods: tuple[str, ...] = ("random", "coreset"),
    out_path: str | None = None,
) -> Path:
    """
    Read per-seed AL runs for each method, keep only shared seeds across methods,
    compute mean AUC per round, and plot.
    """
    scenario_dir = Path(root_dir) / pool / user / f"{fruit}_{scenario}"

    method_seed_curves: dict[str, dict[int, pd.DataFrame]] = {}
    for method in methods:
        mdir = scenario_dir / method
        seed_map: dict[int, pd.DataFrame] = {}
        for sdir in sorted(mdir.glob("seed_*")):
            try:
                seed = int(sdir.name.split("_", 1)[1])
            except Exception:
                continue
            curve = _read_seed_curve(sdir)
            if curve is not None:
                seed_map[seed] = curve
        method_seed_curves[method] = seed_map

    if not method_seed_curves:
        raise FileNotFoundError(f"No method folders found under: {scenario_dir}")

    seed_sets = [set(v.keys()) for v in method_seed_curves.values() if v]
    if not seed_sets:
        raise FileNotFoundError(f"No seed runs with al_progress.csv under: {scenario_dir}")
    common_seeds = sorted(set.intersection(*seed_sets))
    if not common_seeds:
        raise FileNotFoundError("No shared seeds found across methods.")

    plt.figure(figsize=(8, 4))
    colors = {"random": "darkorange", "coreset": "seagreen", "uncertainty": "tab:blue"}
    upper_vals = []

    for method in methods:
        rows = []
        for seed in common_seeds:
            df = method_seed_curves[method].get(seed)
            if df is None:
                continue
            d = df.copy()
            d["seed"] = seed
            rows.append(d)
            ub = _read_upper_bound(scenario_dir / method / f"seed_{seed}")
            if ub is not None:
                upper_vals.append(ub)
        if not rows:
            continue
        stacked = pd.concat(rows, ignore_index=True)
        agg = stacked.groupby("round", as_index=False)["auc"].agg(["mean", "std"]).reset_index()
        agg.columns = ["round", "auc_mean", "auc_std"]
        plt.plot(
            agg["round"],
            agg["auc_mean"],
            marker="o",
            linewidth=1.8,
            label=f"{method} mean",
            color=colors.get(method, None),
        )
        plt.fill_between(
            agg["round"],
            agg["auc_mean"] - agg["auc_std"].fillna(0.0),
            agg["auc_mean"] + agg["auc_std"].fillna(0.0),
            alpha=0.18,
            color=colors.get(method, None),
        )

    if upper_vals:
        upper_mean = float(sum(upper_vals) / len(upper_vals))
        plt.axhline(
            upper_mean,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"100% AUC = {upper_mean:.3f}",
        )

    plt.xlabel("Round")
    plt.ylabel("AUC Mean")
    plt.title(f"Mean AUC Across Shared Seeds ({user} {fruit}_{scenario})")
    plt.legend()
    plt.tight_layout()

    out_p = Path(out_path) if out_path else (scenario_dir / "auc_mean_across_seed_pairs.png")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150)
    plt.close()
    return out_p


if __name__ == "__main__":
    # Edit these and run: python3 scripts/plot_multiseed_mean.py
    ROOT_DIR = "/Users/hiwotbelaytadesse/Desktop/Banaware_AL/multi_seed_results"
    POOL = "global"
    USER = "30"
    FRUIT = "BP"
    SCENARIO = "spike"
    METHODS = ("random", "coreset")

    out = plot_mean_across_seed_pairs(
        root_dir=ROOT_DIR,
        pool=POOL,
        user=USER,
        fruit=FRUIT,
        scenario=SCENARIO,
        methods=METHODS,
    )
    print(f"Saved: {out}")
