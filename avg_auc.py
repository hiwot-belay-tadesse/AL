"""
Run run.py across seeds, then report mean/std AUC per AL round.

Assumption: run.py now writes one run-level AUC value for each active-learning
round. This script runs those single-seed experiments and computes the mean and
std across seeds for each round.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


METHOD_COLORS = {
    "random": "#ff7f0e",
    "coreset": "#2ca02c",
    "uncertainty": "#1f77b4",
}


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_user_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item for item in re.split(r"[,\s]+", str(value).strip()) if item]


def configure_matplotlib_cache() -> None:
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "avg_auc_matplotlib_cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def submit_seed_job(repo_root, output_dir, template, exp_name, exp_dir, exp_kwargs):
    exp_dir_path = Path(exp_dir)
    mkdir_exp_dir = exp_dir_path if exp_dir_path.is_absolute() else repo_root / exp_dir_path
    mkdir_exp_dir.mkdir(parents=True, exist_ok=True)

    script = template.replace("# python -u run.py EXPDIR EXPNAME KWARGS\n", "")
    script = script.replace(
        "python -u refactor_run.py EXPDIR EXPNAME KWARGS",
        'cd "REPOROOT"\nexport BAN_AL_OUTPUT_DIR="OUTPUTDIR"\npython -u run.py "EXPDIR" "EXPNAME" KWARGS',
    )
    script = script.replace("REPOROOT", str(repo_root))
    script = script.replace("OUTPUTDIR", output_dir)
    script = script.replace("EXPNAME", exp_name)
    script = script.replace("EXPDIR", str(exp_dir))
    script = script.replace("KWARGS", "'{}'".format(json.dumps(exp_kwargs)))

    tmp_dir = repo_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_key = "_".join(
        [
            exp_name,
            str(exp_kwargs.get("user", "user")),
            str(exp_kwargs.get("seed", "noseed")),
        ]
    )
    script_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", script_key)
    script_path = tmp_dir / f"slurm_avg_auc_{script_key}.sh"
    script_path.write_text(script)

    try:
        ret = subprocess.call(["sbatch", str(script_path)])
    except FileNotFoundError as exc:
        raise SystemExit("sbatch was not found. Run with --local for sequential local execution.") from exc
    if ret != 0:
        print(f"Error code {ret} when submitting job for {script_path}")


def discover_bp_candidate_users() -> list[str]:
    """All BP user IDs with the three required files present on disk.

    Used as the universe of candidate users when deriving target cohort from
    exclusion logic (so the target list equals 'candidates - excluded').
    """
    bp_root = Path("DATA/Cardiomate")
    candidate_dirs = (
        sorted((bp_root / "hp").glob("hp*"))
        if (bp_root / "hp").exists()
        else sorted(bp_root.glob("hp*"))
    )
    candidates: list[str] = []
    for p in candidate_dirs:
        m = re.search(r"\d+", p.name)
        if not m:
            continue
        pid = m.group(0)
        required = [
            p / f"hp{pid}_hr.csv",
            p / f"hp{pid}_steps.csv",
            p / f"blood_pressure_readings_ID{pid}_cleaned.csv",
        ]
        if not all(path.exists() for path in required):
            continue
        candidates.append(pid)
    return candidates


def discover_invalid_users(seeds: list[int], task: str, input_df: str) -> set[str]:
    """For BP task: load each candidate user once, then try ensure_train_val_test_days
    across every seed. Any user that raises for any seed is marked invalid.

    Returns the union of users to exclude. Skips silently for non-BP tasks.
    """
    if task != "bp":
        return set()

    from new_prep import _bp_load_all
    from src.compare_pipelines import (
        ensure_train_val_test_days,
        ensure_train_val_test_days_retry,
    )

    bp_root = Path("DATA/Cardiomate")
    candidate_dirs = (
        sorted((bp_root / "hp").glob("hp*"))
        if (bp_root / "hp").exists()
        else sorted(bp_root.glob("hp*"))
    )
    invalid: set[str] = set()

    for p in candidate_dirs:
        m = re.search(r"\d+", p.name)
        if not m:
            continue
        pid = m.group(0)
        base = p
        required = [
            base / f"hp{pid}_hr.csv",
            base / f"hp{pid}_steps.csv",
            base / f"blood_pressure_readings_ID{pid}_cleaned.csv",
        ]
        if not all(path.exists() for path in required):
            continue

        try:
            hr_df, st_df, pos_df, neg_df = _bp_load_all(pid)
        except Exception as e:
            print(f"[sweep] pid={pid} dropped (load failed): {e}")
            invalid.add(pid)
            continue

        for seed in seeds:
            try:
                ensure_train_val_test_days(
                    pos_df, neg_df, hr_df, st_df,
                    input_df=input_df,
                    seed=int(seed),
                )
                # ensure_train_val_test_days_retry(
                #     pos_df, neg_df, hr_df, st_df,
                #     input_df=input_df,
                #     seed=int(seed),
                # )
            except RuntimeError as e:
                print(f"[sweep] pid={pid} fails for seed={seed}: {e}")
                invalid.add(pid)
                break

    if invalid:
        print(f"[sweep] invalid users (excluded from pool): {sorted(invalid)}")
    return invalid


def import_run_module(args, repo_root: Path, outdir: str):
    original_argv = sys.argv[:]
    run_argv = [
        original_argv[0],
        "--task",
        args.task,
        "--user",
        str(args.user),
        "--pool",
        args.pool,
        "--fruit",
        args.fruit,
        "--scenario",
        args.scenario,
        "--unlabeled_frac",
        str(args.unlabeled_frac),
        "--dropout_rate",
        str(args.dropout_rate),
        "--warm_start",
        str(args.warm_start),
        "--input_df",
        args.input_df,
        "--classifier",
        args.classifier,
    ]
    if getattr(args, "exclude_users", ""):
        run_argv.extend(["--exclude_users", args.exclude_users])
    if args.task == "bp":
        run_argv.extend(["--participant_id", str(args.participant_id)])

    original_output_dir = os.environ.get("BAN_AL_OUTPUT_DIR")
    os.environ["BAN_AL_OUTPUT_DIR"] = outdir
    sys.argv = run_argv
    try:
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import run as run_module
    finally:
        sys.argv = original_argv
        if original_output_dir is None:
            os.environ.pop("BAN_AL_OUTPUT_DIR", None)
        else:
            os.environ["BAN_AL_OUTPUT_DIR"] = original_output_dir

    run_module.OUTPUT_DIR = outdir
    return run_module


def prepare_shared_encoders_if_needed(args, run_module, outdir: str, seeds: list[int]) -> None:
    if not (args.submit and args.pool == "global" and args.input_df == "raw"):
        return

    top_out = Path(outdir)
    warmup_args = SimpleNamespace(
        user=args.user,
        pool=args.pool,
        fruit=args.fruit,
        scenario=args.scenario,
        task=args.task,
        participant_id=args.participant_id,
        unlabeled_frac=float(args.unlabeled_frac),
        dropout_rate=float(args.dropout_rate),
        warm_start=int(args.warm_start),
        results_subdir=args.results_subdir,
        input_df=args.input_df,
    )
    print(f"Preparing shared global encoders under {top_out / '_global_encoders'} for seeds {seeds}")
    # One warmup per AL seed so per-seed encoder caches are pre-populated and
    # the submitted jobs all hit a cache hit instead of racing on SimCLR training.
    for seed in seeds:
        print(f"[warmup] training encoder for seed={seed}")
        run_module.prepare_data(
            args=warmup_args,
            top_out=top_out,
            shared_enc_root=top_out / "_global_encoders",
            shared_cnn_root=top_out / "global_cnns",
            batch_ssl=32,
            ssl_epochs=100,
            pool=args.pool,
            task=args.task,
            input_df=args.input_df,
            seed=int(seed),
        )
    print("Shared global encoders are ready; submitting seed/method jobs.")


def exp_kwargs_for_seed(args, run_module, seed: int, method: str, output_dir: str) -> dict:
    return {
        "user": args.user,
        "pool": args.pool,
        "fruit": args.fruit,
        "scenario": args.scenario,
        "unlabeled_frac": float(args.unlabeled_frac),
        "dropout_rate": float(args.dropout_rate),
        "warm_start": int(args.warm_start),
        "seed": seed,
        "aq": method,
        "K": run_module.K[0],
        "Budget": run_module.Budget[0],
        "T": run_module.T[0],
        "input_df": args.input_df,
        "task": args.task,
        "participant_id": args.participant_id,
        "results_subdir": args.results_subdir,
        "output_dir": output_dir,
        "classifier": args.classifier,
        "exclude_users": getattr(args, "exclude_users", ""),
    }


def run_seed_jobs(args, run_module, repo_root: Path, outdir: str, job_outdir: str) -> Path:
    seeds = parse_csv_list(args.seeds, int)
    methods = parse_csv_list(args.methods, str)
    if not seeds:
        raise SystemExit("No seeds provided.")
    if not methods:
        raise SystemExit("No methods provided.")

    base_dir = Path(run_module.OUTPUT_DIR)
    job_base_dir = Path(job_outdir)
    scenario_dir = base_dir / args.pool / args.user / f"{args.fruit}_{args.scenario}"
    job_scenario_dir = job_base_dir / args.pool / args.user / f"{args.fruit}_{args.scenario}"

    template = None
    if args.submit:
        template_path = repo_root / "template.sh"
        template = template_path.read_text()

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        if not args.submit and hasattr(run_module, "reset_seeds"):
            run_module.reset_seeds(seed)

        for method in methods:
            exp_kwargs = exp_kwargs_for_seed(
                args,
                run_module,
                seed,
                method,
                job_outdir if args.submit else outdir,
            )
            seed_dir = scenario_dir / method / f"seed_{seed}"

            if args.submit:
                job_seed_dir = job_scenario_dir / method / f"seed_{seed}"
                submit_seed_job(
                    repo_root,
                    job_outdir,
                    template,
                    method,
                    job_seed_dir,
                    exp_kwargs,
                )
            else:
                seed_dir.mkdir(parents=True, exist_ok=True)
                run_module.run(str(seed_dir), method, exp_kwargs)

    return scenario_dir


def _read_progress_csv(path: Path, method: str | None = None, seed: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "round" not in df.columns:
        raise ValueError(f"{path} is missing required column: round")

    auc_col = None
    for candidate in ("AUC_Mean", "auc_mean", "auc", "AUC", "roc_auc", "test_auc"):
        if candidate in df.columns:
            auc_col = candidate
            break
    if auc_col is None:
        raise ValueError(
            f"{path} is missing an AUC column "
            "(AUC_Mean, auc_mean, auc, AUC, roc_auc, or test_auc)"
        )

    out = df[["round", auc_col]].copy()
    out = out.rename(columns={auc_col: "auc"})
    if "Pct_Total_Labeled" in df.columns:
        out["pct_labeled"] = pd.to_numeric(df["Pct_Total_Labeled"], errors="coerce")
    elif {"Num_Labeled", "Total_Data"}.issubset(df.columns):
        num_labeled = pd.to_numeric(df["Num_Labeled"], errors="coerce")
        total_data = pd.to_numeric(df["Total_Data"], errors="coerce")
        out["pct_labeled"] = 100.0 * num_labeled / total_data.where(total_data != 0)
    else:
        raise ValueError(
            f"{path} is missing Pct_Total_Labeled or Num_Labeled/Total_Data; "
            "cannot plot AUC against % labeled."
        )
    if "Num_Labeled" in df.columns:
        out["num_labeled"] = pd.to_numeric(df["Num_Labeled"], errors="coerce")
    else:
        out["num_labeled"] = pd.NA
    if "Total_Data" in df.columns:
        out["total_data"] = pd.to_numeric(df["Total_Data"], errors="coerce")
    else:
        out["total_data"] = pd.NA
    out["source"] = str(path)
    if seed is not None:
        out["seed"] = seed
    elif "seed" in df.columns:
        out["seed"] = df["seed"]
    else:
        path_seed = _seed_from_path(path)
        out["seed"] = path_seed if path_seed is not None else path.parent.name
    if method is not None:
        out["method"] = method
    return out


def _seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def load_auc_rows_from_runs(
    scenario_dir: Path,
    seeds: list[int],
    methods: list[str],
    hp_contains: str | None = None,
) -> pd.DataFrame:
    frames = []
    for method in methods:
        for seed in seeds:
            seed_dir = scenario_dir / method / f"seed_{seed}"
            paths = sorted(seed_dir.rglob("al_progress.csv"))
            if hp_contains:
                paths = [path for path in paths if hp_contains in str(path.parent)]
            if not paths:
                print(f"Missing al_progress.csv for {seed_dir}")
                continue
            if len(paths) > 1:
                print(f"Found multiple al_progress.csv files for {seed_dir}; using {paths[0]}")
            frames.append(_read_progress_csv(paths[0], method=method, seed=seed))

    if not frames:
        raise FileNotFoundError(f"No al_progress.csv files found under {scenario_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["pct_labeled"] = pd.to_numeric(df["pct_labeled"], errors="coerce")
    df["num_labeled"] = pd.to_numeric(df["num_labeled"], errors="coerce")
    df["total_data"] = pd.to_numeric(df["total_data"], errors="coerce")
    df = df.dropna(subset=["round", "auc", "pct_labeled"])
    df["round"] = df["round"].astype(int)
    return df


def _read_upper_bound_auc(path: Path) -> float:
    import numpy as np

    value = np.load(path, allow_pickle=True)
    return float(value.reshape(-1)[0])


def load_full_data_auc_rows_from_runs(
    scenario_dir: Path,
    seeds: list[int],
    methods: list[str],
    hp_contains: str | None = None,
) -> pd.DataFrame:
    rows = []
    for method in methods:
        for seed in seeds:
            seed_dir = scenario_dir / method / f"seed_{seed}"
            paths = sorted(seed_dir.rglob("upper_bound_auc.npy"))
            if hp_contains:
                paths = [path for path in paths if hp_contains in str(path.parent)]
            if not paths:
                print(f"Missing 100% labeled AUC for {seed_dir}")
                continue
            if len(paths) > 1:
                print(f"Found multiple upper_bound_auc.npy files for {seed_dir}; using {paths[0]}")
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "pct_labeled": 100.0,
                    "auc": _read_upper_bound_auc(paths[0]),
                    "source": str(paths[0]),
                }
            ) 
    return pd.DataFrame(rows)


def average_auc_per_round(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["method", "round"], as_index=False)
        .agg(
            Pct_Total_Labeled=("pct_labeled", "mean"),
            Pct_Total_Labeled_STD=("pct_labeled", "std"),
            Num_Labeled=("num_labeled", "mean"),
            Num_Labeled_STD=("num_labeled", "std"),
            Total_Data=("total_data", "mean"),
            AUC_Mean=("auc", "mean"),
            AUC_STD=("auc", "std"),
            N_Runs=("auc", "count"),
        )
        .sort_values(["method", "round"])
        .reset_index(drop=True)
    )
    summary["AUC_STD"] = summary["AUC_STD"].fillna(0.0)
    summary["Pct_Total_Labeled_STD"] = summary["Pct_Total_Labeled_STD"].fillna(0.0)
    summary["Num_Labeled_STD"] = summary["Num_Labeled_STD"].fillna(0.0)
    return summary


def average_full_data_auc(full_df: pd.DataFrame) -> pd.DataFrame:
    if full_df.empty:
        return pd.DataFrame()
    summary = (
        full_df.groupby("method", as_index=False)
        .agg(
            Pct_Total_Labeled=("pct_labeled", "mean"),
            AUC_Mean=("auc", "mean"),
            AUC_STD=("auc", "std"),
            N_Runs=("auc", "count"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    summary["AUC_STD"] = summary["AUC_STD"].fillna(0.0)
    return summary


def plot_auc_summary(
    summary: pd.DataFrame,
    out_path: Path,
    title: str,
    full_data_summary: pd.DataFrame | None = None,
) -> None:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {"uncertainty": "tab:blue", "random": "darkorange", "coreset": "seagreen"}

    plt.figure(figsize=(9, 5.5))
    for method, method_df in summary.groupby("method", sort=False):
        method_df = method_df.sort_values("round")
        x = method_df["Pct_Total_Labeled"].to_numpy()
        y = method_df["AUC_Mean"].to_numpy()
        yerr = method_df["AUC_STD"].to_numpy()
        color = colors.get(method)
        plt.plot(x, y, marker="o", linewidth=2, label=f"{method} mean", color=color)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)
  
    tick_df = (
        summary.groupby("round", as_index=False)
        .agg(Pct_Total_Labeled=("Pct_Total_Labeled", "mean"))
        .dropna(subset=["Pct_Total_Labeled"])
        .sort_values("round")
    )
    tick_positions = tick_df["Pct_Total_Labeled"].tolist()

    if full_data_summary is not None and not full_data_summary.empty:
        xmin = float(summary["Pct_Total_Labeled"].min())
        full_label_used = False
        for _, row in full_data_summary.iterrows():
            y = float(row["AUC_Mean"])
            yerr = float(row["AUC_STD"])
            label = f"100% labeled AUC = {y:.3f} ± {yerr:.3f}" if not full_label_used else None
            plt.hlines(
                y,
                xmin=xmin,
                xmax=max(tick_positions),
                colors="red",
                linestyles="--",
                linewidth=1.8,
                alpha=0.85,
                label=label,
            )
            full_label_used = True

    plt.xlabel("% Labeled")
    plt.ylabel("AUC")
    plt.title(title)


    tick_labels = [f"{pct:.1f}" for pct in tick_df["Pct_Total_Labeled"]]

    step = max(1, len(tick_positions) // 8)

    plt.xticks(
        tick_positions[::step],
        [f"{pct:.1f}" for pct in tick_positions[::step]],
        rotation=30,
    )

    plt.xlim(min(tick_positions), max(tick_positions))
    
    # plt.xlim(left=0.0, right=float(tick_labels[-1]) if tick_labels else 100.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200, facecolor="white")
    plt.close()
    

def _set_y_padding(ax, values: list[float]) -> None:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if vals.empty:
        return
    ymin = float(vals.min())
    ymax = float(vals.max())
    pad = max(0.02, (ymax - ymin) * 0.15)
    ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))


def _plot_summary_lines(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    methods: list[str],
    *,
    show_legend: bool = True,
) -> list[float]:
    y_values = []
    for method in methods:
        method_df = summary[summary["method"] == method].sort_values("round")
        if method_df.empty or x_col not in method_df.columns:
            continue

        x = pd.to_numeric(method_df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(method_df["AUC_Mean"], errors="coerce").to_numpy(dtype=float)
        yerr = pd.to_numeric(method_df["AUC_STD"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mask = ~(pd.isna(x) | pd.isna(y))
        if not mask.any():
            continue

        x = x[mask]
        y = y[mask]
        yerr = yerr[mask]
        color = METHOD_COLORS.get(method)
        ax.plot(x, y, linewidth=2.0, color=color, label=method)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.18, linewidth=0)
        y_values.extend(y.tolist())
        y_values.extend((y - yerr).tolist())
        y_values.extend((y + yerr).tolist())

    if show_legend:
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    return y_values


def _plot_full_data_reference(ax, full_summary: pd.DataFrame, y_values: list[float]) -> None:
    if not isinstance(full_summary, pd.DataFrame) or full_summary.empty:
        return
    auc_vals = pd.to_numeric(full_summary["AUC_Mean"], errors="coerce").dropna()
    if auc_vals.empty:
        return
    std_vals = pd.to_numeric(full_summary.get("AUC_STD"), errors="coerce").dropna()
    upper_auc = float(auc_vals.mean())
    upper_std = float(std_vals.mean()) if not std_vals.empty else 0.0
    ax.axhline(
        upper_auc,
        color="red",
        linestyle="--",
        linewidth=1.6,
        alpha=0.85,
        label=f"100%-data ceiling ({upper_auc:.3f})",
    )
    y_values.extend([upper_auc - upper_std, upper_auc, upper_auc + upper_std])


def aggregate_grid_records(records: list[dict], methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    full_frames = []
    for record in records:
        summary = record.get("summary")
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            df = summary.copy()
            df["target_user"] = record["user"]
            frames.append(df)
        full_summary = record.get("full_summary")
        if isinstance(full_summary, pd.DataFrame) and not full_summary.empty:
            full_df = full_summary.copy()
            full_df["target_user"] = record["user"]
            full_frames.append(full_df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        aggregate = (
            combined.groupby(["method", "round"], as_index=False)
            .agg(
                Pct_Total_Labeled=("Pct_Total_Labeled", "mean"),
                AUC_Mean=("AUC_Mean", "mean"),
                AUC_STD=("AUC_Mean", "std"),
                N_Targets=("target_user", "nunique"),
            )
            .sort_values(["method", "round"])
            .reset_index(drop=True)
        )
        aggregate["AUC_STD"] = aggregate["AUC_STD"].fillna(0.0)
    else:
        aggregate = pd.DataFrame()

    if full_frames:
        full_combined = pd.concat(full_frames, ignore_index=True)
        full_aggregate = (
            full_combined.groupby("method", as_index=False)
            .agg(
                Pct_Total_Labeled=("Pct_Total_Labeled", "mean"),
                AUC_Mean=("AUC_Mean", "mean"),
                AUC_STD=("AUC_Mean", "std"),
                N_Targets=("target_user", "nunique"),
            )
            .sort_values("method")
            .reset_index(drop=True)
        )
        full_aggregate["AUC_STD"] = full_aggregate["AUC_STD"].fillna(0.0)
    else:
        full_aggregate = pd.DataFrame()

    if not aggregate.empty:
        aggregate = aggregate[aggregate["method"].isin(methods)]
    if not full_aggregate.empty:
        full_aggregate = full_aggregate[full_aggregate["method"].isin(methods)]
    return aggregate, full_aggregate


def plot_auc_grid(records: list[dict], methods: list[str], out_path: Path) -> None:
    if not records:
        return

    configure_matplotlib_cache()
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(records) + 1
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes_2d = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows),
        sharey="row", sharex=False,
    )
    axes_2d = np.asarray(axes_2d).reshape(nrows, ncols)
    axes = axes_2d.reshape(-1)

    # With sharex=True, compute a single union x-range from every target.
    all_x_vals: list[float] = []
    for record in records:
        summary = record.get("summary")
        if isinstance(summary, pd.DataFrame) and "Pct_Total_Labeled" in summary.columns:
            vals = pd.to_numeric(summary["Pct_Total_Labeled"], errors="coerce").dropna()
            all_x_vals.extend(vals.tolist())
    global_xlim = (float(min(all_x_vals)), float(max(all_x_vals))) if all_x_vals else None

    row_y_values: dict[int, list[float]] = {}
    for idx, record in enumerate(records):
        ax = axes[idx]
        summary = record["summary"]
        x_col = "Pct_Total_Labeled"
        y_values = _plot_summary_lines(ax, summary, x_col, methods)
        _plot_full_data_reference(ax, record.get("full_summary"), y_values)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"P-{record['user']}")
        ax.set_xlabel("% Labeled")
        # Only the leftmost panel in each row keeps the y-label.
        ax.set_ylabel("ROC-AUC" if idx % ncols == 0 else "")
        row_y_values.setdefault(idx // ncols, []).extend(y_values)

    if global_xlim is not None:
        axes[0].set_xlim(*global_xlim)  # sharex=True propagates this to every panel

    # With sharey="row", set ylim once per row from the pooled y-values of all panels in that row.
    for row, vals in row_y_values.items():
        if not vals:
            continue
        _set_y_padding(axes_2d[row, 0], vals)

    aggregate, full_aggregate = aggregate_grid_records(records, methods)
    aggregate_ax = axes[len(records)]
    if not aggregate.empty:
        y_values = _plot_summary_lines(
            aggregate_ax,
            aggregate,
            "Pct_Total_Labeled",
            methods,
        )
        if not full_aggregate.empty:
            _plot_full_data_reference(aggregate_ax, full_aggregate, y_values)
            aggregate_ax.legend(fontsize=8, loc="lower right")
        aggregate_ax.set_title("Aggregate (pooled labels across participants)")
        aggregate_ax.set_xlabel("% of total training data labeled")
        aggregate_ax.set_ylabel("ROC-AUC")
        # xlim is set globally above via sharex=True; per-panel set_xlim would override the shared range.
        _set_y_padding(aggregate_ax, y_values)
    else:
        aggregate_ax.axis("off")

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("BP per-participant target AUC + aggregate", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved grid plot: {out_path}")


def plot_personal_cumulative_grid_one_method(
    records: list[dict],
    method: str,
    out_path: Path,
    bin_size: int = 1,
) -> None:
    """One panel per user; bars showing cumulative Num_Labeled per round for a single method.

    Intended for pool=personal where Pct_Total_Labeled has a per-user denominator and is therefore
    not directly comparable across users. Num_Labeled is the raw integer label count.

    bin_size > 1 collapses every `bin_size` consecutive rounds into one bar, taking the
    cumulative value at the end of the bin (so the staircase still climbs).
    """
    if not records:
        return

    configure_matplotlib_cache()
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(records)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes_2d = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharex=False,
    )
    axes_2d = np.asarray(axes_2d).reshape(nrows, ncols)
    axes = axes_2d.reshape(-1)

    # Color bars per target user using the same glasbey palette as the global
    # query-distribution heatmaps so colors are consistent across plots.
    try:
        import colorcet as cc
        palette = cc.glasbey_category10
    except ImportError:
        palette = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    bin_size = max(1, int(bin_size))

    for idx, record in enumerate(records):
        ax = axes[idx]
        color = palette[idx % len(palette)]
        summary = record.get("summary")
        if not isinstance(summary, pd.DataFrame) or summary.empty:
            ax.axis("off")
            continue
        if "Num_Labeled" not in summary.columns or "Pct_Total_Labeled" not in summary.columns:
            ax.axis("off")
            continue

        method_df = summary[summary["method"] == method].sort_values("round")
        if method_df.empty:
            ax.axis("off")
            continue

        rounds = pd.to_numeric(method_df["round"], errors="coerce").to_numpy(dtype=float)
        counts = pd.to_numeric(method_df["Num_Labeled"], errors="coerce").to_numpy(dtype=float)
        pcts = pd.to_numeric(method_df["Pct_Total_Labeled"], errors="coerce").to_numpy(dtype=float)
        mask = ~(pd.isna(rounds) | pd.isna(counts) | pd.isna(pcts))
        if not mask.any():
            ax.axis("off")
            continue
        rounds = rounds[mask]
        counts = counts[mask]
        pcts = pcts[mask]

        if bin_size > 1:
            # Keep only the last round in each bin so the cumulative staircase is preserved.
            min_round = int(rounds.min())
            bin_idx = ((rounds - min_round) // bin_size).astype(int)
            keep_positions = []
            for b in np.unique(bin_idx):
                positions_in_bin = np.where(bin_idx == b)[0]
                keep_positions.append(positions_in_bin[-1])
            keep_positions = np.asarray(sorted(keep_positions))
            rounds = rounds[keep_positions]
            counts = counts[keep_positions]
            pcts = pcts[keep_positions]

        x_base = np.arange(len(pcts))
        ax.bar(
            x_base,
            counts,
            width=0.8,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )

        ax.set_title(f"P-{record['user']}")
        ax.set_xlabel("% of total data labeled")
        ax.set_ylabel("Cumulative labels acquired" if idx % ncols == 0 else "")
        if len(pcts) > 15:
            step = max(1, len(pcts) // 10)
            tick_idx = list(range(0, len(pcts), step))
            ax.set_xticks([x_base[i] for i in tick_idx])
            ax.set_xticklabels([f"{pcts[i]:.1f}" for i in tick_idx], fontsize=8, rotation=30)
        else:
            ax.set_xticks(x_base)
            ax.set_xticklabels([f"{p:.1f}" for p in pcts], fontsize=8, rotation=30)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"Personal pool: cumulative labels vs % labeled per user ({method})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved personal cumulative-counts grid ({method}): {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run run.py across seeds and average AUC per AL round.")
    parser.add_argument("--seeds", default="41,42,43", help="Comma-separated seeds.")
    parser.add_argument("--methods", default="random,coreset", help="Comma-separated AL methods.")
    parser.add_argument("--results_subdir", default="results")
    parser.add_argument("--user", default="35")
    parser.add_argument(
        "--users",
        default=None,
        help="Optional comma- or space-separated users for multi-user grid/analyze.",
    )
    parser.add_argument("--participant_id", default=None)
    parser.add_argument("--pool", default="global")
    parser.add_argument("--fruit", default="BP")
    parser.add_argument("--scenario", default="spike")
    parser.add_argument("--unlabeled_frac", default="0.22")
    parser.add_argument("--dropout_rate", default="0.5")
    parser.add_argument("--warm_start", default="0")
    parser.add_argument("--task", default="bp")
    parser.add_argument("--input_df", default="raw")
    parser.add_argument("--classifier", default="mlp", choices=["mlp", "lr"])
    parser.add_argument(
        "--exclude_users",
        default="",
        help="Comma-separated user IDs to exclude from the global pool. Overrides --auto_exclude.",
    )
    parser.add_argument(
        "--auto_exclude",
        action="store_true",
        help="Before launching runs, sweep every (user, seed) for valid splits; exclude users that fail any seed.",
    )
    parser.add_argument(
        "--hp_contains",
        default=None,
        help="Optional substring to disambiguate hp folders (e.g. 'K10') when multiple exist per seed.",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=1,
        help="Bin size for the personal cumulative-counts grid (1 = one bar per round, 5 = one bar per 5 rounds).",
    )
    parser.add_argument("--outdir", default="avg_auc_results")
    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        default=False,
        help="Submit each seed/method as a Slurm job.",
    )
    parser.add_argument(
        "--local",
        dest="submit",
        action="store_false",
        help="Run each seed/method locally.",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Skip run.py and aggregate existing al_progress.csv files.",
    )
    parser.add_argument("--out_csv", type=Path, default=None, help="Output CSV for per-round AUC summary.")
    parser.add_argument("--raw_csv", type=Path, default=None, help="Output CSV for seed-level AUC rows.")
    parser.add_argument("--out_plot", type=Path, default=None, help="Output PNG for the AUC plot.")
    parser.add_argument("--grid_out", type=Path, default=None, help="Output PNG for multi-user grid plot.")
    parser.add_argument("--title", default="Average AUC per Round Across Seeds", help="Plot title.")
    return parser.parse_args()


      
def process_user(
    args: argparse.Namespace,
    run_module,
    repo_root: Path,
    outdir: str,
    job_outdir: str,
    seeds: list[int],
    methods: list[str],
) -> dict | None:
    scenario_dir = Path(run_module.OUTPUT_DIR) / args.pool / args.user / f"{args.fruit}_{args.scenario}"

    if not args.analyze_only:
        scenario_dir = run_seed_jobs(args, run_module, repo_root, outdir, job_outdir)

    df = load_auc_rows_from_runs(scenario_dir, seeds, methods, hp_contains=args.hp_contains)
    summary = average_auc_per_round(df)
    full_df = load_full_data_auc_rows_from_runs(scenario_dir, seeds, methods, hp_contains=args.hp_contains)
    full_summary = average_full_data_auc(full_df)

    out_csv = args.out_csv or (scenario_dir / "auc_mean_std_per_round.csv")
    raw_csv = args.raw_csv or (scenario_dir / "auc_by_seed_per_round.csv")
    full_raw_csv = scenario_dir / "full_data_auc_by_seed.csv"
    full_summary_csv = scenario_dir / "full_data_auc_summary.csv"
    out_plot = args.out_plot or (scenario_dir / "auc_mean_std_per_round.png")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_csv, index=False)
    summary.to_csv(out_csv, index=False)
    if not full_df.empty:
        full_df.to_csv(full_raw_csv, index=False)
        full_summary.to_csv(full_summary_csv, index=False)
    plot_auc_summary(summary, out_plot, args.title, full_summary)

    print(f"\nTarget user {args.user}")
    print(summary.to_string(index=False))
    if not full_summary.empty:
        print("\n100% labeled AUC:")
        print(full_summary.to_string(index=False))
    print(f"Saved seed-level AUC CSV: {raw_csv}")
    print(f"Saved summary CSV: {out_csv}")
    if not full_df.empty:
        print(f"Saved 100% labeled AUC CSV: {full_summary_csv}")
    print(f"Saved plot: {out_plot}")

    return {
        "user": str(args.user),
        "scenario_dir": scenario_dir,
        "summary": summary,
        "full_summary": full_summary,
    }


def main() -> int:
    args = parse_args()
    configure_matplotlib_cache()

    users = parse_user_list(args.users) or parse_user_list(args.user)
    # breakpoint()
    if not users:
        raise SystemExit("No users provided.")

    if args.task == "bp":
        args.participant_id = args.participant_id or users[0]
        args.user = str(args.participant_id)
    else:
        args.user = users[0]

    seeds = parse_csv_list(args.seeds, int)
    methods = parse_csv_list(args.methods, str)

    # If auto_exclude is on, sweep first and derive the target cohort from the
    # survivors. The target list (what auc_grid plots) and the source pool (who
    # AL can query) come from the same set, so they can't drift apart.
    if args.auto_exclude and not args.analyze_only and args.task == "bp":
        discovered_invalid = discover_invalid_users(seeds, args.task, args.input_df)
        manual_exclude = {u.strip() for u in args.exclude_users.split(",") if u.strip()}
        candidates = discover_bp_candidate_users()
        survivors = sorted(
            (set(candidates) - discovered_invalid - manual_exclude),
            key=lambda s: int(s),
        )

        # Source-pool exclusion = everyone in candidates who isn't a survivor.
        # This guarantees the source pool inside new_prep.py == the target list.
        full_exclude = sorted(
            set(candidates) - set(survivors),
            key=lambda s: int(s),
        )
        args.exclude_users = ",".join(full_exclude)

        print(f"[cohort] candidates: {sorted(candidates, key=lambda s: int(s))}")
        if discovered_invalid:
            print(f"[cohort] auto-excluded (invalid splits): {sorted(discovered_invalid, key=lambda s: int(s))}")
        if manual_exclude:
            print(f"[cohort] manually excluded: {sorted(manual_exclude, key=lambda s: int(s))}")
        print(f"[cohort] surviving targets (also source pool): {survivors}")

        # Replace the user list passed on the CLI with the survivor set.
        users = survivors
        if not users:
            raise SystemExit("[cohort] No users survived exclusion.")
        if args.task == "bp":
            args.participant_id = users[0]
            args.user = str(args.participant_id)
        else:
            args.user = users[0]
    elif args.exclude_users:
        # Manual-only path: keep behavior as-is (don't override the user list).
        manual_exclude = {u.strip() for u in args.exclude_users.split(",") if u.strip()}
        if manual_exclude:
            print(f"[exclude_users] effective: {sorted(manual_exclude, key=lambda s: int(s))}")

    repo_root = Path(__file__).resolve().parent
    outdir_path = Path(args.outdir)
    if not outdir_path.is_absolute():
        outdir_path = repo_root / outdir_path
    outdir = str(outdir_path)
    job_outdir = str(Path(args.outdir)) if args.submit else outdir

    run_module = import_run_module(args, repo_root, outdir)
    records = []

    if not args.analyze_only and args.submit:
        prepare_shared_encoders_if_needed(args, run_module, outdir, seeds)

    for user in users:
        user_args = argparse.Namespace(**vars(args))
        user_args.user = str(user)
        if user_args.task == "bp":
            user_args.participant_id = str(user)

        if args.submit and not args.analyze_only:
            run_seed_jobs(user_args, run_module, repo_root, outdir, job_outdir)
            continue

        records.append(
            process_user(
                user_args,
                run_module,
                repo_root,
                outdir,
                job_outdir,
                seeds,
                methods,
            )
        )

    if args.submit and not args.analyze_only:
        total_jobs = len(users) * len(seeds) * len(methods)
        print(f"Submitted {total_jobs} jobs for {len(users)} user(s).")
        print("After the jobs finish, rerun with --analyze_only to compute summaries and the grid.")
        return 0

    records = [record for record in records if record is not None]
    if len(records) > 1 or args.grid_out is not None:
        grid_out = args.grid_out
        if grid_out is None:
            grid_out = Path(run_module.OUTPUT_DIR) / args.pool / f"{args.fruit}_{args.scenario}" / "auc_grid.png"
        elif not grid_out.is_absolute():
            grid_out = repo_root / grid_out
        plot_auc_grid(records, methods, grid_out)

        if args.pool == "personal":
            for method in methods:
                cum_out = grid_out.parent / f"cumulative_counts_grid_{method}.png"
                plot_personal_cumulative_grid_one_method(records, method, cum_out, bin_size=args.bin_size)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
