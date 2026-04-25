'''
script to plot label disubtion among users 
'''
import os
import glob
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

# ----------------------------
# CONFIG (defaults)
# ----------------------------
stacked = True   # True => cumulative (per-participant) grouped bars; False => non-cumulative grouped bars
stack_participants = False  # True => stack participants into one bar per round
bin_size = 3 # number of rounds to combine per bar (>=1)
tick_gap = 10.0 # spacing multiplier between x positions
group_width = 3.80 # width of each round group (leave gaps between groups)
break_round = None
aq = "uncertainty"  # active learning aquistion function (for save path)
output_dir = "Cardiomate_AL"  # root output directory (for searching pkls and saving plots)
# output_dir = "/Users/hiwotbelaytadesse/Desktop/Banaware_AL/Count_check/"
# ID11_pkl = "cluster_results/global_SSL/global/ID5/Melon_Crave/random/UF70_K20_B7_DR20/queried_participant_counts_ID5_global.pkl"
# ID12_pkl = "cluster_results/global_SSL/global/ID12/Melon_Crave/random/UF70_K20_B7_DR20/queried_participant_counts_ID12_global.pkl"
# with open(Path(ID11_pkl), "rb") as f:
#     data = pickle.load(f)
# with open(Path(ID11_pkl), "rb") as f:
#     data2 = pickle.load(f)

pool = "global_supervised"  # participant pool (for saving plots)
fruit_scenario = "Nectarine_Crave"
# results_root_name = "Final_Banaware/GS"
results_root_name = "/Users/hiwotbelaytadesse/Desktop/Banaware_AL/Count_check/"
stacked_plots_dirname = "Stacked_plots"
path_root_override = None  # override root for parsing user/fruit from paths (e.g., "Mirror_auc_test")
POOL_ROOTS = {
    "personal": "P_SSL",
    "global": "global_SSL",
    "global_supervised": "GS",
}

# IMPORTANT: global participant set (keep fixed across ALL plots)
# Set to None to derive from the loaded data.
GLOBAL_PARTICIPANTS = None

pa = argparse.ArgumentParser()
pa.add_argument("--task", choices=["fruit", "bp"], default="fruit")
pa.add_argument("--output_dir", default=output_dir)
pa.add_argument("--pool", default=pool)
pa.add_argument("--aq", default=aq)
pa.add_argument("--fruit_scenario", default=fruit_scenario)
pa.add_argument("--all_from_list", type=int, default=1, help="Use built-in user/scenario lists")
pa.add_argument("--stacked", type=int, default=int(stacked))
pa.add_argument("--stack_participants", type=int, default=int(stack_participants))
pa.add_argument("--bin_size", type=int, default=bin_size)
pa.add_argument("--tick_gap", type=float, default=tick_gap)
pa.add_argument("--group_width", type=float, default=group_width)
pa.add_argument("--break_round", type=int, default=-1)
pa.add_argument("--results_root_name", default=results_root_name)
pa.add_argument("--stacked_plots_dirname", default=stacked_plots_dirname)
pa.add_argument("--path_root_override", default=path_root_override)
args, _ = pa.parse_known_args()

task = args.task
output_dir = args.output_dir
pool = args.pool
aq = args.aq
fruit_scenario = args.fruit_scenario
all_from_list = bool(args.all_from_list)
stacked = bool(args.stacked)
stack_participants = bool(args.stack_participants)
bin_size = args.bin_size
tick_gap = args.tick_gap
group_width = args.group_width
break_round = None if args.break_round < 0 else args.break_round
results_root_name = args.results_root_name
stacked_plots_dirname = args.stacked_plots_dirname
path_root_override = args.path_root_override

pool_root = POOL_ROOTS.get(pool)
if pool_root is None:
    raise ValueError(f"Unknown pool: {pool}")
base_dir = os.path.join(output_dir, pool_root, pool)
save_dir = os.path.join(output_dir, stacked_plots_dirname, pool, aq, fruit_scenario)
if path_root_override is None and results_root_name == output_dir:
    results_root_name = base_dir

# User/scenario lists (from Makefile)
def get_bp_users():
    users = []
    bp_root = Path("DATA/Cardiomate/hp")
    for p in sorted(bp_root.glob("hp*")):
        pid = "".join([c for c in p.name if c.isdigit()])
        if not pid:
            continue
        base = bp_root / f"hp{pid}"
        if not (base / f"hp{pid}_hr.csv").exists():
            continue
        if not (base / f"hp{pid}_steps.csv").exists():
            continue
        if not (base / f"blood_pressure_readings_ID{pid}_cleaned.csv").exists():
            continue
        users.append(pid)
    return users


SCENARIO_USERS = (
    {"BP_spike": get_bp_users()}
    if task == "bp"
    else {
        "Almond_Use": ["ID11", "ID13", "ID19", "ID25", "ID28"],
        "Almond_Crave": ["ID11", "ID19", "ID25"],
        "Melon_Crave": ["ID5", "ID9", "ID12", "ID19", "ID20", "ID21", "ID27"],
        "Melon_Use": ["ID12", "ID19", "ID20", "ID27"],
        "Carrot_Crave": ["ID10", "ID11", "ID14", "ID15", "ID18", "ID25"],
        "Carrot_Use": ["ID10", "ID11", "ID13", "ID14", "ID15", "ID18", "ID26"],
        "Nectarine_Crave": ["ID10", "ID11", "ID12", "ID20", "ID21", "ID27"],
        "Nectarine_Use": ["ID10", "ID11", "ID12", "ID13", "ID20", "ID21", "ID27"],
    }
)

def _distinct_colors(n: int, bp_mode: bool) -> list:
    """Return n visually distinct colors; BP mode avoids repeats."""
    if n <= 0:
        return []
    if bp_mode:
        # Fixed high-contrast saturated palette for BP users (no gradients, no repeats).
        bp_palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
            "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
            "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#393b79",
            "#ad494a", "#637939", "#8c6d31", "#843c39", "#7b4173",
            "#08519c", "#a50f15", "#00441b", "#7f2704", "#3f007d",
        ]
        if n > len(bp_palette):
            raise ValueError(
                f"BP palette supports up to {len(bp_palette)} distinct users without repeats; got {n}."
            )
        return bp_palette[:n]
    # Keep fruit defaults stable.
    base = [
        "#377eb8",  # blue
        "#e41a1c",  # red
        "#4daf4a",  # green
        "#ff7f00",  # orange
        "#984ea3",  # purple
        "#ffff33",  # yellow
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # gray
    ]
    if n <= len(base):
        return base[:n]
    cmap = plt.get_cmap("tab20", n)
    return [cmap(i) for i in range(n)]

def build_df_wide(participants_count_per_round):
    if isinstance(participants_count_per_round, pd.DataFrame):
        df_wide = participants_count_per_round.copy()
        if "round" in df_wide.columns:
            df_wide = df_wide.set_index("round")
    elif isinstance(participants_count_per_round, dict):
        df_wide = (
            pd.DataFrame.from_dict(participants_count_per_round, orient="index")
            .fillna(0)
        )
    else:
        raise ValueError(f"Unsupported pickle type: {type(participants_count_per_round)}")

    try:
        df_wide.index = df_wide.index.astype(int)
        df_wide = df_wide.sort_index()
    except Exception:
        pass

    df_wide = df_wide.fillna(0).astype(int)
    df_wide.index.name = "round"
    df_wide.columns = df_wide.columns.astype(str)

    participants = GLOBAL_PARTICIPANTS
    if participants is None:
        participants = sorted(df_wide.columns.tolist())

    df_wide = df_wide.reindex(columns=sorted(participants), fill_value=0)
    df_wide.columns.name = "participant"
    return df_wide


def _parse_user_scenario_from_path(pkl_path, scenario_name=None):
    parts = Path(pkl_path).parts
    if scenario_name and scenario_name in parts:
        idx = parts.index(scenario_name)
        user_id = parts[idx - 1] if idx - 1 >= 0 else "unknown"
        return user_id, scenario_name
    return "unknown", "unknown"

def plot_counts_for_pickle(pkl_path, pool, scenario_name=None, ax=None):
    with open(pkl_path, "rb") as f:
        participants_count_per_round = pickle.load(f)
    print(f"Loaded PKL: {pkl_path}")
    print(f"PKL type: {type(participants_count_per_round)}")
    if isinstance(participants_count_per_round, dict):
        rounds = sorted(list(participants_count_per_round.keys()))[:5]
        print(f"PKL rounds sample: {rounds}")
    elif isinstance(participants_count_per_round, pd.DataFrame):
        print(participants_count_per_round.head())

    df_wide = build_df_wide(participants_count_per_round)

    # FIXED COLOR MAPPING (GLOBAL)
    all_participants = df_wide.columns.tolist()
    if pool == "personal":
        distinct_colors = ["#ff7f00"]  # orange
    else:
        distinct_colors = _distinct_colors(len(all_participants), bp_mode=(task == "bp"))
    participant_to_color = {p: distinct_colors[i] for i, p in enumerate(all_participants)}

    # CHOOSE PLOT DATA
    df_plot = df_wide
    if bin_size > 1:
        round_index = pd.to_numeric(df_plot.index, errors="coerce")
        if round_index.isna().any():
            print("Skipping binning: round index contains non-numeric values.")
            bin_index = None
        else:
            min_round = int(round_index.min())
            bin_index = (round_index - min_round) // bin_size
        if bin_index is not None:
            df_plot = df_plot.groupby(bin_index).sum()
            df_plot.index = df_plot.index * bin_size + min_round
    if stacked:
        df_plot = df_plot.cumsum(axis=0)
    rounds = df_plot.index.tolist()
    participants = [
        p for p in df_plot.columns if df_plot[p].to_numpy().sum() > 0
    ]

    n_rounds = len(rounds)
    n_participants = len(participants)
    group_x = np.arange(n_rounds) * tick_gap

    created_fig = False
    in_grid = False
    if ax is None:
        fig_width = 12
        fig_height = 6
        plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()
        created_fig = True
    else:
        in_grid = True

    if stack_participants:
        bottom = np.zeros(n_rounds)
        for participant in participants:
            values = df_plot[participant].values
            ax.bar(
                group_x,
                values,
                bottom=bottom,
                edgecolor="black",
                color=participant_to_color.get(participant, "tab:gray"),
                alpha=1.0 if task == "bp" else 0.85,
                label=participant,
            )
            bottom += values
    else:
        bar_width =max(0.5, group_width / max(1, n_participants))
        for j, participant in enumerate(participants):
            x_pos = group_x - group_width / 2 + j * bar_width + bar_width / 2
            ax.bar(
                x_pos,
                df_plot[participant].values,
                width=bar_width,
                edgecolor="black",
                color=participant_to_color.get(participant, "tab:gray"),
                alpha=1.0 if task == "bp" else 0.85,
                label=participant,
            )

    ax.set_xticks(group_x, [str(r) for r in rounds])
    tick_fs = 10 if in_grid else 20
    label_fs = 12 if in_grid else 22
    label_pad = 6 if in_grid else 20
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.set_xlabel("Active Learning Round", fontsize=label_fs, fontweight='bold', labelpad=label_pad)
    ax.set_ylabel(
        "Cumulative Number of \n Queried Windows" if stacked else "Queried Windows (Count)",
        fontsize=label_fs,  fontweight='bold', labelpad=label_pad
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    if break_round is not None and break_round in rounds:
        break_idx = rounds.index(break_round)
        ax.axvline(group_x[break_idx], color="black", linestyle="--", linewidth=1.5)
    legend_fs = 8 if in_grid else 20
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=legend_fs)
    if created_fig:
        plt.tight_layout(pad=0.3)

   
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(os.path.dirname(pkl_path))
    user_id, fruit_scenario = _parse_user_scenario_from_path(pkl_path, scenario_name)
    parts = pkl_path.split(os.sep)
    root_name = path_root_override or results_root_name
    root_parts = [p for p in root_name.split(os.sep) if p]

    def find_sublist(haystack, needle):
        if not needle:
            return None
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return None

    if root_parts:
        out_idx = find_sublist(parts, root_parts)
    else:
        out_idx = None

    if out_idx is None and root_name in parts:
        out_idx = parts.index(root_name)

    if out_idx is not None:
        base_idx = out_idx + len(root_parts)
        if base_idx < len(parts):
            user_id = parts[base_idx]
        if base_idx + 2 < len(parts):
            fruit_scenario = parts[base_idx + 2]
    suffix = "stacked" if stacked else "per_round"
    png_path = os.path.join(
        save_dir,
        f"{user_id}_{fruit_scenario}_{base_name}_{suffix}_{aq}.png",
    )
    if created_fig:
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved plot to {png_path}")
    return {
        "png_path": png_path,
        "user_id": user_id,
        "fruit_scenario": fruit_scenario,
    }

def plot_counts_grid_from_pkls(pkl_paths, pool, ncols=3, save_path=None, scenario_name=None):
    """
    Plot individual queried-participant-count plots into a single grid figure.
    """
    if not pkl_paths:
        print("No PKL paths provided for grid plotting.")
        return

    plot_infos = []

    n = len(pkl_paths)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.2 * nrows))
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
        pkl_path = pkl_paths[idx]
        info = plot_counts_for_pickle(pkl_path, pool, scenario_name=scenario_name, ax=ax)
        plot_infos.append(info)
        user_id = info.get("user_id")
        if user_id and user_id != "unknown":
            ax.set_title(user_id, fontsize=10)

    if scenario_name:
        fig.suptitle(scenario_name.replace("_", " "), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is None:
        if scenario_name is None:
            scenario_name = "all"
        save_path = os.path.join(
            output_dir,
            stacked_plots_dirname,
            pool,
            aq,
            scenario_name,
            "counts_grid.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved grid to {save_path}")

def collect_participants(pkl_path):
    with open(pkl_path, "rb") as f:
        participants_count_per_round = pickle.load(f)
    if isinstance(participants_count_per_round, pd.DataFrame):
        df_wide = participants_count_per_round.copy()
        if "round" in df_wide.columns:
            df_wide = df_wide.set_index("round")
        participants = df_wide.columns.astype(str).tolist()
    elif isinstance(participants_count_per_round, dict):
        df_wide = (
            pd.DataFrame.from_dict(participants_count_per_round, orient="index")
            .fillna(0)
        )
        participants = df_wide.columns.astype(str).tolist()
    else:
        participants = []
    return participants

def run_for_pattern(pkl_paths, scenario_name):
    global GLOBAL_PARTICIPANTS, save_dir

    if not pkl_paths:
        print(f"No queried_participant_counts_*.pkl files found for {scenario_name}.")
        return
    GLOBAL_PARTICIPANTS = None
    save_dir = os.path.join(output_dir, stacked_plots_dirname, pool, aq, scenario_name)
    all_participants = set()
    for pkl_path in pkl_paths:
        all_participants.update(collect_participants(pkl_path))
    GLOBAL_PARTICIPANTS = sorted(all_participants)
    for pkl_path in sorted(pkl_paths):
        plot_counts_for_pickle(pkl_path, pool)
    plot_counts_grid_from_pkls(
        sorted(pkl_paths),
        pool,
        ncols=3,
        scenario_name=scenario_name,
    )

if all_from_list:
    for scenario_name, users in SCENARIO_USERS.items():
        scenario_pkls = []
        for user in users:
            scenario_pkls.extend(
                glob.glob(
                    os.path.join(
                        base_dir,
                        user,
                        scenario_name,
                        aq,
                        "*",
                        "queried_participant_counts_*.pkl",
                    ),
                    recursive=True,
                )
            )
        run_for_pattern(scenario_pkls, scenario_name)
else:
    all_pkls = glob.glob(
        os.path.join(base_dir, "*", fruit_scenario, aq, "*", "queried_participant_counts_*.pkl"),
        recursive=True,
    )
    run_for_pattern(all_pkls, fruit_scenario)
