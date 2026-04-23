from pathlib import Path
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.errors import EmptyDataError


ROOT_DIR = Path("Cardiomate_AL/G_SSL_augmented")
BP_MODE = True
PLOTS_DIR = ROOT_DIR / ("AUC_plots_bp" if BP_MODE else "AUC_plots")
POOLS = ["global"]
METHODS = ["random", "coreset"]
METHOD_COLORS = {
    "uncertainty": "black",
    "random": "orange",
    "coreset": "green",
}
CORESET_ONLY = False


def get_bp_users():
    users = []
    for hp_dir in sorted(Path("DATA/Cardiomate/hp").glob("hp*")):
        pid = "".join(ch for ch in hp_dir.name if ch.isdigit())
        if not pid:
            continue
        base = hp_dir.parent / f"hp{pid}"
        required = [
            base / f"hp{pid}_hr.csv",
            base / f"hp{pid}_steps.csv",
            base / f"blood_pressure_readings_ID{pid}_cleaned.csv",
        ]
        if all(path.exists() for path in required):
            users.append(pid)
    return users


FRUIT_SCENARIO_USERS = {
    "Almond_Use": ["ID11", "ID13", "ID19", "ID25", "ID28"],
    "Almond_Crave": ["ID11", "ID19", "ID25"],
    "Melon_Crave": ["ID5", "ID9", "ID12", "ID19", "ID20", "ID21", "ID27"],
    "Melon_Use": ["ID12", "ID19", "ID20", "ID27"],
    "Carrot_Crave": ["ID10", "ID11", "ID14", "ID15", "ID18", "ID25"],
    "Carrot_Use": ["ID10", "ID11", "ID13", "ID14", "ID15", "ID18", "ID26"],
    "Nectarine_Crave": ["ID10", "ID11", "ID12", "ID20", "ID21", "ID27"],
    "Nectarine_Use": ["ID10", "ID11", "ID12", "ID13", "ID20", "ID21", "ID27"],
}

SCENARIO_USERS = dict(FRUIT_SCENARIO_USERS)
if BP_MODE:
    SCENARIO_USERS["BP_spike"] = get_bp_users()

combined_random = {}


def as_scalar(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def safe_read_csv(path: Path, label: str):
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        print(f"Skipping (empty {label}): {path}")
        return None
    except Exception as exc:
        print(f"Skipping ({label} read error): {path} ({exc})")
        return None
    if df is None or len(df.columns) == 0:
        print(f"Skipping (no columns {label}): {path}")
        return None
    return df


def safe_load_json(path: Path, label: str):
    if path is None or not path.exists():
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Skipping ({label} json error): {path} ({exc})")
        return None


def safe_load_npy_scalar(path: Path, label: str):
    if path is None or not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
        flat = np.asarray(arr).reshape(-1)
        if flat.size == 0:
            print(f"Skipping (empty npy {label}): {path}")
            return None
        return float(flat[0])
    except Exception as exc:
        print(f"Skipping (invalid npy {label}): {path} ({exc})")
        return None


def get_x_series(df):
    if df is None or "Pct_Total_Labeled" not in df.columns:
        return None
    return df["Pct_Total_Labeled"]


def set_percentage_ticks(ax, x_values):
    vals = pd.to_numeric(pd.Series(x_values), errors="coerce").dropna()
    if vals.empty:
        return
    ticks = sorted(vals.unique().tolist())
    ax.set_xticks(ticks)
    # ax.set_xticklabels([f"{tick:g}" for tick in ticks])
    ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])



def plot_with_variance(ax, df, label, color):
    if df is None or "AUC_Mean" not in df.columns:
        return []
    x = get_x_series(df)
    if x is None:
        return []
    y = df["AUC_Mean"]
    ax.plot(x, y, marker="o", color=color, label=label, linewidth=3.0)
    if "AUC_STD" in df.columns:
        s = df["AUC_STD"].fillna(0.0)
        ax.errorbar(
            x,
            y,
            yerr=s,
            fmt="none",
            ecolor=color,
            elinewidth=2.5,
            capsize=4,
            capthick=2.5,
            alpha=0.9,
        )
    return x.tolist()


def find_summary_files(user, fruit_scenario, pool):
    summary_files = sorted(ROOT_DIR.glob(f"{user}/{fruit_scenario}/*/*/exp_summary.json"))
    if CORESET_ONLY:
        summary_files = [path for path in summary_files if path.parent.parent.name == "coreset"]
    if not summary_files:
        summary_files = sorted(ROOT_DIR.glob(f"{pool}/{user}/{fruit_scenario}/*/*/exp_summary.json"))
    return summary_files


def resolve_method_folder(scenario_base_dir: Path, exp_folder: Path, method_name: str, hp_suffix: str):
    method_root = scenario_base_dir / method_name
    exact = method_root / exp_folder.name
    if exact.exists():
        return exact
    matches = sorted(path for path in method_root.glob(f"UF*{hp_suffix}") if path.is_dir())
    return matches[0] if len(matches) == 1 else None


def load_method_progress(method_folder: Path, label: str):
    if method_folder is None:
        print(f"Skipping (missing {label} folder): {method_folder}")
        return None
    al_path = method_folder / "al_progress.csv"
    if not al_path.exists():
        print(f"Skipping (missing {label} AL): {al_path}")
        return None
    return safe_read_csv(al_path, f"{label} AL")


def load_aggregated_auc(method_name: str, hp_name: str, pool: str, method_folder: Path):
    candidates = [ROOT_DIR / pool / "aggregates" / method_name / hp_name / "auc_per_round_aggregated.csv"]
    if method_folder is not None:
        candidates.append(method_folder / "_aggregates" / "auc_per_round_aggregated.csv")
    for path in candidates:
        df = safe_read_csv(path, f"aggregated AUC ({method_name})")
        if df is not None:
            return df
    return None


def load_user_scenario(user, fruit_scenario, pool):
    records = []
    for summary_path in find_summary_files(user, fruit_scenario, pool):
        exp_folder = summary_path.parent
        method_dir = exp_folder.parent
        scenario_base_dir = method_dir.parent

        summary = safe_load_json(summary_path, "exp_summary")
        if summary is None:
            continue

        unlabeled_frac = float(as_scalar(summary["unlabeled_frac"]))
        dropout_rate = float(as_scalar(summary["dropout_rate"]))
        t_raw = as_scalar(summary.get("T"))
        k_val = int(as_scalar(summary["K"]))
        budget = int(as_scalar(summary["Budget"]))
        hp_folder = exp_folder.name
        hp_suffix = f"_K{k_val}_B{budget}_DR{int(dropout_rate * 100)}"

        upper_bound_auc = safe_load_npy_scalar(exp_folder / "upper_bound_auc.npy", "upper_bound_auc")
        if upper_bound_auc is None:
            print("Skipping (missing upper bound): upper_bound_auc.npy not found")
            continue

        uncertainty_folder = ROOT_DIR / user / fruit_scenario / "uncertainty" / hp_folder
        random_folder = resolve_method_folder(scenario_base_dir, exp_folder, "random", hp_suffix)
        coreset_folder = resolve_method_folder(scenario_base_dir, exp_folder, "coreset", hp_suffix)

        al_uncertainty = None if CORESET_ONLY else load_method_progress(uncertainty_folder, "uncertainty")
        al_random = None if CORESET_ONLY else load_method_progress(random_folder, "random")
        al_coreset = load_method_progress(coreset_folder, "coreset")

        aggregated_auc_by_method = {}
        method_sources = {
            "random": (random_folder.name if random_folder is not None else hp_folder, random_folder),
            "coreset": (coreset_folder.name if coreset_folder is not None else hp_folder, coreset_folder),
        }
        for method_name, (hp_name, method_folder) in method_sources.items():
            agg_df = load_aggregated_auc(method_name, hp_name, pool, method_folder)
            if agg_df is not None:
                aggregated_auc_by_method[method_name] = agg_df

        records.append(
            {
                "user": user,
                "fruit_scenario": fruit_scenario,
                "unlabeled_frac": unlabeled_frac,
                "dropout_rate": dropout_rate,
                "K": k_val,
                "T": int(t_raw) if t_raw is not None else None,
                "upper_bound_auc": upper_bound_auc,
                "uncertainty": al_uncertainty,
                "random": al_random,
                "coreset": al_coreset,
                "aggregated_auc_by_method": aggregated_auc_by_method,
                "hp_folder": hp_folder,
            }
        )

        if al_random is not None:
            combined_random.setdefault(f"{user}_{fruit_scenario}", al_random.copy())

    return records


def plot_aggregate_auc(rec, hp_plot_dir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plotted_x_values = []
    for method_name in METHODS:
        agg_df = rec.get("aggregated_auc_by_method", {}).get(method_name)
        plotted_x_values.extend(
            plot_with_variance(ax, agg_df, method_name, METHOD_COLORS[method_name])
        )
    if plotted_x_values:
        set_percentage_ticks(ax, plotted_x_values)
        ax.set_title("Aggregated AUC")
        ax.set_xlabel("% Total Data Labeled")
        ax.set_ylabel("AUC Mean")
        ax.legend()
        fig.tight_layout()
        fig.savefig(hp_plot_dir / "aggregate_auc.png", dpi=150)
    plt.close(fig)


def plot_user_auc(rec, hp_plot_dir: Path):
    if not any(rec.get(name) is not None for name in ["uncertainty", "random", "coreset"]):
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plotted_x_values = []
    plotted_x_values.extend(plot_with_variance(ax, rec.get("uncertainty"), "uncertainty", METHOD_COLORS["uncertainty"]))
    plotted_x_values.extend(plot_with_variance(ax, rec.get("random"), "random", METHOD_COLORS["random"]))
    plotted_x_values.extend(plot_with_variance(ax, rec.get("coreset"), "coreset", METHOD_COLORS["coreset"]))

    if rec.get("upper_bound_auc") is not None:
        ax.axhline(
            rec["upper_bound_auc"],
            linestyle="--",
            label=f"upper_bound={rec['upper_bound_auc']:.3f}",
            color="red",
            alpha=0.7,
        )

    if plotted_x_values:
        set_percentage_ticks(ax, plotted_x_values)
    ax.set_xlabel("Total % Data Labeled")
    ax.set_ylabel("AUC Mean")
    ax.set_title(f"Per-User AL ({rec['user']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hp_plot_dir / f"{rec['user']}.png", dpi=150)
    plt.close(fig)


def plot_for_user_scenario(user, fruit_scenario, pool):
    records = load_user_scenario(user, fruit_scenario, pool)
    if not records:
        return

    scenario_dir = PLOTS_DIR / fruit_scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        hp_plot_dir = scenario_dir / rec["hp_folder"]
        hp_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_aggregate_auc(rec, hp_plot_dir)
        plot_user_auc(rec, hp_plot_dir)


def plot_grid_by_participant(source_dir, ncols=3, rows_per_page=3, output_pdf="grid_by_participant.pdf"):
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    png_paths = sorted(
        path
        for path in source_dir.glob("*.png")
        if not (path.stem.startswith("aggregate_auc_") and path.stem != "aggregate_auc")
    )
    if not png_paths:
        print(f"No subplot PNGs found under {source_dir}. Skipping grid PDF.")
        return

    per_page = max(1, int(ncols) * int(rows_per_page))
    output_path = source_dir / output_pdf

    with PdfPages(output_path) as pdf:
        for start in range(0, len(png_paths), per_page):
            page_imgs = png_paths[start : start + per_page]
            nrows = int(np.ceil(len(page_imgs) / ncols))
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(4.5 * ncols, 3.6 * nrows),
            )
            axes = np.array(axes).reshape(nrows, ncols)

            for idx in range(nrows * ncols):
                row, col = divmod(idx, ncols)
                ax = axes[row, col]
                if idx >= len(page_imgs):
                    ax.axis("off")
                    continue
                img_path = page_imgs[idx]
                ax.imshow(plt.imread(img_path))
                ax.set_title(img_path.stem, fontsize=8)
                ax.axis("off")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved grid PDF: {output_path}")


def plot_grids_for_all_hp_folders():
    for fruit_scenario in SCENARIO_USERS:
        scenario_dir = PLOTS_DIR / fruit_scenario
        if not scenario_dir.exists():
            continue
        for hp_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir()):
            if any(hp_dir.glob("*.png")):
                plot_grid_by_participant(source_dir=hp_dir)


def plot_all_users_by_scenario(results_map, suffix):
    for scenario, users in SCENARIO_USERS.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        plotted_x_values = []
        any_plotted = False

        for user in users:
            df = results_map.get(f"{user}_{scenario}")
            x = get_x_series(df)
            if df is None or x is None or "AUC_Mean" not in df.columns:
                continue
            ax.plot(x, df["AUC_Mean"], marker="o", label=user)
            plotted_x_values.extend(x.tolist())
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        set_percentage_ticks(ax, plotted_x_values)
        ax.set_xlabel("Total % Data Labeled")
        ax.set_ylabel("AUC Mean")
        ax.set_title(f"{scenario.replace('_', ' ')} - {suffix.title()} AUC Mean per User")
        ax.legend()
        fig.tight_layout()

        out_path = (PLOTS_DIR / scenario) / f"all_users_{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


for fruit_scenario, users in SCENARIO_USERS.items():
    for user in users:
        for pool in POOLS:
            plot_for_user_scenario(user, fruit_scenario, pool)

plot_grids_for_all_hp_folders()
plot_all_users_by_scenario(combined_random, "random")
