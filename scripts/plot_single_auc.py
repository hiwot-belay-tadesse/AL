from multiprocessing import pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- local config (edit these) ----
pool = "global"
ROOT_DIR = f"Cardiomate_AL/G_SSL_augmented/{pool}"
# ROOT_DIR = f"Ban_AL/Global/{pool}"
FRUIT_SCENARIO = "BP_spike"
# FRUIT_SCENARIO = "Melon_Crave"
OUT_DIR = f"/Users/hiwotbelaytadesse/Desktop/Banaware_AL/Cardiomate_AL/AUC_plot_bp/{pool}_augmented"
# OUT_DIR = f"/Users/hiwotbelaytadesse/Desktop/Banaware_AL/Ban_AL/AUC_plot_bp/{pool}_augmented"

  # e.g. "Cardiomate_AL/G_SSL_with_target_quota/global/25/BP_spike/plots"


def _read_al_progress(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "round" not in df.columns:
        raise ValueError(f"'round' column missing: {csv_path}")
    auc_col = "AUC_Mean" if "AUC_Mean" in df.columns else "auc_mean" if "auc_mean" in df.columns else None
    if auc_col is None:
        raise ValueError(f"No AUC column in: {csv_path}")
    std_col = None
    for c in ("AUC_Std", "auc_std", "AUC_STD"):
        if c in df.columns:
            std_col = c
            break

    out = df.rename(columns={auc_col: "auc"}).copy()
    if std_col is not None:
        out = out.rename(columns={std_col: "auc_std"})
    else:
        out["auc_std"] = np.nan
    return out


def plot_random_vs_coreset_from_root(
    root_dir: str,
    user: str,
    fruit_scenario: str,
    out_dir: str | None = None,
) -> list[Path]:
    """
    Reads:
      <root_dir>/<user>/<fruit_scenario>/random/<hp>/al_progress.csv
      <root_dir>/<user>/<fruit_scenario>/coreset/<hp>/al_progress.csv
    and writes one comparison plot per matching <hp>.
    """
    base = Path(root_dir) / str(user) / str(fruit_scenario)
    
    random_root = base / "random"
    coreset_root = base / "coreset"
    uncertainty_root = base / "uncertainty"

    random_csv = {p.parent.name: p for p in random_root.glob("*/al_progress.csv")}
    coreset_csv = {p.parent.name: p for p in coreset_root.glob("*/al_progress.csv")}
    uncertainty_csv = {p.parent.name: p for p in uncertainty_root.glob("*/al_progress.csv")}

    common_hp = sorted(set(random_csv) & set(coreset_csv))
    if not common_hp:
        raise FileNotFoundError(
            f"No matching HP folders with al_progress.csv under:\n"
            f"  {random_root}\n"
            f"  {coreset_root}"
        )

    out_base = Path(out_dir) if out_dir else (base / "plots")
    out_base.mkdir(parents=True, exist_ok=True)
    saved = []

    for hp in common_hp:
        df_r = _read_al_progress(random_csv[hp])
        df_c = _read_al_progress(coreset_csv[hp])
        df_u = _read_al_progress(uncertainty_csv[hp]) if hp in uncertainty_csv else None

        upper_bound_auc = None
        for folder in (random_root / hp, coreset_root / hp, uncertainty_root / hp):
            ub_path = folder / "upper_bound_auc.npy"
            if ub_path.exists():
                upper_bound_auc = float(np.load(ub_path))
                break

        plt.figure(figsize=(7, 4))
        plt.errorbar(
            df_r["round"], df_r["auc"], yerr=df_r["auc_std"], color="darkorange",
            marker="o", linewidth=1.6, capsize=3, label="random"
        )
        plt.errorbar(
            df_c["round"], df_c["auc"], yerr=df_c["auc_std"], color="seagreen",
            marker="o", linewidth=1.6, capsize=3, label="coreset"
        )
        if df_u is not None:
            plt.errorbar(
                df_u["round"], df_u["auc"], yerr=df_u["auc_std"], color="tab:blue",
                marker="o", linewidth=1.6, capsize=3, label="uncertainty"
            )
            
        if upper_bound_auc is not None:
            plt.axhline(
                upper_bound_auc,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"upper_bound={upper_bound_auc:.3f}",
            )
        plt.xlabel("Round")
        plt.ylabel("AUC")
        # plt.title(f"{user} {fruit_scenario} ({hp})")
        plt.title(f"{user}")
        plt.legend()
        # plt.grid(alpha=0.25)
        plt.tight_layout()


        out_path = out_base / f"{user}_{fruit_scenario}_{hp}_random_vs_coreset.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved.append(out_path)

    return saved


def discover_users(root_dir: str, fruit_scenario: str) -> list[str]:
    root = Path(root_dir)
    users = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / fruit_scenario).exists():
            users.append(p.name)
    return users


def main():
    users = discover_users(ROOT_DIR, FRUIT_SCENARIO)
    if not users:
        print(f"No users found under {ROOT_DIR} for scenario {FRUIT_SCENARIO}")
        return

    all_files = []
    for user in users:
        try:
            files = plot_random_vs_coreset_from_root(
                root_dir=ROOT_DIR,
                user=user,
                fruit_scenario=FRUIT_SCENARIO,
                out_dir=OUT_DIR,
            )

            all_files.extend(files)
        except FileNotFoundError:
            continue

    for f in all_files:
        print(f"Saved: {f}")


if __name__ == "__main__":
    main()
