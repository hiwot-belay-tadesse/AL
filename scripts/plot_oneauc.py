import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob

root = Path("Cardiomate_AL/")
pattern = str(root / "GS" /"global_supervised"/"*" /"BP_spike"/ "uncertainty" / "*" / "al_progress.csv")

# pattern = str(root / "*" / "global_supervised" / "*" / "uncertainty" / "*" / "al_progress.csv")
# pattern = str(root / "global_SSL" / "*" / "global" / "*" / "uncertainty" / "*" / "al_progress.csv")

csvs = glob.glob(pattern)
breakpoint()
if not csvs:
    print(f"No files found for pattern: {pattern}")
    raise SystemExit(0)
groups = {}
for csv_path in csvs:
    parts = Path(csv_path).parts
    # .../<root>/<user>/global_supervised/<fruit_scenario>/uncertainty/<hp>/al_progress.csv
    if len(parts) < 6:
        continue
    user = parts[-5]
    fruit_scenario = parts[-4]
    groups.setdefault(fruit_scenario, []).append((user, csv_path))

for fruit_scenario, entries in groups.items():
    plt.figure(figsize=(9, 6))
    distinct_colors = [
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
    participants = sorted({u for u, _ in entries})
    participant_to_color = {
        p: distinct_colors[i % len(distinct_colors)]
        for i, p in enumerate(participants)
    }
    for user, csv_path in entries:
        df = pd.read_csv(csv_path)
        if "AUC_Mean" not in df.columns or "AUC_STD" not in df.columns:
            continue
        xcol = "round" if "round" in df.columns else df.index
        x = df[xcol]
        y = df["AUC_Mean"]
        yerr = df["AUC_STD"]
        color = participant_to_color.get(user, "tab:gray")
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.2, label=user, color=color)

    plt.xlabel("Round")
    plt.ylabel("AUC Mean")
    plt.title(f"AUC Mean ± Std per User (Uncertainty) - {fruit_scenario}")
    plt.legend()
    plt.tight_layout()
    out_path = root / f"auc_users_{fruit_scenario}.png"
    # plt.savefig(out_path, dpi=150)
    plt.show()
    print("saved plot in ",{out_path})
    plt.close()
