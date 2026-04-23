"""
Create a box plot of Train AUC by pipeline from aggregated BP results.
Run: python scripts/plot_train_auc.py
Input: BP_SPIKE_PRED/user_pipeline_auc_wide.csv
Output: supp_plots/train_auc_boxplot.png

"""

import pandas as pd
import matplotlib.pyplot as plt

# Load wide table
df = pd.read_csv("BP_SPIKE_PRED/user_pipeline_auc_wide.csv")

# Pick only train-AUC columns
train_cols = [c for c in df.columns if c.endswith("_AUC_Train")]

# Convert to long format for seaborn/matplotlib boxplot
long_df = df.melt(
    id_vars=["uid"],
    value_vars=train_cols,
    var_name="pipeline",
    value_name="train_auc"
)

# Clean labels and numeric conversion
long_df["pipeline"] = long_df["pipeline"].str.replace("_AUC_Train", "", regex=False)
long_df["train_auc"] = pd.to_numeric(long_df["train_auc"], errors="coerce")
long_df = long_df.dropna(subset=["train_auc"])

# Boxplot
plt.figure(figsize=(8, 4))
plt.boxplot(
    [long_df.loc[long_df["pipeline"] == p, "train_auc"].values
     for p in sorted(long_df["pipeline"].unique())],
    labels=sorted(long_df["pipeline"].unique())
)
plt.ylim(0, 1)
plt.ylabel("Train AUC")
plt.title("Train AUC Distribution by Pipeline")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig("supp_plots/train_auc_boxplot.png")