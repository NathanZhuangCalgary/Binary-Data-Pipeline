import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import font_manager as fm

# Data
comparisons = [
    "RF-positive SS vs RA",
    "RF-positive SS vs RF-negative SS",
    "RF-negative SS vs RA",
    "SS vs RA"
]
auc_mean = np.array([0.362, 0.618, 0.639, 0.557])
auc_sd = np.array([0.102, 0.130, 0.093, 0.088])

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Simulate distributions
np.random.seed(42)
sim_data = []
for name, mean, sd in zip(comparisons, auc_mean, auc_sd):
    vals = np.random.normal(mean, sd, 500)
    vals = np.clip(vals, 0, 1)
    sim_data.extend(zip([name]*len(vals), vals))
df = pd.DataFrame(sim_data, columns=["Comparison", "ROC AUC"])

# Find a Times New Roman font file on the system
times_font = fm.FontProperties(family='Times New Roman')
plt.rcParams['font.family'] = times_font.get_name()

# Plot
plt.figure(figsize=(10,6))
sns.set(style="whitegrid", font_scale=1.3)

# Violin plots with narrower width and spacing
sns.violinplot(x="Comparison", y="ROC AUC", data=df, inner=None,
               palette=colors, cut=0, linewidth=1, width=0.7)

# Overlay mean ± SD
sns.pointplot(x="Comparison", y="ROC AUC", data=df, estimator=np.mean,
              color="black", errorbar="sd", join=False, capsize=0.2)

# Add mean ± SD text slightly above error bars
for i, (mean, sd) in enumerate(zip(auc_mean, auc_sd)):
    plt.text(i, min(mean+sd+0.03, 1), f"{mean:.3f} ± {sd:.3f}",
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Reference line
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)

# Labels and formatting
plt.title("ROC AUC Distribution per Comparison", fontsize=16, weight='bold')
plt.ylabel("ROC AUC", fontsize=14)
plt.xlabel("")
plt.ylim(0, 1)

# Remove x-axis labels
plt.xticks([])

sns.despine(trim=True)

# Legend
handles = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]
plt.legend(handles + [plt.Line2D([0], [0], color='gray', linestyle='--')],
           comparisons + ["Random Classifier"], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()
