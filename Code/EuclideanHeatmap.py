import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Updated data with new Euclidean distances and CIs
data = {
    "Comparison": ["RF-positive SS vs RA", "RF-negative SS vs RA", "SS vs RA"],
    "Distance": [0.195, 0.348, 0.223],
    "CI_Lower": [0.068, 0.197, 0.095],
    "CI_Upper": [0.335, 0.507, 0.362]
}

df = pd.DataFrame(data)

# Create combined annotation with distance and CI
df["Annotation"] = df.apply(lambda row: f"{row['Distance']:.3f}\n({row['CI_Lower']:.3f}-{row['CI_Upper']:.3f})", axis=1)

# Set Comparison as index
df.set_index("Comparison", inplace=True)

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[["Distance"]], 
            annot=df[["Annotation"]].values,  # Use custom annotations
            fmt='',  # Empty since we're providing formatted text
            cmap="YlOrRd", 
            cbar_kws={'label': 'Euclidean Distance'}, 
            linewidths=0.5,
            vmin=0,  # Set minimum for color scale
            vmax=0.5)  # Set maximum for color scale

plt.title("Euclidean Distance Between Groups\n(Values: Distance with 95% Bootstrap CI)", 
          fontsize=12, fontweight='bold', pad=20)
plt.ylabel("")
plt.tight_layout()
plt.show()

# Alternative version with just distances in heatmap and CI as separate text
plt.figure(figsize=(6, 4))
ax = sns.heatmap(df[["Distance"]], 
            annot=True, 
            fmt='.3f',
            cmap="YlOrRd", 
            cbar_kws={'label': 'Euclidean Distance'}, 
            linewidths=0.5,
            vmin=0,
            vmax=0.5)

# Add CI as additional text below the main values
for i, (idx, row) in enumerate(df.iterrows()):
    ax.text(0.5, i + 0.7, f"({row['CI_Lower']:.3f}-{row['CI_Upper']:.3f})", 
            ha='center', va='top', fontsize=9, color='blue')

plt.title("Euclidean Distance Between Groups\n(Blue text: 95% Bootstrap Confidence Intervals)", 
          fontsize=12, fontweight='bold', pad=20)
plt.ylabel("")
plt.tight_layout()
plt.show()

# Clean version with just distances (simplest)
plt.figure(figsize=(5, 3))
sns.heatmap(df[["Distance"]], 
            annot=True, 
            fmt='.3f',
            cmap="YlOrRd", 
            cbar_kws={'label': 'Euclidean Distance'}, 
            linewidths=0.5,
            vmin=0,
            vmax=0.5)

plt.title("Euclidean Distance Between Groups", fontsize=12, fontweight='bold')
plt.ylabel("")
plt.tight_layout()
plt.show()

# Print the data for reference
print("Distance Data with Confidence Intervals:")
print(df[['Distance', 'CI_Lower', 'CI_Upper']])

comparisons = ['RF+ SS vs RA', 'RF+ SS vs RF- SS', 'RF- SS vs RA', 'SS vs RA']
auc_means = [0.362, 0.618, 0.639, 0.557]
auc_std = [0.102, 0.130, 0.093, 0.088]

plt.figure(figsize=(10, 6))
bars = plt.bar(comparisons, auc_means, yerr=auc_std, capsize=5, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
plt.axhline(y=0.5, color='red', linestyle='--', label='Random (AUC=0.5)')
plt.ylabel('ROC AUC ± SD')
plt.title('Classifier Performance: ROC AUC with Standard Deviation')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()