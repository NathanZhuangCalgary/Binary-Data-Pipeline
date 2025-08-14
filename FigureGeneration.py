import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# ---------- t-values for Autoantibody Comparison ----------
antibodies = ['ACPA','ANA','Anti-dsDNA','Anti-Sm']
t_values_rfpos = [-0.194,0.691,-0.372,-0.194]
t_values_rfneg = [0.514,2.424,2.842,-0.673]

x = np.arange(len(antibodies))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, t_values_rfpos, width, label='RF+ SS', color='green')
ax.bar(x + width/2, t_values_rfneg, width, label='RF- SS', color='blue')

ax.set_xticks(x)
ax.set_xticklabels(antibodies)
ax.set_ylabel('t-value')
ax.set_xlabel('Autoantibody')
ax.set_title('t-values for Autoantibody Comparison vs RA')
ax.axhline(y=1.96, color='red', linestyle='--', label='Critical t = ±1.96')
ax.axhline(y=-1.96, color='red', linestyle='--')
ax.grid(which='major', linestyle='-', linewidth=0.7, alpha=0.7)
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
ax.minorticks_on()
ax.legend()
plt.show()


# ---------- Euclidean Distances ----------
distances = pd.DataFrame({
    'Euclidean Distance': [0.1641, 0.6443]
}, index=['RF+ SS','RF- SS'])

sns.heatmap(distances, annot=True, cmap='coolwarm', cbar_kws={'label':'Euclidean Distance'})
plt.title('Euclidean Distances of Autoantibody Profiles vs RA')
plt.xlabel('Feature')
plt.ylabel('SS Subgroup')
plt.show()


# ---------- Tukey HSD for Anti-dsDNA ----------
groups = ['RA vs RF+ SS', 'RA vs RF- SS', 'RF+ SS vs RF- SS']
differences = [-0.0361, 0.2399, 0.276]
significant = [False, True, True]

colors = ['red' if sig else 'gray' for sig in significant]

plt.figure(figsize=(7,4))
plt.bar(groups, differences, color=colors)
plt.axhline(y=0, color='black', linewidth=1)  # baseline
plt.ylabel('Mean Difference (Anti-dsDNA)')
plt.xlabel('Group Comparison')
plt.title("Tukey HSD: Anti-dsDNA Comparisons")
plt.grid(axis='y', which='major', linestyle='-', linewidth=0.7, alpha=0.7)
plt.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
plt.minorticks_on()
plt.show()


# ---------- Cohen's d Effect Sizes ----------
d_rfpos = [0.038,-0.131,0.072,0.038]
d_rfneg = [-0.092,-0.404,-0.494,0.122]

x = np.arange(len(antibodies))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.barh(x - width/2, d_rfpos, width, label='RF+ SS', color='green')
ax.barh(x + width/2, d_rfneg, width, label='RF- SS', color='blue')

ax.set_yticks(x)
ax.set_yticklabels(antibodies)
ax.set_xlabel("Cohen's d")
ax.set_ylabel("Autoantibody")
ax.set_title("Effect Sizes for Autoantibody Differences")
ax.axvline(x=0, color='black', linewidth=1)  # baseline
ax.grid(which='major', axis='x', linestyle='-', linewidth=0.7, alpha=0.7)
ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.5)
ax.minorticks_on()
ax.legend()
plt.show()
