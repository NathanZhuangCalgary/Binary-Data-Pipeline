import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Create clinical_figures directory
os.makedirs("clinical_figures", exist_ok=True)

def create_clinical_figures():
    """Create all clinical figures with your data"""
    
    # FIGURE 1: Patient Cohort Overview
    plt.figure(figsize=(10, 6))
    
    groups = ['RA', 'SS', 'RF+ SS', 'RF- SS']
    counts = [766, 463, 433, 30]  # Your data
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = plt.bar(groups, counts, color=colors, alpha=0.8)
    plt.title('Patient Cohort Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure1_Patient_Cohorts_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 2: Model Performance Comparison with Standard Deviation Error Bars
    comparisons = ['RF+ SS vs RA', 'RF+ SS vs RF- SS', 'RF- SS vs RA', 'SS vs RA']
    
    # Your performance data
    accuracy = [0.485, 0.591, 0.553, 0.486]
    balanced_accuracy = [0.485, 0.547, 0.547, 0.486]
    roc_auc = [0.490, 0.580, 0.572, 0.482]
    
    # Your standard deviations
    accuracy_std = [0.028, 0.085, 0.071, 0.029]
    balanced_std = [0.027, 0.097, 0.087, 0.024]
    roc_std = [0.033, 0.102, 0.081, 0.026]
    
    plt.figure(figsize=(12, 6))
    
    x_pos = np.arange(len(comparisons))
    width = 0.25
    
    # Plot with error bars
    plt.bar(x_pos - width, accuracy, width, yerr=accuracy_std, capsize=5,
            label='Accuracy', alpha=0.8, color='#2E86AB')
    plt.bar(x_pos, balanced_accuracy, width, yerr=balanced_std, capsize=5,
            label='Balanced Accuracy', alpha=0.8, color='#A23B72')
    plt.bar(x_pos + width, roc_auc, width, yerr=roc_std, capsize=5,
            label='ROC AUC', alpha=0.8, color='#F18F01')
    
    plt.ylabel('Performance Score')
    plt.title('Model Performance Across Comparisons\n(Error bars: Standard Deviation from CV)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x_pos, comparisons, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 0.8)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Chance')
    
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure2_Model_Performance_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 3: Statistical Significance Analysis
    antibodies = ['HLA_B27', 'ANTI_DSDNA', 'ANTI_SM']
    
    chi2_pvals = [0.9823, 0.6536, 0.4176]
    fisher_pvals = [0.9531, 0.6378, 0.4091]
    odds_ratios = [1.010, 0.942, 1.108]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: P-values
    x_pos = np.arange(len(antibodies))
    width = 0.35
    
    ax1.bar(x_pos - width/2, chi2_pvals, width, label='Chi-squared', alpha=0.8, color='#2E86AB')
    ax1.bar(x_pos + width/2, fisher_pvals, width, label="Fisher's exact", alpha=0.8, color='#A23B72')
    
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance (0.05)')
    ax1.set_ylabel('P-value')
    ax1.set_title('P-values by Antibody', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(antibodies)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    
    # Panel B: Odds ratios
    bars = ax2.bar(antibodies, odds_ratios, alpha=0.8, color='#F18F01')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No effect (OR=1)')
    ax2.set_ylabel('Odds Ratio')
    ax2.set_title('Fisher\'s Exact Test Odds Ratios', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 2.0)
    
    # Add value labels
    for bar, or_val in zip(bars, odds_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{or_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure3_Statistical_Analysis_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 4: Antibody Prevalence Patterns
    antibodies = ['HLA_B27', 'ANTI_DSDNA', 'ANTI_SM']
    groups = ['RF+ SS', 'RF- SS', 'SS', 'RA']
    
    # Your prevalence data
    prevalence_data = np.array([
        [0.490, 0.520, 0.457],  # RF+ SS
        [0.633, 0.367, 0.400],  # RF- SS
        [0.499, 0.510, 0.454],  # SS
        [0.501, 0.495, 0.479]   # RA
    ])
    
    plt.figure(figsize=(10, 6))
    
    x_pos = np.arange(len(antibodies))
    width = 0.2
    
    for i, group in enumerate(groups):
        plt.bar(x_pos + i*width, prevalence_data[i], width, label=group, alpha=0.8)
    
    plt.xlabel('Antibodies')
    plt.ylabel('Prevalence')
    plt.title('Antibody Prevalence by Patient Group', fontsize=14, fontweight='bold')
    plt.xticks(x_pos + width*1.5, antibodies)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure4_Antibody_Prevalence_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 5: Euclidean Distance Heatmap
    groups = ['RF+ SS', 'RF- SS', 'SS', 'RA']
    
    # Your distance data
    euclidean_matrix = np.array([
        [0.000, 0.247, 0.055, 0.059],  # RF+ SS distances
        [0.247, 0.000, 0.055, 0.247],  # RF- SS distances  
        [0.055, 0.055, 0.000, 0.055],  # SS distances
        [0.059, 0.247, 0.055, 0.000]   # RA distances
    ])
    
    # Create mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(euclidean_matrix, dtype=bool), k=1)
    
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(euclidean_matrix, 
                xticklabels=groups,
                yticklabels=groups,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Euclidean Distance'},
                mask=mask,
                square=True)
    
    plt.title('Euclidean Distance Between Patient Groups', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure5_Euclidean_Heatmap_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 6: Performance Summary Heatmap with Standard Deviation Annotation
    comparisons = ['RF+ SS vs RA', 'RF+ SS vs RF- SS', 'RF- SS vs RA', 'SS vs RA']
    metrics = ['Accuracy', 'Balanced\nAccuracy', 'ROC AUC']
    
    performance_data = np.array([
        [0.485, 0.591, 0.553, 0.486],
        [0.485, 0.547, 0.547, 0.486],
        [0.490, 0.580, 0.572, 0.482]
    ])
    
    # Standard deviations for annotation
    performance_std = np.array([
        [0.028, 0.085, 0.071, 0.029],
        [0.027, 0.097, 0.087, 0.024],
        [0.033, 0.102, 0.081, 0.026]
    ])
    
    plt.figure(figsize=(10, 6))
    
    im = plt.imshow(performance_data, cmap='RdYlBu', aspect='auto', vmin=0.3, vmax=0.7)
    
    plt.xticks(np.arange(len(comparisons)), comparisons, rotation=45, ha='right')
    plt.yticks(np.arange(len(metrics)), metrics)
    plt.title('Model Performance Heatmap\n(Values show mean ± standard deviation)', 
              fontsize=14, fontweight='bold')
    
    # Add text annotations with mean ± std
    for i in range(len(metrics)):
        for j in range(len(comparisons)):
            mean_val = performance_data[i, j]
            std_val = performance_std[i, j]
            text = plt.text(j, i, f'{mean_val:.3f} ± {std_val:.3f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=8)
    
    plt.colorbar(im, shrink=0.8, label='Performance Score')
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure6_Performance_Heatmap_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 7: Distance Analysis with Confidence Intervals
    comparisons = ['RF+ SS vs RA', 'RF- SS vs RA', 'SS vs RA']
    euclidean_dist = [0.059, 0.247, 0.055]  # Your data
    ci_lower = [0.019, 0.108, 0.018]  # Your CI lower bounds
    ci_upper = [0.108, 0.389, 0.101]  # Your CI upper bounds
    
    plt.figure(figsize=(10, 6))
    
    x_pos = np.arange(len(comparisons))
    
    # Calculate error bars
    errors_lower = [dist - lower for dist, lower in zip(euclidean_dist, ci_lower)]
    errors_upper = [upper - dist for dist, upper in zip(euclidean_dist, ci_upper)]
    errors = [errors_lower, errors_upper]
    
    bars = plt.bar(x_pos, euclidean_dist, yerr=errors, capsize=10, 
                   alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
    
    plt.ylabel('Euclidean Distance')
    plt.title('Euclidean Distance Between Groups\n(Error bars: 95% Bootstrap Confidence Intervals)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x_pos, comparisons)
    
    # Add value labels with CI
    for i, (bar, dist, lower, upper) in enumerate(zip(bars, euclidean_dist, ci_lower, ci_upper)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{dist:.3f}\n({lower:.3f}-{upper:.3f})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure7_Distance_CI_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 8: ROC AUC Distribution with Violin Plots
    comparisons = [
        "RF-positive SS vs RA",
        "RF-positive SS vs RF-negative SS", 
        "RF-negative SS vs RA",
        "SS vs RA"
    ]
    auc_mean = np.array([0.490, 0.580, 0.572, 0.482])  # Your ROC AUC data
    auc_sd = np.array([0.033, 0.102, 0.081, 0.026])    # Your ROC AUC SD
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # Simulate distributions
    np.random.seed(42)
    sim_data = []
    for name, mean, sd in zip(comparisons, auc_mean, auc_sd):
        vals = np.random.normal(mean, sd, 500)
        vals = np.clip(vals, 0, 1)
        sim_data.extend(zip([name]*len(vals), vals))
    df = pd.DataFrame(sim_data, columns=["Comparison", "ROC AUC"])
    
    # Plot
    plt.figure(figsize=(10,6))
    sns.set(style="whitegrid", font_scale=1.3)
    
    # Violin plots
    sns.violinplot(x="Comparison", y="ROC AUC", data=df, inner=None,
                   palette=colors, cut=0, linewidth=1, width=0.7)
    
    # Overlay mean ± SD
    sns.pointplot(x="Comparison", y="ROC AUC", data=df, estimator=np.mean,
                  color="black", errorbar="sd", join=False, capsize=0.2)
    
    # Add mean ± SD text
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
    plt.savefig('clinical_figures/Figure8_ROC_AUC_Distribution_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 9: Simple ROC AUC Bar Plot
    comparisons = ['RF+ SS vs RA', 'RF+ SS vs RF- SS', 'RF- SS vs RA', 'SS vs RA']
    auc_means = [0.490, 0.580, 0.572, 0.482]  # Your data
    auc_std = [0.033, 0.102, 0.081, 0.026]    # Your data
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparisons, auc_means, yerr=auc_std, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random (AUC=0.5)')
    plt.ylabel('ROC AUC ± SD')
    plt.title('Classifier Performance: ROC AUC with Standard Deviation')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('clinical_figures/Figure9_ROC_AUC_Bar_clinical.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All clinical figures generated successfully!")
    print("Figures saved in 'clinical_figures/' directory:")
    for i in range(1, 10):
        print(f"  Figure{i}_*_clinical.png")

if __name__ == "__main__":
    create_clinical_figures()