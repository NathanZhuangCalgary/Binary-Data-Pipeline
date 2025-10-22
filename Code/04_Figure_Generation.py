"""
Simplified Visualization of Binary Antibody Analysis Results
---------------------------------------------------
Creates clean, simple figures for all analysis results with error bars:
- Patient cohort overview
- Model performance comparisons with stdv error bars
- Statistical significance
- Antibody prevalence patterns
- Distance analysis with Euclidean heatmap and CI error bars

Author: Nathan Zhuang
Version: Updated Results Visualization with Error Bars
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set style for clean figures
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_all_figures():
    """Create simplified visualization of all analysis results with error bars"""
    
    # Create output directory
    import os
    os.makedirs("results_figures", exist_ok=True)
    
    # =========================================================================
    # FIGURE 1: Patient Cohort Overview
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    groups = ['RA', 'SS', 'RF+ SS', 'RF- SS']
    counts = [91, 85, 38, 47]
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
    plt.savefig('results_figures/Figure1_Patient_Cohorts.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 2: Model Performance Comparison with Standard Deviation Error Bars
    # =========================================================================
    comparisons = ['RF+ SS vs RA', 'RF+ SS vs RF- SS', 'RF- SS vs RA', 'SS vs RA']
    
    accuracy = [0.391, 0.608, 0.635, 0.566]
    balanced_accuracy = [0.364, 0.602, 0.639, 0.568]
    roc_auc = [0.362, 0.618, 0.639, 0.557]
    
    # Standard deviations
    accuracy_std = [0.091, 0.113, 0.073, 0.085]
    balanced_std = [0.080, 0.115, 0.073, 0.085]
    roc_std = [0.102, 0.130, 0.093, 0.088]
    
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
    plt.title('Model Performance Across Comparisons\n(Error bars: Standard Deviation from 10×5-fold CV)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x_pos, comparisons, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 0.8)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Chance')
    
    plt.tight_layout()
    plt.savefig('results_figures/Figure2_Model_Performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 3: Statistical Significance Analysis
    # =========================================================================
    antibodies = ['ACPA', 'ANA', 'ANTI_DSDNA', 'ANTI_SM']
    
    chi2_pvals = [0.9419, 0.0880, 0.1621, 0.6830]
    fisher_pvals = [0.8790, 0.0689, 0.1325, 0.6497]
    odds_ratios = [0.933, 0.531, 0.624, 1.185]
    
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
    plt.savefig('results_figures/Figure3_Statistical_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 4: Antibody Prevalence Patterns
    # =========================================================================
    antibodies = ['ACPA', 'ANA', 'ANTI_DSDNA', 'ANTI_SM']
    groups = ['RF+ SS', 'RF- SS', 'SS', 'RA']
    
    prevalence_data = np.array([
        [0.553, 0.711, 0.447, 0.553],
        [0.617, 0.830, 0.723, 0.511],
        [0.588, 0.776, 0.600, 0.529],
        [0.571, 0.648, 0.484, 0.571]
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
    plt.savefig('results_figures/Figure4_Antibody_Prevalence.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 5: Euclidean Distance Heatmap
    # =========================================================================
    groups = ['RF+ SS', 'RF- SS', 'SS', 'RA']
    
    # Create Euclidean distance matrix with updated values
    euclidean_matrix = np.array([
        [0.000, 0.310, 0.195, 0.195],  # RF+ SS distances
        [0.310, 0.000, 0.348, 0.348],  # RF- SS distances  
        [0.195, 0.348, 0.000, 0.223],  # SS distances
        [0.195, 0.348, 0.223, 0.000]   # RA distances
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
    plt.savefig('results_figures/Figure5_Euclidean_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 6: Performance Summary Heatmap with Standard Deviation Annotation
    # =========================================================================
    comparisons = ['RF+ SS vs RA', 'RF+ SS vs RF- SS', 'RF- SS vs RA', 'SS vs RA']
    metrics = ['Accuracy', 'Balanced\nAccuracy', 'ROC AUC']
    
    performance_data = np.array([
        [0.391, 0.608, 0.635, 0.566],
        [0.364, 0.602, 0.639, 0.568],
        [0.362, 0.618, 0.639, 0.557]
    ])
    
    # Standard deviations for annotation
    performance_std = np.array([
        [0.091, 0.113, 0.073, 0.085],
        [0.080, 0.115, 0.073, 0.085],
        [0.102, 0.130, 0.093, 0.088]
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
    plt.savefig('results_figures/Figure6_Performance_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 7: Distance Analysis with Confidence Intervals
    # =========================================================================
    comparisons = ['RF+ SS vs RA', 'RF- SS vs RA', 'SS vs RA']
    euclidean_dist = [0.195, 0.348, 0.223]
    ci_lower = [0.068, 0.197, 0.095]
    ci_upper = [0.335, 0.507, 0.362]
    
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
    plt.savefig('results_figures/Figure7_Distance_CI.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 8: Comprehensive Performance with All Error Metrics
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel A: Performance metrics with std error bars
    comparisons = ['RF+ SS vs RA', 'RF+ SS vs RF- SS', 'RF- SS vs RA', 'SS vs RA']
    x_pos = np.arange(len(comparisons))
    width = 0.25
    
    ax1.bar(x_pos - width, accuracy, width, yerr=accuracy_std, capsize=5,
            label='Accuracy', alpha=0.8, color='#2E86AB')
    ax1.bar(x_pos, balanced_accuracy, width, yerr=balanced_std, capsize=5,
            label='Balanced Accuracy', alpha=0.8, color='#A23B72')
    ax1.bar(x_pos + width, roc_auc, width, yerr=roc_std, capsize=5,
            label='ROC AUC', alpha=0.8, color='#F18F01')
    
    ax1.set_ylabel('Performance Score')
    ax1.set_title('A. Model Performance Metrics\n(Error bars: Standard Deviation from 10×5-fold CV)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comparisons, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 0.8)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    # Panel B: Distance metrics with CI error bars
    x_pos_dist = np.arange(len(['RF+ SS vs RA', 'RF- SS vs RA', 'SS vs RA']))
    
    errors_lower = [dist - lower for dist, lower in zip(euclidean_dist, ci_lower)]
    errors_upper = [upper - dist for dist, upper in zip(euclidean_dist, ci_upper)]
    errors_dist = [errors_lower, errors_upper]
    
    bars = ax2.bar(x_pos_dist, euclidean_dist, yerr=errors_dist, capsize=10, 
                   alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
    
    ax2.set_ylabel('Euclidean Distance')
    ax2.set_title('B. Immunological Distance\n(Error bars: 95% Bootstrap Confidence Intervals)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos_dist)
    ax2.set_xticklabels(['RF+ SS vs RA', 'RF- SS vs RA', 'SS vs RA'])
    
    # Add value labels with CI
    for i, (bar, dist, lower, upper) in enumerate(zip(bars, euclidean_dist, ci_lower, ci_upper)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{dist:.3f}\n({lower:.3f}-{upper:.3f})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results_figures/Figure8_Comprehensive_Performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All figures generated successfully with error bars!")
    print("Figures saved in 'results_figures/' directory:")
    print("  Figure1_Patient_Cohorts.png")
    print("  Figure2_Model_Performance.png (with std error bars)") 
    print("  Figure3_Statistical_Analysis.png")
    print("  Figure4_Antibody_Prevalence.png")
    print("  Figure5_Euclidean_Heatmap.png")
    print("  Figure6_Performance_Heatmap.png (with ± std annotation)")
    print("  Figure7_Distance_CI.png (with CI error bars)")
    print("  Figure8_Comprehensive_Performance.png (combined view)")

if __name__ == "__main__":
    create_all_figures()
