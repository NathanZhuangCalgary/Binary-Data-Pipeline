import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard, euclidean

# Set plot style for better visualization
sns.set(style="whitegrid")

# Define antibody features
antibody_features = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']

# Manually calculated prevalence data and positive counts
prevalence_data = {
    'RF+ SS': {
        'prevalence': [0.553, 0.711, 0.447, 0.553],
        'counts': [21, 27, 17, 21],
        'total': 38
    },
    'RF- SS': {
        'prevalence': [0.6166, 0.8294, 0.7236, 0.5105],
        'counts': [29, 39, 34, 24],
        'total': 47
    },
    'SS': {
        'prevalence': [0.588, 0.776, 0.600, 0.529],
        'counts': [50, 66, 51, 45],
        'total': 85
    },
    'RA': {
        'prevalence': [0.571, 0.648, 0.484, 0.571],
        'counts': [52, 59, 44, 52],
        'total': 91
    }
}

# Print prevalence and counts for cross-checking
print("\nAntibody Prevalence and Positive Counts:")
for group in prevalence_data:
    print(f"\n{group}:")
    df = pd.DataFrame({
        'Antibody': antibody_features,
        'Prevalence': prevalence_data[group]['prevalence'],
        'Positive Counts': prevalence_data[group]['counts'],
        'Total Patients': [prevalence_data[group]['total']] * len(antibody_features)
    })
    print(df)

# Comparisons to perform
comparisons = [
    ('RF+ SS', 'RA'),
    ('RF- SS', 'RA'),
    ('SS', 'RA')
]

# Calculate Euclidean and Jaccard distances
results = []
for group1, group2 in comparisons:
    # Get prevalence vectors
    vec1 = np.array(prevalence_data[group1]['prevalence'])
    vec2 = np.array(prevalence_data[group2]['prevalence'])
    
    # Binary vectors for Jaccard (threshold > 0.5)
    binary_vec1 = vec1 > 0.5
    binary_vec2 = vec2 > 0.5
    
    # Calculate distances
    euclidean_dist = euclidean(vec1, vec2)
    jaccard_dist = jaccard(binary_vec1, binary_vec2)
    
    # Weight by harmonic mean of patient counts
    n1 = prevalence_data[group1]['total']
    n2 = prevalence_data[group2]['total']
    weight = 2 * (n1 * n2) / (n1 + n2)
    weighted_euclidean = euclidean_dist * weight
    weighted_jaccard = jaccard_dist * weight
    
    results.append({
        'Comparison': f'{group1} vs {group2}',
        'Euclidean Distance': euclidean_dist,
        'Weighted Euclidean Distance': weighted_euclidean,
        'Jaccard Distance': jaccard_dist,
        'Weighted Jaccard Distance': weighted_jaccard,
        'Group1 Count': n1,
        'Group2 Count': n2
    })

# Create DataFrame for results
results_df = pd.DataFrame(results)

# Print results
print("\nDistance Results:")
print(results_df[['Comparison', 'Euclidean Distance', 'Weighted Euclidean Distance', 
                 'Jaccard Distance', 'Weighted Jaccard Distance']])

# Save results to CSV
results_df.to_csv('distance_results.csv', index=False)
print("\nDistance results saved to 'distance_results.csv'")

# Visualization 1: Bar Plot of Euclidean Distances
plt.figure(figsize=(8, 6))
sns.barplot(x='Euclidean Distance', y='Comparison', data=results_df)
plt.title('Euclidean Distance Between Groups')
plt.xlabel('Euclidean Distance')
plt.xlim(0, max(results_df['Euclidean Distance']) + 0.1)
for i, dist in enumerate(results_df['Euclidean Distance']):
    plt.text(dist + 0.01, i, f'{dist:.3f}', va='center')
plt.tight_layout()
plt.savefig('euclidean_distance_bar.png')
plt.close()

# Visualization 2: Bar Plot of Jaccard Distances
plt.figure(figsize=(8, 6))
sns.barplot(x='Jaccard Distance', y='Comparison', data=results_df)
plt.title('Jaccard Distance Between Groups')
plt.xlabel('Jaccard Distance')
plt.xlim(0, 1)
for i, dist in enumerate(results_df['Jaccard Distance']):
    plt.text(dist + 0.01, i, f'{dist:.3f}', va='center')
plt.tight_layout()
plt.savefig('jaccard_distance_bar.png')
plt.close()

# Visualization 3: Heatmap of Euclidean Distances
groups = ['RF+ SS', 'RF- SS', 'SS', 'RA']
euclidean_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)
for group1 in groups:
    for group2 in groups:
        if group1 == group2:
            euclidean_matrix.loc[group1, group2] = 0.0
        else:
            vec1 = np.array(prevalence_data[group1]['prevalence'])
            vec2 = np.array(prevalence_data[group2]['prevalence'])
            euclidean_matrix.loc[group1, group2] = euclidean(vec1, vec2)

plt.figure(figsize=(8, 6))
sns.heatmap(euclidean_matrix.astype(float), annot=True, fmt='.3f', cmap='Blues')
plt.title('Euclidean Distance Heatmap Between Groups')
plt.tight_layout()
plt.savefig('euclidean_distance_heatmap.png')
plt.close()

# Visualization 4: Antibody Prevalence Bar Plot
prevalence_plot_df = pd.DataFrame({
    'Feature': antibody_features * len(prevalence_data),
    'Prevalence': sum([prevalence_data[group]['prevalence'] for group in prevalence_data], []),
    'Group': sum([[group] * len(antibody_features) for group in prevalence_data], [])
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Prevalence', hue='Group', data=prevalence_plot_df)
plt.title('Antibody Prevalence Across Groups')
plt.ylabel('Proportion Positive')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('antibody_prevalence.png')
plt.close()

print("\nVisualizations generated successfully!")
print("Saved as: euclidean_distance_bar.png, jaccard_distance_bar.png, euclidean_distance_heatmap.png, antibody_prevalence.png")