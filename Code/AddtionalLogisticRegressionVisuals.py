import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style for better visualization
sns.set(style="whitegrid")

# Define the data from the output
antibody_features = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']

# Analysis 1: RF+ SS vs RA
analysis1 = {
    'name': 'RF+ SS vs RA',
    'accuracy': 0.4359,
    'confusion_matrix': np.array([[7, 4], [18, 10]]),
    'labels': ['RF+ SS', 'RA'],
    'feature_importance': pd.DataFrame({
        'feature': ['ANA', 'Anti_Sm', 'Anti_dsDNA', 'ACPA'],
        'coefficient': [-0.189720, 0.066271, 0.057721, -0.046294]
    }),
    'prevalence': pd.DataFrame({
        'Feature': antibody_features * 2,
        'Prevalence': [0.553, 0.711, 0.447, 0.553, 0.571, 0.648, 0.484, 0.571],
        'Group': ['RF+ SS'] * 4 + ['RA'] * 4
    })
}

# Analysis 2: RF+ SS vs RF- SS
analysis2 = {
    'name': 'RF+ SS vs RF- SS',
    'accuracy': 0.6154,
    'confusion_matrix': np.array([[8, 4], [6, 8]]),
    'labels': ['RF+ SS', 'RF- SS'],
    'feature_importance': pd.DataFrame({
        'feature': ['Anti_dsDNA', 'Anti_Sm', 'ACPA', 'ANA'],
        'coefficient': [0.607747, -0.169004, 0.097305, 0.000123]
    }),
    'prevalence': pd.DataFrame({
        'Feature': antibody_features * 2,
        'Prevalence': [0.553, 0.711, 0.447, 0.553, 1.0, 1.0, 1.0, 0.8],  # RF- SS prevalence estimated from sample data
        'Group': ['RF+ SS'] * 4 + ['RF- SS'] * 4
    })
}

# Analysis 3: RF- SS vs RA
analysis3 = {
    'name': 'RF- SS vs RA',
    'accuracy': 0.5238,
    'confusion_matrix': np.array([[4, 10], [10, 18]]),
    'labels': ['RF- SS', 'RA'],
    'feature_importance': pd.DataFrame({
        'feature': ['ANA', 'ACPA', 'Anti_dsDNA', 'Anti_Sm'],
        'coefficient': [-0.437004, -0.405600, -0.293953, 0.046591]
    }),
    'prevalence': pd.DataFrame({
        'Feature': antibody_features * 2,
        'Prevalence': [1.0, 1.0, 1.0, 0.8, 0.571, 0.648, 0.484, 0.571],  # RF- SS prevalence estimated from sample data
        'Group': ['RF- SS'] * 4 + ['RA'] * 4
    })
}

# Analysis 4: SS vs RA
analysis4 = {
    'name': 'SS vs RA',
    'accuracy': 0.5472,
    'confusion_matrix': np.array([[15, 11], [13, 14]]),
    'labels': ['SS', 'RA'],
    'feature_importance': pd.DataFrame({
        'feature': ['ANA', 'Anti_dsDNA', 'ACPA', 'Anti_Sm'],
        'coefficient': [-0.283427, -0.238328, -0.192613, 0.150471]
    }),
    'prevalence': pd.DataFrame({
        'Feature': antibody_features * 2,
        'Prevalence': [0.753, 0.859, 0.718, 0.682, 0.571, 0.648, 0.484, 0.571],  # SS prevalence computed as weighted average of RF+ and RF- SS
        'Group': ['SS'] * 4 + ['RA'] * 4
    })
}

# List of all analyses
analyses = [analysis1, analysis2, analysis3, analysis4]

# Accuracy comparison data
accuracy_data = pd.DataFrame({
    'Analysis': [a['name'] for a in analyses],
    'Accuracy': [a['accuracy'] for a in analyses]
})

# 1. Feature Importance Bar Plots
for analysis in analyses:
    plt.figure(figsize=(8, 6))
    sns.barplot(x='coefficient', y='feature', data=analysis['feature_importance'])
    plt.title(f'Feature Importance: {analysis["name"]}')
    plt.xlabel('Logistic Regression Coefficient')
    plt.ylabel('Antibody')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{analysis["name"].replace(" ", "_").replace("+", "plus").replace("-", "minus")}.png')
    plt.close()

# 2. Confusion Matrix Heatmaps
for analysis in analyses:
    plt.figure(figsize=(6, 5))
    sns.heatmap(analysis['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=analysis['labels'], yticklabels=analysis['labels'])
    plt.title(f'Confusion Matrix: {analysis["name"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{analysis["name"].replace(" ", "_").replace("+", "plus").replace("-", "minus")}.png')
    plt.close()

# 3. Accuracy Comparison Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Analysis', data=accuracy_data)
plt.title('Accuracy Comparison Across Analyses')
plt.xlabel('Accuracy')
plt.xlim(0, 1)
for i, acc in enumerate(accuracy_data['Accuracy']):
    plt.text(acc + 0.01, i, f'{acc:.4f}', va='center')
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.close()

# 4. Antibody Prevalence Bar Plots
for analysis in analyses:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y='Prevalence', hue='Group', data=analysis['prevalence'])
    plt.title(f'Antibody Prevalence: {analysis["name"]}')
    plt.ylabel('Proportion Positive')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'prevalence_{analysis["name"].replace(" ", "_").replace("+", "plus").replace("-", "minus")}.png')
    plt.close()

print("Visualizations generated successfully!")
print("Saved as: feature_importance_*.png, confusion_matrix_*.png, accuracy_comparison.png, prevalence_*.png")