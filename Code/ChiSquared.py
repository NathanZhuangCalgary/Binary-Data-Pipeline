# Import modules
import pandas as pd
import os
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set plot style for better visualization
sns.set(style="whitegrid")

# Load and combine CSV files, selecting only the first 32 columns (up to ACPA)
folder_path = r"C:\Users\natha\Documents\Polygence Reasearch Project\Autoimmune Diseases"

all_dfs = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        path = os.path.join(folder_path, file)
        # Get column names up to 'ACPA'
        all_columns = pd.read_csv(path, nrows=0).columns
        selected_columns = all_columns[:all_columns.get_loc('ACPA') + 1].tolist()
        df = pd.read_csv(path, usecols=selected_columns)
        if 'Diagnosis' in df.columns:
            all_dfs.append(df)

if not all_dfs:
    print("Error: No valid CSV files with 'Diagnosis' column found")
    exit()

combined = pd.concat(all_dfs, ignore_index=True)
combined['Diagnosis'] = combined['Diagnosis'].astype(str)

# Clean column names
combined.columns = [col.replace('-', '_').replace(' ', '_') for col in combined.columns]

# Check for duplicate columns
duplicate_columns = [col for col in combined.columns if combined.columns.tolist().count(col) > 1]
if duplicate_columns:
    print(f"WARNING: Duplicate columns found: {duplicate_columns}")
    for col in duplicate_columns:
        print(f"Dtypes for {col}:")
        print(combined[[c for c in combined.columns if c == col]].dtypes)

# Verify columns and their data types
print("\nAll columns and their data types:")
print(combined.dtypes)

# Simplify diagnosis groups
combined['Diagnosis_simple'] = combined['Diagnosis'].str.lower().str.strip()
combined['Diagnosis_simple'] = combined['Diagnosis_simple'].apply(
    lambda x: 'RA' if 'rheumatoid arthritis' in x else ('SS' if 'sj' in x else None)
)
combined = combined.dropna(subset=['Diagnosis_simple'])

# Autoantibody markers
autoantibodies = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']

# Chi-squared test for independence
chi2_results = []
print("\nChi-squared Test Results:")
for ab in autoantibodies:
    if ab in combined.columns:
        contingency = pd.crosstab(combined['Diagnosis_simple'], combined[ab])
        print(f"\nContingency table for {ab}:")
        print(contingency)
        if contingency.shape == (2, 2) or contingency.shape == (3, 2):  # groups x present/absent
            chi2, p, dof, ex = chi2_contingency(contingency)
            signif = "Yes" if p < 0.05 else "No"
            print(f"{ab:<10} | chi2 = {chi2:.3f} | p = {p:.4f} | Significant: {signif}")
            chi2_results.append({'Antibody': ab, 'Chi2': chi2, 'P-value': p, 'Significant': signif})
        else:
            print(f"{ab}: Contingency table shape not valid: {contingency.shape}")
    else:
        print(f"{ab}: Column not found in dataset")

# Visualization: -log(p-value) bar plot
chi2_df = pd.DataFrame(chi2_results)
if not chi2_df.empty:
    chi2_df['-log10(P-value)'] = -np.log10(chi2_df['P-value'].clip(lower=1e-10))  # Avoid log(0)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='-log10(P-value)', y='Antibody', hue='Significant', data=chi2_df)
    plt.title('Chi-Squared Test: Significance of Antibodies (SS vs RA)')
    plt.xlabel('-log10(P-value)')
    plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.legend()
    plt.tight_layout()
    plt.savefig('chi2_significance.png')
    plt.close()

    # Save results to CSV for pipeline
    chi2_df.to_csv('chi2_results.csv', index=False)
    print("\nChi-squared results saved to 'chi2_results.csv'")
else:
    print("\nNo valid chi-squared results to visualize")

print("\nChi-squared analysis complete!")