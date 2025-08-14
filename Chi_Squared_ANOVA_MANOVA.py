# Import modules
import pandas as pd
import os
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.multivariate.manova import MANOVA


# Load and combine CSV files

folder_path = "<folder path>"

all_dfs = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        if 'Diagnosis' in df.columns:
            all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
combined['Diagnosis'] = combined['Diagnosis'].astype(str)

# Simplify diagnosis groups
combined['Diagnosis_simple'] = combined['Diagnosis'].str.lower().str.strip()
combined['Diagnosis_simple'] = combined['Diagnosis_simple'].apply(
    lambda x: 'RA' if 'rheumatoid arthritis' in x else ('SS' if 'sj' in x else None)
)
combined = combined.dropna(subset=['Diagnosis_simple'])

# Autoantibody markers
autoantibodies = ['ACPA', 'ANA', 'Anti-dsDNA', 'Anti-Sm']


# Chi-squared test for independence

print("Chi-squared Test Results:")
for ab in autoantibodies:
    contingency = pd.crosstab(combined['Diagnosis_simple'], combined[ab])
    if contingency.shape == (2, 2) or contingency.shape == (3, 2):  # groups x present/absent
        chi2, p, dof, ex = chi2_contingency(contingency)
        signif = "Yes" if p < 0.05 else "No"
        print(f"{ab:<10} | chi2 = {chi2:.3f} | p = {p:.4f} | Significant: {signif}")
    else:
        print(f"{ab}: Contingency table shape not valid")


# One-way ANOVA

groups = {
    "RF+ SS": combined[(combined['Diagnosis_simple'] == 'SS') & (combined['Rheumatoid factor'] > 0)],
    "RF- SS": combined[(combined['Diagnosis_simple'] == 'SS') & (combined['Rheumatoid factor'] == 0)],
    "RA": combined[combined['Diagnosis_simple'] == 'RA']
}

print("\nOne-way ANOVA Results:")
for ab in autoantibodies:
    data_groups = [pd.to_numeric(groups[g][ab], errors='coerce').dropna() for g in groups]
    if all(len(d) > 1 for d in data_groups):
        f_stat, p_val = f_oneway(*data_groups)
        signif = "Yes" if p_val < 0.05 else "No"
        print(f"{ab:<10} | F = {f_stat:.3f} | p = {p_val:.4f} | Significant: {signif}")
    else:
        print(f"{ab}: Not enough data for ANOVA")


# MANOVA

manova_df = combined[['Diagnosis_simple'] + autoantibodies].copy()
manova_df[autoantibodies] = manova_df[autoantibodies].apply(pd.to_numeric, errors='coerce')
manova_df = manova_df.dropna()

maov = MANOVA.from_formula('ACPA + ANA + Q("Anti-dsDNA") + Q("Anti-Sm") ~ Diagnosis_simple', data=manova_df)
print("\nMANOVA Results:")
print(maov.mv_test())
