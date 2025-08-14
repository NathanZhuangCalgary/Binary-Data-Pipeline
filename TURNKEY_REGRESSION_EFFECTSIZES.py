

import pandas as pd
import os
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from statsmodels.formula.api import mnlogit as logit


folder_path = "/Users/nathanzhuang/Documents/Polygence Reasearch Project/Autoimmune Diseases"
all_dfs = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        if 'Diagnosis' in df.columns:
            all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
combined['Diagnosis'] = combined['Diagnosis'].astype(str)

# -------------------------------
# Create simplified diagnosis variable
# -------------------------------
def classify_diagnosis(row):
    diag = row['Diagnosis'].lower()
    rf = row.get('Rheumatoid factor', 0)
    try:
        rf_val = float(rf)
    except:
        rf_val = 0
    if 'rheumatoid arthritis' in diag:
        return 'RA'
    elif 'sj' in diag and rf_val > 0:
        return 'RF+ SS'
    elif 'sj' in diag:
        return 'RF- SS'
    else:
        return None

combined['Diagnosis_simple'] = combined.apply(classify_diagnosis, axis=1)
combined = combined.dropna(subset=['Diagnosis_simple'])

print("Patient counts by simplified diagnosis:")
print(combined['Diagnosis_simple'].value_counts())

# -------------------------------
# Autoantibody markers
# -------------------------------
ab_markers = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']

# -------------------------------
# Tukey HSD
# -------------------------------
print("\nTukey's HSD for all autoantibodies:")
for marker in ab_markers:
    mask = combined[marker].notna()
    values = combined.loc[mask, marker]
    groups = combined.loc[mask, 'Diagnosis_simple']

    tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)
    print(f"\n--- {marker} ---")
    print(tukey)

# -------------------------------
# Cohen's d (effect size) between RA vs SS subgroups
# -------------------------------
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

print("\nCohen's d for autoantibodies (RA vs SS subgroups):")
for marker in ab_markers:
    ra_vals = combined.loc[combined['Diagnosis_simple'] == 'RA', marker].dropna()
    rfpos_vals = combined.loc[combined['Diagnosis_simple'] == 'RF+ SS', marker].dropna()
    rfneg_vals = combined.loc[combined['Diagnosis_simple'] == 'RF- SS', marker].dropna()

    print(f"{marker} | RA vs RF+ SS: d = {cohen_d(ra_vals, rfpos_vals):.3f}")
    print(f"{marker} | RA vs RF- SS: d = {cohen_d(ra_vals, rfneg_vals):.3f}")

# -------------------------------
# Multinomial logistic regression
# -------------------------------
# Replace '-' with '_' for formula compatibility
combined = combined.rename(columns={
    'Anti-dsDNA': 'Anti_dsDNA',
    'Anti-Sm': 'Anti_Sm'
})

# Only include SS subgroups
df_logit = combined[combined['Diagnosis_simple'].isin(['RF+ SS', 'RF- SS'])].copy()
df_logit['Diagnosis_simple_cat'] = df_logit['Diagnosis_simple'].astype('category')

formula = 'Diagnosis_simple_cat ~ ACPA + ANA + Anti_dsDNA + Anti_Sm'

try:
    model = logit(formula=formula, data=df_logit).fit(disp=False)
    print("\nMultinomial Logistic Regression Results (SS subgroups only):")
    print(model.summary())
except Exception as e:
    print("\nError running logistic regression:", e)
    print("Check that column names have no spaces or special characters and that there are enough non-missing values.")

