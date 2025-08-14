# Import modules

import pandas as pd
import os
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind


# Load and combine CSV files

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


# Identify groups

sj_mask = combined['Diagnosis'].str.lower().str.startswith("sj")
ra_mask = combined['Diagnosis'].str.lower() == "rheumatoid arthritis"

sj_df = combined[sj_mask].copy()
ra_df = combined[ra_mask].copy()

# Split Sjögren into RF+ and RF-
sj_df['Rheumatoid factor'] = pd.to_numeric(sj_df['Rheumatoid factor'], errors='coerce')
sj_pos = sj_df[sj_df['Rheumatoid factor'] > 0]
sj_neg = sj_df[sj_df['Rheumatoid factor'] == 0]


# Autoantibody markers

ab_markers = ['ACPA', 'ANA', 'Anti-dsDNA', 'Anti-Sm']


# Euclidean distance calculation

def calc_distance(features):
    """Calculate Euclidean distance between group centroids for selected features."""
    combined_data = pd.concat([sj_pos[features], sj_neg[features], ra_df[features]], ignore_index=True)
    combined_data = combined_data.dropna()

    scaler = StandardScaler()
    scaler.fit(combined_data)

    sj_pos_scaled = scaler.transform(sj_pos[features].dropna())
    sj_neg_scaled = scaler.transform(sj_neg[features].dropna())
    ra_scaled = scaler.transform(ra_df[features].dropna())

    mean_sj_pos = np.nanmean(sj_pos_scaled, axis=0)
    mean_sj_neg = np.nanmean(sj_neg_scaled, axis=0)
    mean_ra = np.nanmean(ra_scaled, axis=0)

    dist_rfpos_ra = euclidean(mean_sj_pos, mean_ra)
    dist_rfneg_ra = euclidean(mean_sj_neg, mean_ra)

    return dist_rfpos_ra, dist_rfneg_ra

dist_rfpos_ra, dist_rfneg_ra = calc_distance(ab_markers)

print("\nEuclidean Distances Between Group Centroids (Z-normalized):")
print(f"  RF+ Sjögren vs RA: {dist_rfpos_ra:.4f}")
print(f"  RF- Sjögren vs RA: {dist_rfneg_ra:.4f}")


# T-tests for autoantibodies

def run_t_tests(group1, group2, features, label1, label2):
    """Run Welch’s t-tests comparing two groups for selected features."""
    print(f"\nT-tests: Comparing {label1} vs {label2}")
    
    for feature in features:
        group1_feature = pd.to_numeric(group1[feature], errors='coerce').dropna()
        group2_feature = pd.to_numeric(group2[feature], errors='coerce').dropna()

        if len(group1_feature) < 2 or len(group2_feature) < 2:
            print(f"  {feature}: Not enough data to run test.")
            continue

        t_stat, p_val = ttest_ind(group1_feature, group2_feature, equal_var=False)
        signif = "Yes" if p_val < 0.05 else "No"
        print(f"  {feature:<10} | t = {t_stat:>6.3f} | p = {p_val:.4f} | Significant: {signif}")

run_t_tests(sj_pos, ra_df, ab_markers, "RF+ SS", "RA")
run_t_tests(sj_neg, ra_df, ab_markers, "RF- SS", "RA")


# Optional: Simple Diagnosis Counts

combined['Diagnosis_simple'] = combined['Diagnosis'].str.lower().str.strip()
combined['Diagnosis_simple'] = combined['Diagnosis_simple'].apply(
    lambda x: 'RHEUMATOID ARTHRITIS' if 'rheumatoid arthritis' in x else ('SJÖGREN' if 'sj' in x else None)
)
combined = combined.dropna(subset=['Diagnosis_simple'])

print("\nPatient counts by simplified diagnosis:")
print(combined['Diagnosis_simple'].value_counts())
print("Rows after filtering:", combined.shape[0])
