# Import modules
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ---------------------------
# Load data CSV file
# ---------------------------
file_path = r"C:\Users\natha\Documents\Polygence Reasearch Project\Autoimmune Diseases\Autoimmune_Disorder_10k_with_All_Disorders.csv"
combined = pd.read_csv(file_path)

# Ensure 'Diagnosis' column is string
combined['Diagnosis'] = combined['Diagnosis'].astype(str)

# Simplify diagnosis groups
combined['Diagnosis_simple'] = combined['Diagnosis'].str.lower().str.strip()
combined['Diagnosis_simple'] = combined['Diagnosis_simple'].apply(
    lambda x: 'RA' if 'rheumatoid arthritis' in x else ('SS' if 'sj' in x else None)
)
combined = combined.dropna(subset=['Diagnosis_simple'])

# Autoantibody markers
autoantibodies = ['ACPA', 'ANA', 'Anti-dsDNA', 'Anti-Sm']

# ---------------------------
# Chi-squared test for independence
# ---------------------------
print("Chi-squared Test Results:")
for ab in autoantibodies:
    contingency = pd.crosstab(combined['Diagnosis_simple'], combined[ab])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p, dof, ex = chi2_contingency(contingency)
        signif = "Yes" if p < 0.05 else "No"
        print(f"{ab:<10} | chi2 = {chi2:.3f} | p = {p:.4f} | Significant: {signif}")
    else:
        print(f"{ab}: Contingency table shape not valid")

# ---------------------------
# One-way ANOVA
# ---------------------------
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

# ---------------------------
# MANOVA
# ---------------------------
manova_df = combined[['Diagnosis_simple'] + autoantibodies].copy()
manova_df[autoantibodies] = manova_df[autoantibodies].apply(pd.to_numeric, errors='coerce')
manova_df = manova_df.dropna()

maov = MANOVA.from_formula('ACPA + ANA + Q("Anti-dsDNA") + Q("Anti-Sm") ~ Diagnosis_simple', data=manova_df)
print("\nMANOVA Results:")
print(maov.mv_test())

# ---------------------------
# Univariate ANOVA + Multiple Comparison Correction
# ---------------------------
print("\nUnivariate ANOVA Results with Holm/FDR Correction:")

pvals = []
for ab in autoantibodies:
    formula = f'Q("{ab}") ~ Diagnosis_simple' if "-" in ab else f'{ab} ~ Diagnosis_simple'
    model = ols(formula, data=manova_df).fit()
    aov_table = sm.stats.anova_lm(model)
    p_val = aov_table["PR(>F)"]["Diagnosis_simple"]
    pvals.append(p_val)

# Apply Holm and FDR corrections
reject_holm, pvals_holm, _, _ = multipletests(pvals, method='holm')
reject_fdr, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

for i, ab in enumerate(autoantibodies):
    print(f"{ab:<10} | raw p = {pvals[i]:.4f} | Holm p = {pvals_holm[i]:.4f} | "
          f"FDR p = {pvals_fdr[i]:.4f} | Holm sig: {reject_holm[i]} | FDR sig: {reject_fdr[i]}")

# ---------------------------
# Logistic Regression (trying to predict Diagnosis_simple from antibody patterns)
# ---------------------------
X = manova_df[autoantibodies]
y = manova_df['Diagnosis_simple']

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, y)
y_pred = log_reg.predict(X)

print("\nLogistic Regression Results:")
print(classification_report(y, y_pred, digits=3, zero_division=0))





