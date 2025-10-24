"""
Binary Antibody Diagnostic Pipeline — Enhanced Version with Complete Outputs
---------------------------------------------------
Purpose:
    Analyze binary-coded antibody data to evaluate serological patterns 
    in SS stratified by RF status.

Key Features:
    - Logistic regression separability with repeated cross-validation
    - Chi-squared & Fisher's exact tests with multiple testing correction
    - Euclidean & Jaccard distance analysis with bootstrap CIs
    - Comprehensive metrics (balanced accuracy, AUC, sensitivity, specificity)
    - Automatic figure and table generation
    - Complete data capture for publication tables

Author: Nathan Zhuang
Version: CSBJ Submission Final Draft - Complete Outputs
Python ≥3.9 required
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, balanced_accuracy_score, roc_curve
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency, fisher_exact
from scipy.spatial.distance import jaccard, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import os
import unicodedata
import re
import hashlib
import pickle
import sys
from statsmodels.stats.multitest import multipletests

sns.set(style="whitegrid")

# -----------------------------
# Utility: Clean column names
# -----------------------------
def normalize_colname(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return s.lower().replace('-', '_').replace(' ', '_').strip()

# -----------------------------
# Load data safely
# -----------------------------
def load_data():
    file_path = r"C:\Users\natha\Desktop\ClinicalDatasetResearch\Code\cleaned_antibody_data.csv"
    
    # Read the file with UTF-8 encoding
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Original columns: {df.columns.tolist()}")
    
    # Normalize column names
    df.columns = [normalize_colname(c) for c in df.columns]
    print(f"Normalized columns: {df.columns.tolist()}")
    
    # Create mapping based on your column order
    rename_map = {
        'hla_b27': 'hla_b27',
        'ana': 'ana',
        'anti_ro': 'anti_ro',
        'anti_la': 'anti_la',
        'anti_dsdna': 'anti_dsdna',
        'anti_sm': 'anti_sm',
        'c3': 'c3',
        'c4': 'c4',
        'disease': 'diagnosis',
        'esr_abn': 'esr_abn',
        'crp_abn': 'crp_abn',
        'rf_abn': 'rf_abn',
        'anticcp_abn': 'anticcp_abn',
        'c3_abn': 'c3_abn',
        'c4_abn': 'c4_abn'
    }
    
    # Apply renaming only for columns that exist
    final_rename = {}
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            final_rename[old_name] = new_name
    
    df.rename(columns=final_rename, inplace=True)
    print(f"After renaming: {df.columns.tolist()}")
    
    # Fill all columns with 0 for missing values
    all_markers = list(rename_map.values())
    for col in all_markers:
        if col in df.columns and col != 'diagnosis':
            df[col] = df[col].fillna(0).astype(int)
            print(f"Column '{col}': {df[col].sum()} positive cases out of {len(df)}")
    
    print(f"Final data shape: {df.shape}")
    return df

# -----------------------------
# Standardized diagnosis matching
# -----------------------------
def standardize_diagnosis(df):
    """Standardize diagnosis column into RA or SS groups."""
    df = df.copy()

    def get_diagnosis_group(diagnosis_str):
        if pd.isna(diagnosis_str):
            return None
        s = str(diagnosis_str).lower()
        # handle abbreviations and common variants
        if re.search(r'\bra\b', s) or ('rheumatoid' in s and 'arthritis' in s):
            return 'ra'
        elif any(x in s for x in ['sjogren', 'sjögren', 'sjogrens', 'sjögrens']):
            return 'ss'
        else:
            return None

    df['diagnosis_group'] = df['diagnosis'].apply(get_diagnosis_group)
    return df

# -----------------------------
# Patient counts
# -----------------------------
def patient_counts(df):
    df_clean = standardize_diagnosis(df)
    ra = df_clean['diagnosis_group'] == 'ra'
    ss = df_clean['diagnosis_group'] == 'ss'
    rf = df_clean['rf_abn'] == 1  # RF positive
    
    ra_count, ss_count = ra.sum(), ss.sum()
    rf_pos_ss, rf_neg_ss = (ss & rf).sum(), (ss & ~rf).sum()
    
    print(f"Patient Counts")
    print(f"RA: {ra_count}")
    print(f"SS: {ss_count}") 
    print(f"RF+ SS: {rf_pos_ss}")
    print(f"RF- SS: {rf_neg_ss}")
    print(f"Total with valid diagnosis: {len(df_clean.dropna(subset=['diagnosis_group']))}")
    
    return ra_count, ss_count, rf_pos_ss, rf_neg_ss

# -----------------------------
# Generate 2x2 contingency tables with raw counts
# -----------------------------
def generate_contingency_tables(df):
    """Generate raw 2x2 contingency tables for each antibody"""
    df_clean = standardize_diagnosis(df)
    df_clean = df_clean.dropna(subset=['diagnosis_group'])
    
    antibodies = ['hla_b27', 'anti_dsdna', 'anti_sm']
    contingency_tables = {}
    
    print("\n=== 2x2 Contingency Tables (Raw Counts) ===")
    
    for ab in antibodies:
        # Create contingency table with raw counts
        cont_table = pd.crosstab(df_clean['diagnosis_group'], df_clean[ab])
        
        # Ensure proper ordering
        if set(['ra', 'ss']).issubset(cont_table.index):
            cont_table = cont_table.loc[['ra', 'ss'], :]
            if set([0, 1]).issubset(cont_table.columns):
                cont_table = cont_table.loc[:, [1, 0]]  # Positive then negative
                
                contingency_tables[ab] = cont_table
                
                print(f"\n{ab.upper()}:")
                print(f"         | Positive | Negative")
                print(f"---------|----------|----------")
                print(f"RA       | {cont_table.loc['ra', 1]:8} | {cont_table.loc['ra', 0]:8}")
                print(f"SS       | {cont_table.loc['ss', 1]:8} | {cont_table.loc['ss', 0]:8}")
    
    return contingency_tables

# -----------------------------
# Logistic regression with cross-validation and ROC data capture
# -----------------------------
def run_logistic_analysis(df):
    antibodies = ['hla_b27', 'anti_dsdna', 'anti_sm']
    df_clean = standardize_diagnosis(df)
    
    # Get groups - stratify SS by RF status
    ra = df_clean[df_clean['diagnosis_group'] == 'ra'].copy()
    ss = df_clean[df_clean['diagnosis_group'] == 'ss'].copy()
    rf_pos_ss = ss[ss['rf_abn'] == 1].copy()  # RF positive SS
    rf_neg_ss = ss[ss['rf_abn'] == 0].copy()  # RF negative SS
    
    comparisons = [
        (rf_pos_ss, ra, 'rf+_ss', 'ra', 'RF+ SS vs RA'),
        (rf_pos_ss, rf_neg_ss, 'rf+_ss', 'rf-_ss', 'RF+ SS vs RF- SS'),
        (rf_neg_ss, ra, 'rf-_ss', 'ra', 'RF- SS vs RA'),
        (ss, ra, 'ss', 'ra', 'SS vs RA')
    ]
    
    analyses = []
    roc_data = {}  # Store ROC curve data
    
    for g1, g2, l1, l2, name in comparisons:
        if len(g1) == 0 or len(g2) == 0:
            print(f"\n{name}: skipped (no samples)")
            continue
            
        df_sub = pd.concat([g1, g2], ignore_index=True)
        df_sub['target'] = [0]*len(g1) + [1]*len(g2)
        X, y = df_sub[antibodies], df_sub['target']
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"{name}: Class distribution - {np.unique(y, return_counts=True)}")
        print(f"{name}: Class weights - {class_weight_dict}")
        
        # Create pipeline to prevent data leakage with class weights
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, class_weight=class_weight_dict, random_state=42))
        ])
        
        # Cross-validation with comprehensive metrics
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
        
        # Calculate multiple metrics
        scoring_metrics = {
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = {}
        for metric_name, scoring in scoring_metrics.items():
            scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            cv_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'all_scores': scores  # Store all CV scores
            }
        
        print(f"\n{name}:")
        print(f"  CV Accuracy = {cv_results['accuracy']['mean']:.3f} ± {cv_results['accuracy']['std']:.3f}")
        print(f"  Balanced Accuracy = {cv_results['balanced_accuracy']['mean']:.3f} ± {cv_results['balanced_accuracy']['std']:.3f}")
        print(f"  ROC AUC = {cv_results['roc_auc']['mean']:.3f} ± {cv_results['roc_auc']['std']:.3f}")
        
        # Fit final model on all data for feature importance and ROC
        pipeline.fit(X, y)
        
        # Get cross-validation predictions for ROC curves
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        cv_roc = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)  # Smaller for ROC
        for train_idx, test_idx in cv_roc.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Calculate class weights for this fold
            fold_classes = np.unique(y_train)
            fold_weights = compute_class_weight('balanced', classes=fold_classes, y=y_train)
            fold_weight_dict = dict(zip(fold_classes, fold_weights))
            
            # Clone pipeline to avoid data leakage
            pipe_clone = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, class_weight=fold_weight_dict, random_state=42))
            ])
            pipe_clone.fit(X_train, y_train)
            y_proba = pipe_clone.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc_score(y_test, y_proba))
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        
        # Store ROC data
        roc_data[name] = {
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'aucs': aucs
        }
        
        # Generate confusion matrix from final model
        y_pred = pipeline.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': antibodies, 
            'coefficient': pipeline.named_steps['model'].coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        analyses.append({
            'name': name,
            'cv_results': cv_results,
            'labels': [l1, l2],
            'feature_importance': feature_importance,
            'prevalence': pd.DataFrame({
                'feature': antibodies * 2,
                'prevalence': [g1[a].mean() for a in antibodies] + [g2[a].mean() for a in antibodies],
                'group': [l1]*len(antibodies) + [l2]*len(antibodies)
            }),
            'confusion_matrix': cm,
            'X': X,
            'y': y
        })
    
    return analyses, roc_data

# -----------------------------
# Permutation testing for classifier significance
# -----------------------------
def run_permutation_tests(analyses, n_permutations=100):
    """Run permutation tests for each classifier comparison"""
    print("Permutation Tests")
    permutation_results = {}
    
    for analysis in analyses:
        name = analysis['name']
        X = analysis['X']
        y = analysis['y']
        
        # Calculate class weights for permutation test too
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, class_weight=class_weight_dict, random_state=42))
        ])
        
        # Use smaller CV for permutation test (faster)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        
        # Run permutation test
        try:
            score, permutation_scores, pvalue = permutation_test_score(
                pipeline, X, y, scoring='roc_auc', cv=cv,
                n_permutations=n_permutations, random_state=42, n_jobs=-1
            )
            
            permutation_results[name] = {
                'true_score': score,
                'permutation_scores': permutation_scores,
                'p_value': pvalue
            }
            
            print(f"{name}:")
            print(f"  True ROC AUC: {score:.3f}")
            print(f"  Permutation p-value: {pvalue:.4f}")
            print(f"  Mean permutation score: {np.mean(permutation_scores):.3f}")
        except Exception as e:
            print(f"Permutation test failed for {name}: {e}")
            permutation_results[name] = {
                'true_score': np.nan,
                'permutation_scores': [],
                'p_value': np.nan
            }
    
    return permutation_results

# -----------------------------
# Statistical tests with Fisher's exact and multiple testing correction
# -----------------------------
def statistical_analysis(df):
    """Perform 2x2 contingency analyses with safety checks and FDR correction."""
    df_clean = standardize_diagnosis(df)
    df_clean = df_clean.dropna(subset=['diagnosis_group'])

    ab_cols = ['hla_b27', 'anti_dsdna', 'anti_sm']
    results = []

    for ab in ab_cols:
        try:
            cont_table = pd.crosstab(df_clean['diagnosis_group'], df_clean[ab])
            # ensure 2x2 shape and fixed ordering
            if not set(['ra', 'ss']).issubset(cont_table.index):
                continue
            cont_table = cont_table.loc[['ss', 'ra'], [0, 1]]
            table_vals = cont_table.values

            # Chi²
            try:
                chi2, p_chi2, dof, expected = chi2_contingency(table_vals)
                chi2_valid = expected.min() >= 5
            except Exception:
                chi2, p_chi2, expected, chi2_valid = np.nan, np.nan, np.nan, False

            # Fisher's exact
            try:
                odds_ratio, p_fisher = fisher_exact(table_vals)
            except Exception:
                odds_ratio, p_fisher = np.nan, np.nan

            results.append({
                'antibody': ab,
                'chi2': chi2,
                'p_chi2': p_chi2,
                'p_fisher': p_fisher,
                'odds_ratio': odds_ratio,
                'chi2_valid': chi2_valid,
                'min_expected': expected.min() if isinstance(expected, np.ndarray) else np.nan,
                'contingency_table': table_vals  # Store raw counts
            })
        except Exception as e:
            print(f"Skipped {ab}: {e}")

    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("No valid antibody comparisons found.")
        return results_df

    # Multiple testing correction
    results_df['p_chi2_adj'] = multipletests(results_df['p_chi2'], method='fdr_bh')[1]
    results_df['p_fisher_adj'] = multipletests(results_df['p_fisher'], method='fdr_bh')[1]

    print("Statistical Analysis")
    for _, row in results_df.iterrows():
        print(f"\n{row['antibody'].upper()}:")
        print(f"  Chi² = {row['chi2']:.3f}, p = {row['p_chi2']:.4f} (adj: {row['p_chi2_adj']:.4f})")
        print(f"  Fisher's OR = {row['odds_ratio']:.3f}, p = {row['p_fisher']:.4f} (adj: {row['p_fisher_adj']:.4f})")
        if not row['chi2_valid']:
            print("Chi-square may be invalid (expected count < 5)")

    return results_df

# -----------------------------
# Distance analysis with bootstrap confidence intervals and sample storage
# -----------------------------
def distance_analysis(df):
    """
    Compute Euclidean and Jaccard distances with proper bootstrap confidence intervals.
    Returns bootstrap samples for distribution plotting.
    """
    antibodies = ['hla_b27', 'anti_dsdna', 'anti_sm']
    df_clean = standardize_diagnosis(df)

    ra = df_clean[df_clean['diagnosis_group'] == 'ra'].copy()
    ss = df_clean[df_clean['diagnosis_group'] == 'ss'].copy()
    rf_pos_ss = ss[ss['rf_abn'] == 1].copy()  # RF positive SS
    rf_neg_ss = ss[ss['rf_abn'] == 0].copy()  # RF negative SS

    groups = {'RF+ SS': rf_pos_ss, 'RF- SS': rf_neg_ss, 'SS': ss, 'RA': ra}
    groups = {k: v for k, v in groups.items() if len(v) > 0}

    # Compute prevalence per antibody
    prevalence_data = {}
    for name, group_df in groups.items():
        prevalence_data[name] = {
            'prevalence': [group_df[a].mean() for a in antibodies],
            'binary_vectors': group_df[antibodies].values,
            'n': len(group_df)
        }

    print("\nAntibody Prevalence:")
    for g, data in prevalence_data.items():
        print(f"{g}: {[f'{p:.3f}' for p in data['prevalence']]} (n={data['n']})")

    def bootstrap_centroid_distance(v1, v2, n_bootstrap=1000, ci=95, seed=42):
        rng = np.random.default_rng(seed)
        n1, n2 = v1.shape[0], v2.shape[0]
        boot_stats = []
        for _ in range(n_bootstrap):
            s1 = v1[rng.integers(0, n1, n1)].mean(axis=0)
            s2 = v2[rng.integers(0, n2, n2)].mean(axis=0)
            boot_stats.append(euclidean(s1, s2))
        lower = np.percentile(boot_stats, (100 - ci) / 2)
        upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
        return np.mean(boot_stats), lower, upper, boot_stats  # Return bootstrap samples

    results = []
    bootstrap_samples = {}  # Store all bootstrap samples
    comparisons = [('RF+ SS', 'RA'), ('RF- SS', 'RA'), ('SS', 'RA')]

    for g1, g2 in comparisons:
        if g1 not in prevalence_data or g2 not in prevalence_data:
            continue

        v1 = prevalence_data[g1]['binary_vectors']
        v2 = prevalence_data[g2]['binary_vectors']
        mean_euc, lower, upper, samples = bootstrap_centroid_distance(v1, v2)

        # Store bootstrap samples
        bootstrap_samples[f"{g1} vs {g2}"] = samples

        # Jaccard based on group-level prevalence > 0
        jaccard_dist = jaccard(np.array(prevalence_data[g1]['prevalence']) > 0,
                               np.array(prevalence_data[g2]['prevalence']) > 0)

        results.append({
            'comparison': f'{g1} vs {g2}',
            'euclidean_distance': mean_euc,
            'euclidean_ci_lower': lower,
            'euclidean_ci_upper': upper,
            'jaccard_distance': jaccard_dist,
            'n1': prevalence_data[g1]['n'],
            'n2': prevalence_data[g2]['n']
        })

    df_results = pd.DataFrame(results)

    print("Distance Analysis (Bootstrap 95% CI)")
    for _, row in df_results.iterrows():
        print(f"{row['comparison']}:")
        print(f"  Euclidean = {row['euclidean_distance']:.3f} (95% CI: {row['euclidean_ci_lower']:.3f}-{row['euclidean_ci_upper']:.3f})")
        print(f"  Jaccard = {row['jaccard_distance']:.3f}")

    return df_results, prevalence_data, bootstrap_samples

# -----------------------------
# Generate code hash/version info (FIXED encoding issue)
# -----------------------------
def generate_code_hash():
    """Generate hash of the current code for reproducibility"""
    current_file = __file__
    try:
        # Try reading with UTF-8 first
        with open(current_file, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except UnicodeDecodeError:
        try:
            # Fallback to latin-1 which should handle any byte
            with open(current_file, 'r', encoding='latin-1') as f:
                code_content = f.read()
        except Exception as e:
            print(f"Could not read file for hashing: {e}")
            code_content = "Unable to read file content"
    
    code_hash = hashlib.sha256(code_content.encode('utf-8')).hexdigest()[:16]
    
    version_info = {
        'code_hash': code_hash,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'packages': {
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'sklearn': 'unknown',  # Will be filled in main
            'scipy': 'unknown'     # Will be filled in main
        }
    }
    
    print("Code Version Information")
    print(f"Code Hash: {code_hash}")
    print(f"Timestamp: {version_info['timestamp']}")
    print(f"Python: {version_info['python_version']}")
    
    return version_info

# -----------------------------
# Enhanced Visualization with all new plots
# -----------------------------
def create_visualizations(analyses, stats_results, distance_results, prevalence_data, 
                         roc_data, permutation_results, bootstrap_samples, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance metrics with error bars
    metrics_data = []
    for a in analyses:
        for metric in ['accuracy', 'balanced_accuracy', 'roc_auc']:
            metrics_data.append({
                'analysis': a['name'],
                'metric': metric.replace('_', ' ').title(),
                'value': a['cv_results'][metric]['mean'],
                'error': a['cv_results'][metric]['std']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='value', y='analysis', hue='metric', data=metrics_df)
    plt.title('Cross-Validated Performance Metrics by Comparison\n(Error bars: Standard Deviation)')
    plt.xlim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves
    if roc_data:
        plt.figure(figsize=(10, 8))
        for name, data in roc_data.items():
            mean_auc = np.mean(data['aucs'])
            std_auc = np.std(data['aucs'])
            
            plt.plot(data['mean_fpr'], data['mean_tpr'], 
                    label=f'{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})', linewidth=2)
            
            # Add confidence interval
            tprs_upper = np.minimum(data['mean_tpr'] + data['std_tpr'], 1)
            tprs_lower = np.maximum(data['mean_tpr'] - data['std_tpr'], 0)
            plt.fill_between(data['mean_fpr'], tprs_lower, tprs_upper, alpha=0.2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves with Confidence Intervals')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Bootstrap distributions for Euclidean distances
    if bootstrap_samples:
        fig, axes = plt.subplots(1, len(bootstrap_samples), figsize=(15, 5))
        if len(bootstrap_samples) == 1:
            axes = [axes]
        
        for idx, (comparison, samples) in enumerate(bootstrap_samples.items()):
            axes[idx].hist(samples, bins=30, alpha=0.7, edgecolor='black')
            axes[idx].axvline(np.mean(samples), color='red', linestyle='--', label=f'Mean: {np.mean(samples):.3f}')
            axes[idx].set_xlabel('Euclidean Distance')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Bootstrap Distribution: {comparison}')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bootstrap_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Confusion matrices
    for a in analyses:
        plt.figure(figsize=(6, 5))
        sns.heatmap(a['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {a["name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{a["name"].replace(" ", "_")}.png'), dpi=300)
        plt.close()
    
    # 5. Permutation test results
    if permutation_results and any(not np.isnan(permutation_results[comp]['p_value']) for comp in permutation_results):
        plt.figure(figsize=(10, 6))
        comparisons = []
        p_values = []
        for comp, results in permutation_results.items():
            if not np.isnan(results['p_value']):
                comparisons.append(comp)
                p_values.append(results['p_value'])
        
        if comparisons:
            bars = plt.bar(comparisons, p_values, color=['red' if p < 0.05 else 'blue' for p in p_values])
            plt.axhline(y=0.05, color='black', linestyle='--', label='Significance threshold (0.05)')
            plt.ylabel('Permutation p-value')
            plt.title('Permutation Test Results')
            plt.xticks(rotation=45)
            plt.legend()
            
            # Add value labels
            for bar, p_val in zip(bars, p_values):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{p_val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'permutation_tests.png'), dpi=300, bbox_inches='tight')
            plt.close()

# -----------------------------
# Save all results to files
# -----------------------------
def save_results(analyses, stats_results, distance_results, contingency_tables,
                roc_data, permutation_results, bootstrap_samples, version_info, output_dir="output"):
    """Save all analysis results to files for publication"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save classifier performance table
    performance_data = []
    for a in analyses:
        performance_data.append({
            'Comparison': a['name'],
            'Accuracy': f"{a['cv_results']['accuracy']['mean']:.3f} ± {a['cv_results']['accuracy']['std']:.3f}",
            'Balanced_Accuracy': f"{a['cv_results']['balanced_accuracy']['mean']:.3f} ± {a['cv_results']['balanced_accuracy']['std']:.3f}",
            'ROC_AUC': f"{a['cv_results']['roc_auc']['mean']:.3f} ± {a['cv_results']['roc_auc']['std']:.3f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(os.path.join(output_dir, 'classifier_performance.csv'), index=False)
    
    # 2. Save statistical analysis results
    stats_results.to_csv(os.path.join(output_dir, 'statistical_analysis.csv'), index=False)
    
    # 3. Save distance analysis results
    distance_results.to_csv(os.path.join(output_dir, 'distance_analysis.csv'), index=False)
    
    # 4. Save contingency tables
    if contingency_tables:
        contingency_df = pd.DataFrame()
        for ab, table in contingency_tables.items():
            table_df = pd.DataFrame(table.values, 
                                   index=[f'RA_{ab}', f'SS_{ab}'], 
                                   columns=['Positive', 'Negative'])
            contingency_df = pd.concat([contingency_df, table_df])
        contingency_df.to_csv(os.path.join(output_dir, 'contingency_tables.csv'))
    
    # 5. Save version information
    with open(os.path.join(output_dir, 'version_info.txt'), 'w') as f:
        for key, value in version_info.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # 6. Save all data as pickle for later use
    all_results = {
        'analyses': analyses,
        'stats_results': stats_results,
        'distance_results': distance_results,
        'contingency_tables': contingency_tables,
        'roc_data': roc_data,
        'permutation_results': permutation_results,
        'bootstrap_samples': bootstrap_samples,
        'version_info': version_info
    }
    
    with open(os.path.join(output_dir, 'all_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nAll results saved to {output_dir}/ directory")

# -----------------------------
# Main execution
# -----------------------------
def main():
    print("Pipeline Start")
    
    # Generate code hash first (with proper encoding handling)
    version_info = generate_code_hash()
    
    # Update package versions
    try:
        import sklearn
        version_info['packages']['sklearn'] = sklearn.__version__
    except ImportError:
        pass
    
    try:
        import scipy
        version_info['packages']['scipy'] = scipy.__version__
    except ImportError:
        pass
    
    # Load and analyze data
    df = load_data()
    patient_counts(df)
    
    # Generate contingency tables
    contingency_tables = generate_contingency_tables(df)
    
    # Run analyses
    analyses, roc_data = run_logistic_analysis(df)
    stats_results = statistical_analysis(df)
    dist_results, prev_data, bootstrap_samples = distance_analysis(df)
    
    # Run permutation tests (can be slow)
    print("\nRunning permutation tests (this may take a while)...")
    permutation_results = run_permutation_tests(analyses, n_permutations=100)
    
    # Create visualizations
    create_visualizations(analyses, stats_results, dist_results, prev_data,
                         roc_data, permutation_results, bootstrap_samples)
    
    # Save all results
    save_results(analyses, stats_results, dist_results, contingency_tables,
                roc_data, permutation_results, bootstrap_samples, version_info)
    
    print("Final Summary")
    for a in analyses:
        print(f"\n{a['name']}:")
        print(f"  Accuracy: {a['cv_results']['accuracy']['mean']:.3f} ± {a['cv_results']['accuracy']['std']:.3f}")
        print(f"  Balanced Accuracy: {a['cv_results']['balanced_accuracy']['mean']:.3f} ± {a['cv_results']['balanced_accuracy']['std']:.3f}")
        print(f"  ROC AUC: {a['cv_results']['roc_auc']['mean']:.3f} ± {a['cv_results']['roc_auc']['std']:.3f}")
    
    print("\nPipeline complete. All outputs saved to 'output/' directory.")

if __name__ == "__main__":
    main()