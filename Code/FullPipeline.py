import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jaccard, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Set plot style for better visualization
sns.set(style="whitegrid")

# Define output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data():
    file_path = r"C:\Users\natha\Documents\Polygence Reasearch Project\Autoimmune Diseases\Autoimmune_Disorder_10k_with_All_Disorders.csv"
    all_columns = pd.read_csv(file_path, nrows=0).columns
    selected_columns = all_columns[:all_columns.get_loc('ACPA') + 1].tolist()
    df = pd.read_csv(file_path, usecols=selected_columns)
    df.columns = [col.replace('-', '_').replace(' ', '_') for col in df.columns]
    df = df.fillna(0)
    return df

def patient_counts(df):
    ra_count = (df["Diagnosis"] == "RA").sum()
    sj_count = (df["Diagnosis"].str.contains("Sj", case=False, na=False)).sum()
    rf_pos_sj = ((df["Diagnosis"].str.contains("Sj", case=False, na=False)) & (df["Rheumatoid_factor"] == 1)).sum()
    rf_neg_sj = ((df["Diagnosis"].str.contains("Sj", case=False, na=False)) & (df["Rheumatoid_factor"] == 0)).sum()
    
    print("Patient Counts:")
    print(f"RA: {ra_count}")
    print(f"SS (total): {sj_count}")
    print(f"SS RF+: {rf_pos_sj}")
    print(f"SS RF-: {rf_neg_sj}")
    print(f"Total Patients: {len(df)}")
    return ra_count, sj_count, rf_pos_sj, rf_neg_sj

def run_logistic_analysis(df):
    antibody_features = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']
    ss_patients = df[df['Diagnosis'].str.contains("Sjögren syndrome", case=False, na=False)].copy()
    ra_patients = df[df['Diagnosis'].str.contains("Rheumatoid arthritis", case=False, na=False)].copy()
    
    rf_plus_ss = ss_patients[ss_patients['Rheumatoid_factor'] == 1].copy()
    rf_neg_ss = ss_patients[ss_patients['Rheumatoid_factor'] == 0].copy()
    
    analyses = []
    
    comparisons = [
        (rf_plus_ss, ra_patients, 'RF+ SS', 'RA', 'RF+ SS vs RA'),
        (rf_plus_ss, rf_neg_ss, 'RF+ SS', 'RF- SS', 'RF+ SS vs RF- SS'),
        (rf_neg_ss, ra_patients, 'RF- SS', 'RA', 'RF- SS vs RA'),
        (ss_patients, ra_patients, 'SS', 'RA', 'SS vs RA')
    ]
    
    for group1, group2, label1, label2, name in comparisons:
        if len(group1) > 0 and len(group2) > 0 and len(antibody_features) > 0:
            comparison_data = pd.concat([group1, group2], ignore_index=True)
            comparison_data['Target'] = [0] * len(group1) + [1] * len(group2)
            
            X = comparison_data[antibody_features]
            y = comparison_data['Target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=[label1, label2], zero_division=0))
            
            feature_importance = pd.DataFrame({
                'Feature': antibody_features,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)
            print("Feature Importance:")
            print(feature_importance)
            
            analyses.append({
                'name': name,
                'accuracy': accuracy,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'labels': [label1, label2],
                'feature_importance': feature_importance,
                'prevalence': pd.DataFrame({
                    'Feature': antibody_features * 2,
                    'Prevalence': [group1[ab].mean() for ab in antibody_features] + [group2[ab].mean() for ab in antibody_features],
                    'Group': [label1] * 4 + [label2] * 4
                })
            })
    
    return analyses

def chi_squared_analysis(df):
    df['Diagnosis_simple'] = df['Diagnosis'].str.lower().str.strip()
    df['Diagnosis_simple'] = df['Diagnosis_simple'].apply(
        lambda x: 'RA' if 'rheumatoid arthritis' in x else ('SS' if 'sj' in x else None)
    )
    df = df.dropna(subset=['Diagnosis_simple'])
    
    autoantibodies = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']
    chi2_results = []
    
    print("\nChi-Squared Test Results:")
    for ab in autoantibodies:
        if ab in df.columns:
            contingency = pd.crosstab(df['Diagnosis_simple'], df[ab])
            print(f"\nContingency table for {ab}:")
            print(contingency)
            if contingency.shape == (2, 2):
                chi2, p, dof, ex = chi2_contingency(contingency)
                chi2_results.append({'Antibody': ab, 'Chi2': chi2, 'P-value': p, 'Significant': p < 0.05})
                print(f"{ab:<10} | Chi2 = {chi2:.3f} | P-value = {p:.4f} | Significant: {p < 0.05}")
    
    return pd.DataFrame(chi2_results)

def distance_analysis(df):
    antibody_features = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']
    ss_patients = df[df['Diagnosis'].str.contains("Sjögren syndrome", case=False, na=False)].copy()
    ra_patients = df[df['Diagnosis'].str.contains("Rheumatoid arthritis", case=False, na=False)].copy()
    rf_plus_ss = ss_patients[ss_patients['Rheumatoid_factor'] == 1].copy()
    rf_neg_ss = ss_patients[ss_patients['Rheumatoid_factor'] == 0].copy()
    
    prevalence_data = {
        'RF+ SS': {
            'prevalence': [rf_plus_ss[ab].mean() for ab in antibody_features],
            'counts': [int(rf_plus_ss[ab].sum()) for ab in antibody_features],
            'total': len(rf_plus_ss)
        },
        'RF- SS': {
            'prevalence': [rf_neg_ss[ab].mean() for ab in antibody_features],
            'counts': [int(rf_neg_ss[ab].sum()) for ab in antibody_features],
            'total': len(rf_neg_ss)
        },
        'SS': {
            'prevalence': [ss_patients[ab].mean() for ab in antibody_features],
            'counts': [int(ss_patients[ab].sum()) for ab in antibody_features],
            'total': len(ss_patients)
        },
        'RA': {
            'prevalence': [ra_patients[ab].mean() for ab in antibody_features],
            'counts': [int(ra_patients[ab].sum()) for ab in antibody_features],
            'total': len(ra_patients)
        }
    }
    
    print("\nAntibody Prevalence and Positive Counts:")
    for group in prevalence_data:
        print(f"\n{group}:")
        df_prev = pd.DataFrame({
            'Antibody': antibody_features,
            'Prevalence': prevalence_data[group]['prevalence'],
            'Positive Counts': prevalence_data[group]['counts'],
            'Total Patients': [prevalence_data[group]['total']] * len(antibody_features)
        })
        print(df_prev)
    
    comparisons = [('RF+ SS', 'RA'), ('RF- SS', 'RA'), ('SS', 'RA')]
    results = []
    
    for group1, group2 in comparisons:
        vec1 = np.array(prevalence_data[group1]['prevalence'])
        vec2 = np.array(prevalence_data[group2]['prevalence'])
        
        binary_vec1 = vec1 > 0.5
        binary_vec2 = vec2 > 0.5
        
        euclidean_dist = euclidean(vec1, vec2)
        jaccard_dist = jaccard(binary_vec1, binary_vec2)
        n1 = prevalence_data[group1]['total']
        n2 = prevalence_data[group2]['total']
        weight = 2 * (n1 * n2) / (n1 + n2)
        
        results.append({
            'Comparison': f'{group1} vs {group2}',
            'Euclidean Distance': euclidean_dist,
            'Weighted Euclidean Distance': euclidean_dist * weight,
            'Jaccard Distance': jaccard_dist,
            'Weighted Jaccard Distance': jaccard_dist * weight
        })
    
    return pd.DataFrame(results), prevalence_data

def create_visualizations(analyses, chi2_results, distance_results, prevalence_data):
    # Accuracy comparison
    accuracy_data = pd.DataFrame({
        'Analysis': [a['name'] for a in analyses],
        'Accuracy': [a['accuracy'] for a in analyses]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Accuracy', y='Analysis', data=accuracy_data)
    plt.title('Accuracy Comparison Across Analyses')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1)
    for i, acc in enumerate(accuracy_data['Accuracy']):
        plt.text(acc + 0.01, i, f'{acc:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # Feature importance plots
    for analysis in analyses:
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Coefficient', y='Feature', data=analysis['feature_importance'])
        plt.title(f'Feature Importance: {analysis["name"]}')
        plt.tight_layout()
        name_clean = analysis["name"].replace(" ", "_").replace("+", "plus").replace("-", "minus")
        plt.savefig(os.path.join(output_dir, f'feature_importance_{name_clean}.png'))
        plt.close()
    
    # Confusion matrix heatmaps
    for analysis in analyses:
        plt.figure(figsize=(6, 5))
        sns.heatmap(analysis['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=analysis['labels'], yticklabels=analysis['labels'])
        plt.title(f'Confusion Matrix: {analysis["name"]}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name_clean}.png'))
        plt.close()
    
    # Prevalence plots
    for analysis in analyses:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Feature', y='Prevalence', hue='Group', data=analysis['prevalence'])
        plt.title(f'Antibody Prevalence: {analysis["name"]}')
        plt.ylabel('Proportion Positive')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prevalence_{name_clean}.png'))
        plt.close()
    
    # Chi-squared significance
    if not chi2_results.empty:
        chi2_results['-log10(P-value)'] = -np.log10(chi2_results['P-value'].clip(lower=1e-10))
        plt.figure(figsize=(8, 6))
        sns.barplot(x='-log10(P-value)', y='Antibody', hue='Significant', data=chi2_results)
        plt.title('Chi-Squared Test Significance')
        plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'chi2_significance.png'))
        plt.close()
        chi2_results.to_csv(os.path.join(output_dir, 'chi2_results.csv'), index=False)
    
    # Distance plots
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Euclidean Distance', y='Comparison', data=distance_results)
    plt.title('Euclidean Distance Between Groups')
    plt.xlim(0, max(distance_results['Euclidean Distance']) + 0.1)
    for i, dist in enumerate(distance_results['Euclidean Distance']):
        plt.text(dist + 0.01, i, f'{dist:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'euclidean_distance.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Jaccard Distance', y='Comparison', data=distance_results)
    plt.title('Jaccard Distance Between Groups')
    plt.xlim(0, 1)
    for i, dist in enumerate(distance_results['Jaccard Distance']):
        plt.text(dist + 0.01, i, f'{dist:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jaccard_distance.png'))
    plt.close()
    
    # Euclidean distance heatmap
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
    plt.savefig(os.path.join(output_dir, 'euclidean_distance_heatmap.png'))
    plt.close()
    
    # Dendrogram
    prevalence_vectors = np.array([prevalence_data[group]['prevalence'] for group in groups])
    linkage_matrix = linkage(prevalence_vectors, method='ward', metric='euclidean')
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=groups, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram (Euclidean Distance)')
    plt.xlabel('Group')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dendrogram.png'))
    plt.close()

def main():
    print("=== Pipeline Start ===")
    df = load_data()
    
    print("\n=== Step 1: Patient Counts ===")
    patient_counts(df)
    
    print("\n=== Step 2: Logistic Regression Analysis ===")
    analyses = run_logistic_analysis(df)
    
    print("\n=== Step 3: Chi-Squared Analysis ===")
    chi2_results = chi_squared_analysis(df)
    print(chi2_results)
    
    print("\n=== Step 4: Distance Analysis ===")
    distance_results, prevalence_data = distance_analysis(df)
    print(distance_results)
    
    print("\n=== Step 5: Creating Visualizations ===")
    create_visualizations(analyses, chi2_results, distance_results, prevalence_data)
    
    print("\n=== Final Comparison ===")
    print(f"RF+ SS vs RA accuracy: {analyses[0]['accuracy']:.4f}")
    print(f"RF+ SS vs RF- SS accuracy: {analyses[1]['accuracy']:.4f}")
    print(f"RF- SS vs RA accuracy: {analyses[2]['accuracy']:.4f}")
    print(f"SS vs RA accuracy: {analyses[3]['accuracy']:.4f}")
    
    print("\n=== Conclusions ===")
    if analyses[0]['accuracy'] < analyses[1]['accuracy']:
        print("1. RF+ SS is HARDER to distinguish from RA than from RF- SS")
        print("   This SUPPORTS your hypothesis that RF+ SS is closer to RA than to RF- SS")
    else:
        print("1. RF+ SS is EASIER to distinguish from RF- SS than from RA")
        print("   This suggests RF+ SS and RF- SS are more distinct than RF+ SS and RA")
    
    if analyses[2]['accuracy'] > analyses[0]['accuracy']:
        print("2. RF- SS is EASIER to distinguish from RA than RF+ SS is")
        print("   This further SUPPORTS your hypothesis that RF+ SS is closer to RA")
    else:
        print("2. RF- SS is HARDER to distinguish from RA than RF+ SS is")
        print("   This suggests RF- SS is closer to RA than RF+ SS is")
    
    if analyses[3]['accuracy'] > max(analyses[0]['accuracy'], analyses[2]['accuracy']):
        print("3. SS (overall) is EASIER to distinguish from RA than RF+ SS or RF- SS")
        print("   This suggests SS and RA are more distinct when not split by RF status")
    else:
        print("3. SS (overall) is HARDER to distinguish from RA than RF+ SS or RF- SS")
        print("   This suggests splitting SS by RF status improves differentiation from RA")
    
    print("\nPipeline complete! Outputs saved to 'output/' directory.")

if __name__ == "__main__":
    main()