# Load modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load your data, selecting only the first 32 columns (up to ACPA)
folder_path = r"C:\Users\natha\Documents\Polygence Reasearch Project\Autoimmune Diseases\Autoimmune_Disorder_10k_with_All_Disorders.csv"

try:
    # Get column names up to 'ACPA'
    all_columns = pd.read_csv(folder_path, nrows=0).columns
    selected_columns = all_columns[:all_columns.get_loc('ACPA') + 1].tolist()
    df = pd.read_csv(folder_path, usecols=selected_columns)
    print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Clean column names and handle missing values
df.columns = [col.replace('-', '_').replace(' ', '_') for col in df.columns]
df = df.fillna(0)

# Print unique Diagnosis values for debugging
print("\nUnique Diagnosis values:")
print(df['Diagnosis'].unique())

# Verify columns and their data types
print("\nAll columns and their data types:")
print(df.dtypes)

# Define antibody features
antibody_features = ['ACPA', 'ANA', 'Anti_dsDNA', 'Anti_Sm']

print("\n=== FOCUSED ANALYSIS: RF+ SS vs RA, RF+ SS vs RF- SS, RF- SS vs RA, SS vs RA USING 4 ANTIBODIES ===")

# Get the specific patient groups
ss_patients = df[df['Diagnosis'].str.contains("Sjögren syndrome", case=False, na=False)].copy()
ra_patients = df[df['Diagnosis'].str.contains("Rheumatoid arthritis", case=False, na=False)].copy()

print(f"Sjögren syndrome patients: {len(ss_patients)}")
print(f"Rheumatoid arthritis patients: {len(ra_patients)}")

# Split SS into RF+ and RF- based on Rheumatoid_factor
if len(ss_patients) > 0:
    rf_plus_ss = ss_patients[ss_patients['Rheumatoid_factor'] == 1].copy()
    rf_neg_ss = ss_patients[ss_patients['Rheumatoid_factor'] == 0].copy()
    
    print(f"RF+ SS patients: {len(rf_plus_ss)}")
    print(f"RF- SS patients: {len(rf_neg_ss)}")
    
    # Reset indices
    rf_plus_ss = rf_plus_ss.reset_index(drop=True)
    rf_neg_ss = rf_neg_ss.reset_index(drop=True)
    ra_patients = ra_patients.reset_index(drop=True)
    
    print(f"\nUsing these 4 antibodies: {antibody_features}")
    
    # Check if all antibody features exist in the dataframe
    missing_features = [feature for feature in antibody_features if feature not in rf_plus_ss.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        print(f"Available columns: {[col for col in rf_plus_ss.columns if any(antibody in col for antibody in ['ACPA', 'ANA', 'dsDNA', 'Sm'])]}")
        # Remove missing features
        antibody_features = [feature for feature in antibody_features if feature not in missing_features]
        print(f"Using available features: {antibody_features}")
    
    # Debugging: Check data types and sample data
    print("\nColumn data types for antibody features:")
    print(rf_plus_ss[antibody_features].dtypes)
    print("\nSample data for RF+ SS:")
    print(rf_plus_ss[antibody_features].head())
    print("\nSample data for RA:")
    print(ra_patients[antibody_features].head())
    print("\nSample data for RF- SS:")
    print(rf_neg_ss[antibody_features].head())
    
    # ANALYSIS 1: RF+ SS vs RA (Main hypothesis test)
    if len(rf_plus_ss) > 0 and len(ra_patients) > 0 and len(antibody_features) > 0:
        print(f"\n--- ANALYSIS 1: RF+ SS vs RA ---")
        
        # Create comparison dataset
        comparison_data = pd.concat([rf_plus_ss, ra_patients], ignore_index=True)
        comparison_data['Group'] = ['RF+_SS'] * len(rf_plus_ss) + ['RA'] * len(ra_patients)
        comparison_data['Target'] = (comparison_data['Group'] == 'RA').astype(int)  # 0=RF+ SS, 1=RA
        
        print(f"Total samples: {len(comparison_data)}")
        print(f"RF+ SS: {len(rf_plus_ss)}, RA: {len(ra_patients)}")
        
        # Use only the available antibody features
        X = comparison_data[antibody_features]
        y = comparison_data['Target']
        
        print(f"\nFeature summary:")
        for feature in antibody_features:
            if feature in rf_plus_ss.columns and feature in ra_patients.columns:
                try:
                    rf_plus_mean = rf_plus_ss[feature].astype(float).mean()
                    ra_mean = ra_patients[feature].astype(float).mean()
                    print(f"{feature}: RF+ SS mean = {rf_plus_mean:.3f}, RA mean = {ra_mean:.3f}")
                except (ValueError, TypeError) as e:
                    print(f"Error computing mean for {feature}: {e}")
            else:
                print(f"Feature {feature} not found in the dataset.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression with class weights
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nLogistic Regression Results (RF+ SS vs RA):")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['RF+ SS', 'RA'], zero_division=0))
        
        # Verify feature alignment
        print(f"\nFeatures used in model: {antibody_features}")
        print(f"Number of coefficients: {len(model.coef_[0])}")
        
        # Feature importance
        coefficients = model.coef_[0]
        if len(coefficients) == len(antibody_features):
            feature_importance = pd.DataFrame({
                'feature': antibody_features,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("\nFeature Importance (coefficients):")
            print(feature_importance[['feature', 'coefficient']])
        else:
            print(f"ERROR: Mismatch between number of features ({len(antibody_features)}) and coefficients ({len(coefficients)})")
        
        # Interpretation
        print(f"\n=== INTERPRETATION ===")
        if accuracy > 0.8:
            print("HIGH accuracy (>0.8): RF+ SS and RA are DISTINCT based on these antibodies")
            print("RF+ SS is different from RA")
        elif accuracy < 0.6:
            print("LOW accuracy (<0.6): RF+ SS and RA are SIMILAR based on these antibodies")
            print("SUPPORTS HYPOTHESIS: RF+ SS is closer to RA than to RF- SS")
        else:
            print(f"MODERATE accuracy ({accuracy:.3f}): Some overlap between RF+ SS and RA")
    
    # ANALYSIS 2: RF+ SS vs RF- SS (Comparison)
    if len(rf_plus_ss) > 0 and len(rf_neg_ss) > 0 and len(antibody_features) > 0:
        print(f"\n--- ANALYSIS 2: RF+ SS vs RF- SS ---")
        
        comparison_data2 = pd.concat([rf_plus_ss, rf_neg_ss], ignore_index=True)
        comparison_data2['Group'] = ['RF+_SS'] * len(rf_plus_ss) + ['RF-_SS'] * len(rf_neg_ss)
        comparison_data2['Target'] = (comparison_data2['Group'] == 'RF-_SS').astype(int)  # 0=RF+ SS, 1=RF- SS
        
        print(f"Total samples: {len(comparison_data2)}")
        print(f"RF+ SS: {len(rf_plus_ss)}, RF- SS: {len(rf_neg_ss)}")
        
        X2 = comparison_data2[antibody_features]
        y2 = comparison_data2['Target']
        
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=0.3, random_state=42, stratify=y2
        )
        
        X2_train_scaled = scaler.fit_transform(X2_train)
        X2_test_scaled = scaler.transform(X2_test)
        
        model2 = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model2.fit(X2_train_scaled, y2_train)
        
        y2_pred = model2.predict(X2_test_scaled)
        accuracy2 = accuracy_score(y2_test, y2_pred)
        
        print(f"Accuracy (RF+ SS vs RF- SS): {accuracy2:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y2_test, y2_pred))
        print("\nClassification Report:")
        print(classification_report(y2_test, y2_pred, target_names=['RF+ SS', 'RF- SS'], zero_division=0))
        
        # Feature importance
        coefficients2 = model2.coef_[0]
        if len(coefficients2) == len(antibody_features):
            feature_importance2 = pd.DataFrame({
                'feature': antibody_features,
                'coefficient': coefficients2,
                'abs_coefficient': np.abs(coefficients2)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("\nFeature Importance (coefficients):")
            print(feature_importance2[['feature', 'coefficient']])
        else:
            print(f"ERROR: Mismatch between number of features ({len(antibody_features)}) and coefficients ({len(coefficients2)})")
        
        # Interpretation
        print(f"\n=== INTERPRETATION ===")
        if accuracy2 > 0.8:
            print("HIGH accuracy (>0.8): RF+ SS and RF- SS are DISTINCT based on these antibodies")
        elif accuracy2 < 0.6:
            print("LOW accuracy (<0.6): RF+ SS and RF- SS are SIMILAR based on these antibodies")
        else:
            print(f"MODERATE accuracy ({accuracy2:.3f}): Some overlap between RF+ SS and RF- SS")
    
    # ANALYSIS 3: RF- SS vs RA (Comparison)
    if len(rf_neg_ss) > 0 and len(ra_patients) > 0 and len(antibody_features) > 0:
        print(f"\n--- ANALYSIS 3: RF- SS vs RA ---")
        
        comparison_data3 = pd.concat([rf_neg_ss, ra_patients], ignore_index=True)
        comparison_data3['Group'] = ['RF-_SS'] * len(rf_neg_ss) + ['RA'] * len(ra_patients)
        comparison_data3['Target'] = (comparison_data3['Group'] == 'RA').astype(int)  # 0=RF- SS, 1=RA
        
        print(f"Total samples: {len(comparison_data3)}")
        print(f"RF- SS: {len(rf_neg_ss)}, RA: {len(ra_patients)}")
        
        X3 = comparison_data3[antibody_features]
        y3 = comparison_data3['Target']
        
        X3_train, X3_test, y3_train, y3_test = train_test_split(
            X3, y3, test_size=0.3, random_state=42, stratify=y3
        )
        
        X3_train_scaled = scaler.fit_transform(X3_train)
        X3_test_scaled = scaler.transform(X3_test)
        
        model3 = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model3.fit(X3_train_scaled, y3_train)
        
        y3_pred = model3.predict(X3_test_scaled)
        accuracy3 = accuracy_score(y3_test, y3_pred)
        
        print(f"Accuracy (RF- SS vs RA): {accuracy3:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y3_test, y3_pred))
        print("\nClassification Report:")
        print(classification_report(y3_test, y3_pred, target_names=['RF- SS', 'RA'], zero_division=0))
        
        # Feature importance
        coefficients3 = model3.coef_[0]
        if len(coefficients3) == len(antibody_features):
            feature_importance3 = pd.DataFrame({
                'feature': antibody_features,
                'coefficient': coefficients3,
                'abs_coefficient': np.abs(coefficients3)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("\nFeature Importance (coefficients):")
            print(feature_importance3[['feature', 'coefficient']])
        else:
            print(f"ERROR: Mismatch between number of features ({len(antibody_features)}) and coefficients ({len(coefficients3)})")
        
        # Interpretation
        print(f"\n=== INTERPRETATION ===")
        if accuracy3 > 0.8:
            print("HIGH accuracy (>0.8): RF- SS and RA are DISTINCT based on these antibodies")
        elif accuracy3 < 0.6:
            print("LOW accuracy (<0.6): RF- SS and RA are SIMILAR based on these antibodies")
        else:
            print(f"MODERATE accuracy ({accuracy3:.3f}): Some overlap between RF- SS and RA")
    
    # ANALYSIS 4: SS vs RA (Comparison)
    if len(ss_patients) > 0 and len(ra_patients) > 0 and len(antibody_features) > 0:
        print(f"\n--- ANALYSIS 4: SS vs RA ---")
        
        comparison_data4 = pd.concat([ss_patients, ra_patients], ignore_index=True)
        comparison_data4['Group'] = ['SS'] * len(ss_patients) + ['RA'] * len(ra_patients)
        comparison_data4['Target'] = (comparison_data4['Group'] == 'RA').astype(int)  # 0=SS, 1=RA
        
        print(f"Total samples: {len(comparison_data4)}")
        print(f"SS: {len(ss_patients)}, RA: {len(ra_patients)}")
        
        X4 = comparison_data4[antibody_features]
        y4 = comparison_data4['Target']
        
        X4_train, X4_test, y4_train, y4_test = train_test_split(
            X4, y4, test_size=0.3, random_state=42, stratify=y4
        )
        
        X4_train_scaled = scaler.fit_transform(X4_train)
        X4_test_scaled = scaler.transform(X4_test)
        
        model4 = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model4.fit(X4_train_scaled, y4_train)
        
        y4_pred = model4.predict(X4_test_scaled)
        accuracy4 = accuracy_score(y4_test, y4_pred)
        
        print(f"Accuracy (SS vs RA): {accuracy4:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y4_test, y4_pred))
        print("\nClassification Report:")
        print(classification_report(y4_test, y4_pred, target_names=['SS', 'RA'], zero_division=0))
        
        # Feature importance
        coefficients4 = model4.coef_[0]
        if len(coefficients4) == len(antibody_features):
            feature_importance4 = pd.DataFrame({
                'feature': antibody_features,
                'coefficient': coefficients4,
                'abs_coefficient': np.abs(coefficients4)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("\nFeature Importance (coefficients):")
            print(feature_importance4[['feature', 'coefficient']])
        else:
            print(f"ERROR: Mismatch between number of features ({len(antibody_features)}) and coefficients ({len(coefficients4)})")
        
        # Interpretation
        print(f"\n=== INTERPRETATION ===")
        if accuracy4 > 0.8:
            print("HIGH accuracy (>0.8): SS and RA are DISTINCT based on these antibodies")
        elif accuracy4 < 0.6:
            print("LOW accuracy (<0.6): SS and RA are SIMILAR based on these antibodies")
        else:
            print(f"MODERATE accuracy ({accuracy4:.3f}): Some overlap between SS and RA")
    
    # Final comparison
    print(f"\n=== FINAL COMPARISON ===")
    print(f"RF+ SS vs RA accuracy: {accuracy:.4f}")
    print(f"RF+ SS vs RF- SS accuracy: {accuracy2:.4f}")
    print(f"RF- SS vs RA accuracy: {accuracy3:.4f}")
    print(f"SS vs RA accuracy: {accuracy4:.4f}")
    
    print("\n=== CONCLUSIONS ===")
    if accuracy < accuracy2:
        print("1. RF+ SS is HARDER to distinguish from RA than from RF- SS")
        print("   This SUPPORTS your hypothesis that RF+ SS is closer to RA than to RF- SS")
    else:
        print("1. RF+ SS is EASIER to distinguish from RF- SS than from RA")
        print("   This suggests RF+ SS and RF- SS are more distinct than RF+ SS and RA")
    
    if accuracy3 > accuracy:
        print("2. RF- SS is EASIER to distinguish from RA than RF+ SS is")
        print("   This further SUPPORTS your hypothesis that RF+ SS is closer to RA")
    else:
        print("2. RF- SS is HARDER to distinguish from RA than RF+ SS is")
        print("   This suggests RF- SS is closer to RA than RF+ SS is")
    
    if accuracy4 > max(accuracy, accuracy3):
        print("3. SS (overall) is EASIER to distinguish from RA than RF+ SS or RF- SS")
        print("   This suggests SS and RA are more distinct when not split by RF status")
    else:
        print("3. SS (overall) is HARDER to distinguish from RA than RF+ SS or RF- SS")
        print("   This suggests splitting SS by RF status improves differentiation from RA")

print("\nAnalysis complete!")