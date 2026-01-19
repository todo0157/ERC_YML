import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_dir = os.path.join(base_dir, "..", "Enhanced_Features_v2")
output_dir = os.path.join(base_dir, "..", "Comparative_Analysis_Enhanced_v2")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # Analyze ALL segments
RANDOM_STATE = 42

print("Starting Top Feature Analysis (All Segments)...")

# Load Datasets (Pickle)
try:
    df_voc_all = pd.read_pickle(os.path.join(feature_dir, "VOC_Only.pkl"))
    df_int_all = pd.read_pickle(os.path.join(feature_dir, "Integrated.pkl"))
except Exception as e:
    print(f"Error loading pickle files: {e}")
    exit()

def get_data_for_pct(df_all, pct):
    if 'Segment_Pct' in df_all.columns:
        return df_all[df_all['Segment_Pct'] == pct].copy()
    return pd.DataFrame()

def analyze_features(X, y, feature_names, model_type='ExtraTrees'):
    """Calculate feature importance and return top features."""
    
    # 0. Handle NaNs
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 1. Variance Threshold
    sel_var = VarianceThreshold(threshold=0)
    X_var = sel_var.fit_transform(X_imputed)
    remaining_indices = sel_var.get_support(indices=True)
    current_feat_names = np.array(feature_names)[remaining_indices]
    
    # 2. SelectKBest (Top 100)
    k_first = min(100, X_var.shape[1])
    sel_kbest = SelectKBest(f_regression, k=k_first)
    X_kbest = sel_kbest.fit_transform(X_var, y)
    
    # Get selected feature names
    selected_mask = sel_kbest.get_support()
    final_feat_names = current_feat_names[selected_mask]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kbest)
    
    # 3. Model Importance
    if model_type == 'SVR':
        model = SVR(C=1.0, epsilon=0.1)
        model.fit(X_scaled, y)
        r = permutation_importance(model, X_scaled, y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
        importances = r.importances_mean
    else:
        model = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_scaled, y)
        importances = model.feature_importances_
        
    # Create DataFrame
    df_imp = pd.DataFrame({
        'Feature': final_feat_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return df_imp

results = []

for pct in PCTS:
    print(f"\nAnalyzing {pct}% Segment...")
    
    # Analyze Integrated (ExtraTrees only as it performed best)
    data_int = get_data_for_pct(df_int_all, pct)
    if not data_int.empty:
        y = data_int['Target_PM_Mean'].values
        meta_cols = ['Segment_Pct', 'Target_PM_Mean', 'Sample_ID']
        X_df = data_int.drop(columns=[c for c in meta_cols if c in data_int.columns]).select_dtypes(include=[np.number])
        X = X_df.values
        feat_names = X_df.columns.tolist()
        
        df_top = analyze_features(X, y, feat_names, 'ExtraTrees')
        
        # Tag feature types
        df_top['Type'] = 'Other'
        df_top.loc[df_top['Feature'].str.contains('Prod_'), 'Type'] = 'Interaction (Prod)'
        df_top.loc[df_top['Feature'].str.contains('Ratio_'), 'Type'] = 'Interaction (Ratio)'
        df_top.loc[df_top['Feature'].str.contains('Lag_'), 'Type'] = 'Enhanced (Lag)'
        df_top.loc[df_top['Feature'].str.contains('Roll_'), 'Type'] = 'Enhanced (Rolling)'
        # Rolling + Lag combination
        df_top.loc[df_top['Feature'].str.contains('Roll_') & df_top['Feature'].str.contains('Lag_'), 'Type'] = 'Enhanced (Roll+Lag)'
        
        # Add metadata
        df_top['Percentage'] = pct
        df_top['Model'] = 'ExtraTrees'
        df_top['Dataset'] = 'Integrated'
        
        results.append(df_top.head(30)) # Save Top 30
        
        print(f"Top 5 Features for {pct}% (Integrated):")
        print(df_top[['Feature', 'Importance', 'Type']].head(5))

# Save
df_final = pd.concat(results)
out_file = os.path.join(output_dir, "Top_Features_Analysis.xlsx")
df_final.to_excel(out_file, index=False)
print(f"\nSaved Top Features to {out_file}")

# Summary Plot of Feature Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df_final, x='Percentage', hue='Type')
plt.title('Distribution of Feature Types in Top 30 (Integrated, ExtraTrees)')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, "Feature_Type_Distribution.png"))
print("Saved Distribution Plot.")

