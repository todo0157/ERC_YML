import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
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
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42
K_STEPS = [20, 40, 60, 80, 100]

print("Starting Comparative Analysis v2 (Enhanced, Top K Search)...")
print(f"Loading features from: {feature_dir}")

# Load Datasets (Pickle)
try:
    df_pm_all = pd.read_pickle(os.path.join(feature_dir, "PM_Only.pkl"))
    df_voc_all = pd.read_pickle(os.path.join(feature_dir, "VOC_Only.pkl"))
    df_int_all = pd.read_pickle(os.path.join(feature_dir, "Integrated.pkl"))
except Exception as e:
    print(f"Error loading pickle files: {e}")
    # Fallback to loading Excel if pickle fails (just in case user reverted)
    print("Trying to load Excel fallback...")
    try:
        excel_path = os.path.join(base_dir, "..", "results_features_p10to100_enhanced.xlsx")
        df_pm_all = pd.read_excel(excel_path, sheet_name='PM_Only')
        df_voc_all = pd.read_excel(excel_path, sheet_name='VOC_Only')
        df_int_all = pd.read_excel(excel_path, sheet_name='Integrated')
    except Exception as e2:
        print(f"Fatal error loading data: {e2}")
        exit()

def get_data_for_pct(df_all, pct):
    """Filter data by percentage segment using 'Segment_Pct' column."""
    if 'Segment_Pct' in df_all.columns:
        return df_all[df_all['Segment_Pct'] == pct].copy()
    else:
        # If Segment_Pct is missing, try to parse from index 'data{i}_{pct}'
        # Assuming index name is 'id' or it is the index
        if df_all.index.name == 'id' or 'id' in df_all.columns:
            # We need to rely on the fact that we split data by pct during extraction
            # But the extracted files (PM_Only.pkl etc) are CONCATENATED.
            # So they SHOULD have 'Segment_Pct'.
            # If step1/step2 saved correctly, it's there.
            pass
        return pd.DataFrame()

def evaluate_top_k(X, y, model_type='SVR'):
    """
    1. Select Top 100 features using SelectKBest.
    2. Calculate Permutation Importance.
    3. Evaluate K=[20, 40, 60, 80, 100] and find Best K.
    """
    
    # 0. Handle NaNs and Infs
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 1. Remove Constant Features
    sel_var = VarianceThreshold(threshold=0)
    X_var = sel_var.fit_transform(X_imputed)
    
    # 2. First Selection: Top 100 (using f_regression)
    n_features = X_var.shape[1]
    k_first = min(100, n_features)
    
    sel_kbest = SelectKBest(f_regression, k=k_first)
    X_kbest = sel_kbest.fit_transform(X_var, y)
    
    # Scale for Model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kbest)
    
    # 3. Permutation Importance (to rank the 100 features)
    if model_type == 'SVR':
        model = SVR(C=1.0, epsilon=0.1)
    else:
        model = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    
    # Fit model once on all data to get importance (or use CV importance?)
    # Using single fit for importance ranking is faster and standard
    model.fit(X_scaled, y)
    
    if model_type == 'ExtraTrees':
        importances = model.feature_importances_
    else:
        # SVR doesn't have feature_importances_, use Permutation Importance
        r = permutation_importance(model, X_scaled, y, n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
        importances = r.importances_mean
        
    # Rank indices
    ranks = np.argsort(importances)[::-1] # Descending order indices
    
    # 4. Search Best K
    best_re = float('inf')
    best_k = 0
    
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    
    # Filter valid K steps (cannot exceed k_first)
    valid_k_steps = [k for k in K_STEPS if k <= k_first]
    if not valid_k_steps:
        valid_k_steps = [k_first]
        
    for k in valid_k_steps:
        # Select Top K columns based on importance rank
        top_k_indices = ranks[:k]
        X_final = X_scaled[:, top_k_indices]
        
        re_scores = []
        
        for train_idx, test_idx in rkf.split(X_final):
            X_train, X_test = X_final[train_idx], X_final[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Re-instantiate model to avoid leakage or state issues
            if model_type == 'SVR':
                sub_model = SVR(C=1.0, epsilon=0.1)
            else:
                sub_model = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
                
            sub_model.fit(X_train, y_train)
            y_pred = sub_model.predict(X_test)
            
            # RE
            y_test_safe = np.where(y_test == 0, 1e-6, y_test)
            re = np.abs((y_test - y_pred) / y_test_safe)
            re_scores.append(np.mean(re))
            
        mean_re = np.mean(re_scores)
        
        if mean_re < best_re:
            best_re = mean_re
            best_k = k
            
    return best_re, best_k

# Main Loop
results_summary = []

print(f"{'Pct':<5} | {'Model':<12} | {'Dataset':<12} | {'Best RE':<10} | {'Best K':<5}")
print("-" * 65)

for pct in PCTS:
    # Prepare Data
    data_pm = get_data_for_pct(df_pm_all, pct)
    data_voc = get_data_for_pct(df_voc_all, pct)
    data_int = get_data_for_pct(df_int_all, pct)
    
    if data_pm.empty:
        continue
        
    # Target (Assume same for all, taken from PM dataset meta)
    # Target_PM_Mean column should exist
    y = data_pm['Target_PM_Mean'].values
    
    # Feature Matrices (Exclude Meta Columns)
    meta_cols = ['Segment_Pct', 'Target_PM_Mean', 'Sample_ID']
    
    X_pm = data_pm.drop(columns=[c for c in meta_cols if c in data_pm.columns]).select_dtypes(include=[np.number]).values
    X_voc = data_voc.drop(columns=[c for c in meta_cols if c in data_voc.columns]).select_dtypes(include=[np.number]).values
    X_int = data_int.drop(columns=[c for c in meta_cols if c in data_int.columns]).select_dtypes(include=[np.number]).values
    
    datasets = {'PM_Only': X_pm, 'VOC_Only': X_voc, 'Integrated': X_int}
    models = ['SVR', 'ExtraTrees']
    
    for m_name in models:
        for d_name, X_data in datasets.items():
            best_re, best_k = evaluate_top_k(X_data, y, m_name)
            
            results_summary.append({
                'Percentage': pct,
                'Model': m_name,
                'Dataset': d_name,
                'Best_RE': best_re,
                'Best_K': best_k
            })
            
            print(f"{pct:<5} | {m_name:<12} | {d_name:<12} | {best_re:.4f}     | {best_k}")

# Save Results
df_res = pd.DataFrame(results_summary)
res_file = os.path.join(output_dir, "Enhanced_Analysis_Results_v2.xlsx")
df_res.to_excel(res_file, index=False)
print(f"\nAnalysis Saved to {res_file}")

# Visualization
for m_name in ['SVR', 'ExtraTrees']:
    plt.figure(figsize=(12, 6))
    
    df_m = df_res[df_res['Model'] == m_name]
    
    # Plot Lines
    sns.lineplot(data=df_m, x='Percentage', y='Best_RE', hue='Dataset', style='Dataset', markers=True, dashes=False, linewidth=2.5)
    
    # Add Text Annotations (Best K)
    for idx, row in df_m.iterrows():
        # Offset slightly to avoid overlap
        offset = 0.005 if row['Dataset'] == 'PM_Only' else -0.005
        plt.text(row['Percentage'], row['Best_RE'] + offset, f"K={row['Best_K']}", 
                 fontsize=9, ha='center', color='black')
                 
    plt.title(f'Enhanced Model Comparison: {m_name} (Top K Selection)')
    plt.ylabel('Best Mean Relative Error (RE)')
    plt.xlabel('Process Completion (%)')
    plt.grid(True, alpha=0.3)
    
    # Save
    plot_path = os.path.join(output_dir, f"Fig_{m_name}_Enhanced_Comparison_v2.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved: {plot_path}")

print("All tasks complete.")

