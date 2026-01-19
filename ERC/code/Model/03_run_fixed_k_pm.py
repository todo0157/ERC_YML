import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
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

print("Starting Fixed K Analysis (Based on VOC Best K)...")

# Load Datasets
try:
    df_pm_all = pd.read_pickle(os.path.join(feature_dir, "PM_Only.pkl"))
    df_voc_all = pd.read_pickle(os.path.join(feature_dir, "VOC_Only.pkl"))
    df_int_all = pd.read_pickle(os.path.join(feature_dir, "Integrated.pkl"))
except Exception as e:
    print(f"Error loading pickle files: {e}")
    exit()

def get_data_for_pct(df_all, pct):
    """Filter data by percentage segment."""
    if 'Segment_Pct' in df_all.columns:
        return df_all[df_all['Segment_Pct'] == pct].copy()
    return pd.DataFrame()

def get_feature_ranking(X, y, model_type='SVR'):
    """Rank features using SelectKBest(100) -> Permutation Importance."""
    # 0. Handle NaNs
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 1. Variance Threshold
    sel_var = VarianceThreshold(threshold=0)
    X_var = sel_var.fit_transform(X_imputed)
    
    # 2. Select Top 100
    n_features = X_var.shape[1]
    k_first = min(100, n_features)
    
    sel_kbest = SelectKBest(f_regression, k=k_first)
    X_kbest = sel_kbest.fit_transform(X_var, y)
    
    # Scale (Use RobustScaler for SVR, StandardScaler for ExtraTrees)
    if model_type == 'SVR':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X_kbest)
    
    # 3. Permutation Importance
    if model_type == 'SVR':
        # Use Tuned Hyperparameters for Ranking as well
        # Using Log Transform for y during fitting
        y_log = np.log1p(y)
        model = SVR(C=1000, epsilon=0.001, gamma=0.001) # Use tuned params
        model.fit(X_scaled, y_log)
        
        # Calculate PI (Note: PI on log-transformed y)
        r = permutation_importance(model, X_scaled, y_log, n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
        importances = r.importances_mean
    else:
        model = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_scaled, y)
        importances = model.feature_importances_
        
    ranks = np.argsort(importances)[::-1] # Descending
    
    return X_scaled, ranks

def evaluate_with_fixed_k(X_scaled, ranks, y, k, model_type='SVR'):
    """Evaluate model using top k features."""
    
    # Select Top K columns
    top_k_indices = ranks[:k]
    X_final = X_scaled[:, top_k_indices]
    
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    re_scores = []
    
    # Pre-calculate Log Y for SVR
    if model_type == 'SVR':
        y_log = np.log1p(y)
    
    for train_idx, test_idx in rkf.split(X_final):
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        
        if model_type == 'SVR':
            # Train on Log Y
            y_train = y_log[train_idx]
            y_test_orig = y[test_idx] # For evaluation
            
            model = SVR(C=1000, epsilon=0.001, gamma=0.001) # Use tuned params
            model.fit(X_train, y_train)
            
            # Predict & Inverse Transform
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            
        else:
            y_train, y_test_orig = y[train_idx], y[test_idx]
            
            model = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        y_test_safe = np.where(y_test_orig == 0, 1e-6, y_test_orig)
        re = np.abs((y_test_orig - y_pred) / y_test_safe)
        re_scores.extend(re)
        
    return np.mean(re_scores), np.std(re_scores)

# Storage for results
results_list = []

# Step 1 & 2 Loop
for pct in PCTS:
    print(f"\nProcessing {pct}% Segment...")
    
    # Prepare Data
    data_pm = get_data_for_pct(df_pm_all, pct)
    data_voc = get_data_for_pct(df_voc_all, pct)
    data_int = get_data_for_pct(df_int_all, pct)
    
    if data_voc.empty or data_pm.empty or data_int.empty:
        print(f"Skipping {pct}% due to missing data.")
        continue
        
    y = data_voc['Target_PM_Mean'].values
    
    meta_cols = ['Segment_Pct', 'Target_PM_Mean', 'Sample_ID']
    X_pm = data_pm.drop(columns=[c for c in meta_cols if c in data_pm.columns]).select_dtypes(include=[np.number]).values
    X_voc = data_voc.drop(columns=[c for c in meta_cols if c in data_voc.columns]).select_dtypes(include=[np.number]).values
    X_int = data_int.drop(columns=[c for c in meta_cols if c in data_int.columns]).select_dtypes(include=[np.number]).values
    
    for model_name in ['SVR', 'ExtraTrees']:
        print(f"  Model: {model_name}")
        
        # --- Step 1: Find Best K for VOC ---
        # Rank features for VOC
        X_voc_sc, ranks_voc = get_feature_ranking(X_voc, y, model_name)
        
        best_k_voc = 0
        best_re_voc = float('inf')
        best_std_voc = 0
        
        # Search Best K for VOC
        for k in K_STEPS:
            # Check if we have enough features
            if k > X_voc_sc.shape[1]:
                continue
                
            mean_re, std_re = evaluate_with_fixed_k(X_voc_sc, ranks_voc, y, k, model_name)
            
            if mean_re < best_re_voc:
                best_re_voc = mean_re
                best_std_voc = std_re
                best_k_voc = k
        
        print(f"    Best K (VOC): {best_k_voc} (RE: {best_re_voc:.4f})")
        
        # Record VOC Result
        results_list.append({
            'Model': model_name,
            'Segment_Pct': pct,
            'Case': 'VOC_Only',
            'Applied_K': best_k_voc,
            'Mean_RE': best_re_voc,
            'Std_RE': best_std_voc
        })
        
        # --- Step 2: Apply Fixed K to PM and Integrated ---
        
        # PM Only
        X_pm_sc, ranks_pm = get_feature_ranking(X_pm, y, model_name)
        # Apply Fixed K
        k_fixed = min(best_k_voc, X_pm_sc.shape[1])
        re_pm, std_pm = evaluate_with_fixed_k(X_pm_sc, ranks_pm, y, k_fixed, model_name)
        results_list.append({
            'Model': model_name,
            'Segment_Pct': pct,
            'Case': 'PM_Only',
            'Applied_K': k_fixed,
            'Mean_RE': re_pm,
            'Std_RE': std_pm
        })
        
        # Integrated
        X_int_sc, ranks_int = get_feature_ranking(X_int, y, model_name)
        # Apply Fixed K
        k_fixed = min(best_k_voc, X_int_sc.shape[1])
        re_int, std_int = evaluate_with_fixed_k(X_int_sc, ranks_int, y, k_fixed, model_name)
        results_list.append({
            'Model': model_name,
            'Segment_Pct': pct,
            'Case': 'Integrated',
            'Applied_K': k_fixed,
            'Mean_RE': re_int,
            'Std_RE': std_int
        })

# Save Results
df_results = pd.DataFrame(results_list)
excel_path = os.path.join(output_dir, "Fixed_K_Analysis_Result.xlsx")
df_results.to_excel(excel_path, index=False)
print(f"Results saved to {excel_path}")

# Plotting
for model_name in ['SVR', 'ExtraTrees']:
    df_model = df_results[df_results['Model'] == model_name]
    
    plt.figure(figsize=(12, 6))
    
    # Colors
    colors = {'PM_Only': 'blue', 'VOC_Only': 'green', 'Integrated': 'red'}
    
    for case in ['PM_Only', 'VOC_Only', 'Integrated']:
        subset = df_model[df_model['Case'] == case]
        plt.plot(subset['Segment_Pct'], subset['Mean_RE'], marker='o', label=case, color=colors[case])
        
        # Annotate K
        for idx, row in subset.iterrows():
            plt.text(row['Segment_Pct'], row['Mean_RE'], f"k={row['Applied_K']}", 
                     fontsize=8, ha='right' if case == 'PM_Only' else 'left')

    plt.xlabel("Process Completion (%)")
    plt.ylabel("Mean Relative Error (RE)")
    plt.title(f"Fixed K Comparison (Based on VOC Best K) - {model_name}")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, f"Fixed_K_Comparison_{model_name}.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

