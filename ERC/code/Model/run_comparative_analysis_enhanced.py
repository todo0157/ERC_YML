import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_file = os.path.join(base_dir, "..", "results_features_p10to100_enhanced.xlsx") # In ERC/
output_dir = os.path.join(base_dir, "..", "Comparative_Analysis_Enhanced")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

print("Starting Comparative Analysis with Enhanced Features...")
print(f"Loading features from: {feature_file}")

try:
    df_pm_all = pd.read_excel(feature_file, sheet_name='PM_Only')
    df_voc_all = pd.read_excel(feature_file, sheet_name='VOC_Only')
    df_int_all = pd.read_excel(feature_file, sheet_name='Integrated')
except Exception as e:
    print(f"Error loading excel: {e}")
    exit()

def get_data_for_pct(df_all, pct):
    """Filter data by percentage segment."""
    # The 'Segment_Pct' column might not be preserved in 'id' index during tsfresh join?
    # Let's check columns.
    # In step1, we joined X_pm.join(df_meta). df_meta had 'Segment_Pct'.
    if 'Segment_Pct' in df_all.columns:
        return df_all[df_all['Segment_Pct'] == pct].copy()
    else:
        # Fallback: parse from 'id' index if it exists or reset index
        # step1 code: unique_id = f"data{i}_{pct}"
        # We need to rely on the sheet having the column.
        print("Warning: 'Segment_Pct' column not found.")
        return pd.DataFrame()

def train_evaluate(X, y, model_type='SVR'):
    """Train and evaluate model with Feature Selection."""
    
    # 0. Handle NaNs and Infs
    # Replace inf with nan first
    X = np.where(np.isinf(X), np.nan, X)
    
    # Impute missing values (mean strategy)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 1. Remove Constant Features
    sel_var = VarianceThreshold(threshold=0)
    X_var = sel_var.fit_transform(X_imputed)
    
    # 2. SelectKBest (Filter method) - Reduce to manageable size (e.g. 100)
    # If features < 100, keep all.
    n_features = X_var.shape[1]
    k_first = min(100, n_features)
    
    sel_kbest = SelectKBest(f_regression, k=k_first)
    X_kbest = sel_kbest.fit_transform(X_var, y)
    selected_indices = sel_var.get_support(indices=True)[sel_kbest.get_support(indices=True)]
    
    # 3. Model & CV
    if model_type == 'SVR':
        model = SVR(C=1.0, epsilon=0.1)
    else:
        model = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    
    re_scores = []
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kbest)
    
    for train_idx, test_idx in rkf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # RE
        y_test_safe = np.where(y_test == 0, 1e-6, y_test)
        re = np.abs((y_test - y_pred) / y_test_safe)
        re_scores.append(np.mean(re))
        
    return np.mean(re_scores), np.std(re_scores), k_first

# Main Loop
results_summary = []

print(f"{'Pct':<5} | {'Model':<12} | {'Dataset':<12} | {'Mean RE':<10} | {'Std RE':<10} | {'Feats':<5}")
print("-" * 75)

for pct in PCTS:
    # Prepare Data
    data_pm = get_data_for_pct(df_pm_all, pct)
    data_voc = get_data_for_pct(df_voc_all, pct)
    data_int = get_data_for_pct(df_int_all, pct)
    
    if data_pm.empty:
        continue
        
    # Target (Assume same for all, taken from PM dataset meta)
    y = data_pm['Target_PM_Mean'].values
    
    # Feature Matrices (Exclude Meta Columns)
    meta_cols = ['Segment_Pct', 'Target_PM_Mean', 'Sample_ID']
    
    X_pm = data_pm.drop(columns=[c for c in meta_cols if c in data_pm.columns]).select_dtypes(include=[np.number]).values
    X_voc = data_voc.drop(columns=[c for c in meta_cols if c in data_voc.columns]).select_dtypes(include=[np.number]).values
    X_int = data_int.drop(columns=[c for c in meta_cols if c in data_int.columns]).select_dtypes(include=[np.number]).values
    
    # Evaluate for each Model and Dataset
    datasets = {'PM_Only': X_pm, 'VOC_Only': X_voc, 'Integrated': X_int}
    models = ['SVR', 'ExtraTrees']
    
    for m_name in models:
        for d_name, X_data in datasets.items():
            mean_re, std_re, n_feats = train_evaluate(X_data, y, m_name)
            
            results_summary.append({
                'Percentage': pct,
                'Model': m_name,
                'Dataset': d_name,
                'Mean_RE': mean_re,
                'Std_RE': std_re,
                'N_Features': n_feats
            })
            
            print(f"{pct:<5} | {m_name:<12} | {d_name:<12} | {mean_re:.4f}     | {std_re:.4f}     | {n_feats}")

# Save Results
df_res = pd.DataFrame(results_summary)
res_file = os.path.join(output_dir, "Enhanced_Analysis_Results.xlsx")
df_res.to_excel(res_file, index=False)
print(f"\nAnalysis Saved to {res_file}")

# Plotting Best RE Comparison
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_res, x='Percentage', y='Mean_RE', hue='Dataset', style='Model', markers=True, dashes=False)
plt.title('Comparison of RE by Dataset and Model (Enhanced Features)')
plt.ylabel('Mean Relative Error (RE)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "Enhanced_RE_Comparison.png"))
print("Plot saved.")

