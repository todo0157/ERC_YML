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
from sklearn.impute import SimpleImputer

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_dir = os.path.join(base_dir, "..", "Enhanced_Features_Roughness")
output_dir = os.path.join(base_dir, "..", "Comparative_Analysis_Roughness")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RANDOM_STATE = 42

# Load Best K info (We need to know which K was used)
# Assuming we use the "Best K" found in previous step.
# For simplicity, we will re-calculate Ranking and pick Top 20 (or user specific K).
# Let's use K=20 for consistency as it was mostly best or sufficient.
TOP_K = 20

print("Starting Top Feature Analysis for Roughness...")

# Load Datasets
try:
    df_pm_all = pd.read_pickle(os.path.join(feature_dir, "PM_Only.pkl"))
    df_voc_all = pd.read_pickle(os.path.join(feature_dir, "VOC_Only.pkl"))
    df_int_all = pd.read_pickle(os.path.join(feature_dir, "Integrated.pkl"))
except Exception as e:
    print(f"Error loading pickle files: {e}")
    exit()

def get_data_for_pct(df_all, pct):
    if 'Segment_Pct' in df_all.columns:
        return df_all[df_all['Segment_Pct'] == pct].copy()
    return pd.DataFrame()

def classify_feature(name):
    """Classify feature name into types."""
    if 'Prod_' in name or 'Ratio_' in name or 'Diff_' in name or 'Sum_' in name:
        return 'Interaction'
    elif 'Lag' in name or 'Roll' in name:
        return 'VOC_Enhanced'
    elif 'VOC' in name:
        return 'VOC_Raw'
    elif 'Num_0.3um' in name:
        return 'PM_Raw'
    else:
        return 'Other'

def analyze_features(X, y, feature_names, model_type='SVR'):
    """Rank and return top features."""
    
    # Preprocess
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    sel_var = VarianceThreshold(threshold=0)
    X_var = sel_var.fit_transform(X_imputed)
    remaining_indices = sel_var.get_support(indices=True)
    curr_feat_names = np.array(feature_names)[remaining_indices]
    
    # SelectKBest (Top 100)
    k_first = min(100, X_var.shape[1])
    sel_kbest = SelectKBest(f_regression, k=k_first)
    X_kbest = sel_kbest.fit_transform(X_var, y)
    selected_mask = sel_kbest.get_support()
    final_feat_names = curr_feat_names[selected_mask]
    
    # Scale & Model
    if model_type == 'SVR':
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_kbest)
        y_log = np.log1p(y)
        
        model = SVR(C=10, epsilon=0.05, gamma=0.01) # Roughness Tuned
        model.fit(X_scaled, y_log)
        r = permutation_importance(model, X_scaled, y_log, n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
        importances = r.importances_mean
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_kbest)
        
        model = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_scaled, y)
        importances = model.feature_importances_
        
    # Rank
    df_imp = pd.DataFrame({
        'Feature': final_feat_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Add Type
    df_imp['Type'] = df_imp['Feature'].apply(classify_feature)
    
    return df_imp

all_top_features = []
type_stats = []

for pct in PCTS:
    print(f"\nAnalyzing {pct}% Segment...")
    
    # Prepare Data dict
    datasets = {
        'PM_Only': get_data_for_pct(df_pm_all, pct),
        'VOC_Only': get_data_for_pct(df_voc_all, pct),
        'Integrated': get_data_for_pct(df_int_all, pct)
    }
    
    for case_name, data in datasets.items():
        if data.empty: continue
        
        y = data['Target_Roughness'].values
        meta_cols = ['Segment_Pct', 'Target_Roughness', 'Sample_ID']
        X_df = data.drop(columns=[c for c in meta_cols if c in data.columns]).select_dtypes(include=[np.number])
        
        for model_name in ['SVR', 'ExtraTrees']:
            # Analyze
            df_ranked = analyze_features(X_df.values, y, X_df.columns, model_name)
            
            # Take Top K
            df_top = df_ranked.head(TOP_K).copy()
            df_top['Segment_Pct'] = pct
            df_top['Case'] = case_name
            df_top['Model'] = model_name
            df_top['Rank'] = range(1, len(df_top) + 1)
            
            all_top_features.append(df_top)
            
            # Calculate Type Stats (for Integrated mainly, but good for all)
            counts = df_top['Type'].value_counts()
            total = len(df_top)
            
            stats = {'Segment_Pct': pct, 'Case': case_name, 'Model': model_name}
            for t in ['PM_Raw', 'VOC_Raw', 'VOC_Enhanced', 'Interaction']:
                stats[t] = counts.get(t, 0)
            stats['Interaction_Ratio'] = stats['Interaction'] / total
            stats['Enhanced_Ratio'] = stats['VOC_Enhanced'] / total
            type_stats.append(stats)

# Save Results
df_all_top = pd.concat(all_top_features, ignore_index=True)
df_stats = pd.DataFrame(type_stats)

excel_path = os.path.join(output_dir, "Top_Features_Roughness.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    df_all_top.to_excel(writer, sheet_name="Top_Features", index=False)
    df_stats.to_excel(writer, sheet_name="Type_Statistics", index=False)

print(f"Saved Top Features to {excel_path}")

# Plotting Interaction Ratio (Integrated Only)
df_stats_int = df_stats[df_stats['Case'] == 'Integrated']

for model_name in ['SVR', 'ExtraTrees']:
    subset = df_stats_int[df_stats_int['Model'] == model_name]
    
    plt.figure(figsize=(10, 6))
    plt.stackplot(subset['Segment_Pct'], 
                  subset['PM_Raw'], subset['VOC_Raw'], subset['VOC_Enhanced'], subset['Interaction'],
                  labels=['PM_Raw', 'VOC_Raw', 'VOC_Enhanced', 'Interaction'],
                  colors=['blue', 'green', 'orange', 'red'], alpha=0.6)
    
    plt.xlabel("Process Completion (%)")
    plt.ylabel("Number of Features (Top 20)")
    plt.title(f"Feature Type Distribution in Top 20 (Integrated, {model_name})")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"Feature_Distribution_{model_name}.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

