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
import matplotlib.ticker as mtick

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_dir = os.path.join(base_dir, "..", "Enhanced_Features_Roughness")
output_dir = os.path.join(base_dir, "..", "Comparative_Analysis_Roughness")
result_file = os.path.join(output_dir, "Fixed_K_Roughness_Result.xlsx")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RANDOM_STATE = 42

print("Starting Dynamic Top K Feature Analysis...")

# 1. Load Best K Info
if not os.path.exists(result_file):
    print("Error: Fixed_K_Roughness_Result.xlsx not found. Please run fixed k analysis first.")
    exit()

df_results = pd.read_excel(result_file)
# Filter for Integrated case only (we want to see composition in Integrated model)
df_k_info = df_results[df_results['Case'] == 'Integrated'][['Model', 'Segment_Pct', 'Applied_K']]
print("Loaded Best K info:")
print(df_k_info.head())

# 2. Load Datasets
try:
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

def analyze_features_dynamic(X, y, feature_names, model_type='SVR', top_k=20):
    """Rank and return top K features."""
    
    # Preprocess
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    sel_var = VarianceThreshold(threshold=0)
    X_var = sel_var.fit_transform(X_imputed)
    remaining_indices = sel_var.get_support(indices=True)
    curr_feat_names = np.array(feature_names)[remaining_indices]
    
    # SelectKBest (Top 100 max, but should be at least top_k)
    # If top_k > 100, we need to select more.
    k_first = max(100, top_k)
    k_first = min(k_first, X_var.shape[1])
    
    sel_kbest = SelectKBest(f_regression, k=k_first)
    X_kbest = sel_kbest.fit_transform(X_var, y)
    selected_mask = sel_kbest.get_support()
    final_feat_names = curr_feat_names[selected_mask]
    
    # Scale & Model
    if model_type == 'SVR':
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_kbest)
        y_log = np.log1p(y)
        
        model = SVR(C=10, epsilon=0.05, gamma=0.01) # Tuned
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
    
    # Cut Top K
    return df_imp.head(top_k)

type_stats = []
all_top_features = []

for pct in PCTS:
    print(f"\nAnalyzing {pct}% Segment (Integrated)...")
    data = get_data_for_pct(df_int_all, pct)
    
    if data.empty: continue
    
    y = data['Target_Roughness'].values
    meta_cols = ['Segment_Pct', 'Target_Roughness', 'Sample_ID']
    X_df = data.drop(columns=[c for c in meta_cols if c in data.columns]).select_dtypes(include=[np.number])
    
    for model_name in ['SVR', 'ExtraTrees']:
        # Get Best K for this condition
        k_row = df_k_info[(df_k_info['Segment_Pct'] == pct) & (df_k_info['Model'] == model_name)]
        if k_row.empty:
            current_k = 20 # Fallback
        else:
            current_k = int(k_row['Applied_K'].values[0])
            
        print(f"  {model_name}: Using Best K={current_k}")
        
        # Analyze
        df_top = analyze_features_dynamic(X_df.values, y, X_df.columns, model_name, top_k=current_k)
        
        df_top['Segment_Pct'] = pct
        df_top['Case'] = 'Integrated'
        df_top['Model'] = model_name
        df_top['Rank'] = range(1, len(df_top) + 1)
        all_top_features.append(df_top)
        
        # Calculate Stats (Counts & Ratios)
        counts = df_top['Type'].value_counts()
        total = len(df_top) # Should be equal to current_k
        
        stats = {'Segment_Pct': pct, 'Model': model_name, 'Applied_K': current_k}
        for t in ['PM_Raw', 'VOC_Raw', 'VOC_Enhanced', 'Interaction']:
            cnt = counts.get(t, 0)
            stats[t] = cnt
            stats[f"{t}_Ratio"] = cnt / total
            
        type_stats.append(stats)

# Save Stats
df_stats = pd.DataFrame(type_stats)
df_all_top = pd.concat(all_top_features, ignore_index=True)

excel_path = os.path.join(output_dir, "Top_Features_Roughness_DynamicK.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    df_stats.to_excel(writer, sheet_name="Type_Statistics", index=False)
    df_all_top.to_excel(writer, sheet_name="Top_Features_List", index=False)
print(f"Saved stats to {excel_path}")

# Plotting (100% Stacked Area/Bar)
for model_name in ['SVR', 'ExtraTrees']:
    subset = df_stats[df_stats['Model'] == model_name]
    
    plt.figure(figsize=(12, 6))
    
    # Prepare data for stackplot
    x = subset['Segment_Pct']
    y1 = subset['PM_Raw_Ratio'] * 100
    y2 = subset['VOC_Raw_Ratio'] * 100
    y3 = subset['VOC_Enhanced_Ratio'] * 100
    y4 = subset['Interaction_Ratio'] * 100
    
    plt.stackplot(x, y1, y2, y3, y4,
                  labels=['PM_Raw', 'VOC_Raw', 'VOC_Enhanced', 'Interaction'],
                  colors=['#4E79A7', '#59A14F', '#F28E2B', '#E15759'], # Tableau Colors
                  alpha=0.8)
    
    # Annotate K values on top
    for idx, row in subset.iterrows():
        plt.text(row['Segment_Pct'], 102, f"K={row['Applied_K']}", 
                 ha='center', fontsize=8, color='black')

    plt.xlabel("Process Completion (%)")
    plt.ylabel("Feature Share (%)")
    plt.title(f"Feature Contribution by Type (Integrated, {model_name}) - Dynamic Top K")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim(0, 110) # Room for K labels
    plt.xlim(10, 100)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(True, axis='x', linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"Feature_Distribution_DynamicK_{model_name}.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

