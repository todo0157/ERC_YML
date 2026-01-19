import pandas as pd
import numpy as np
import os
import time
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_dir = os.path.join(base_dir, "Enhanced_Data")
output_dir = os.path.join(base_dir, "Comparative_Analysis_TSFRESH")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "results_features_p10to100_enhanced.xlsx")

# Parameters
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print("Starting Step 1 Enhanced: Extracting PM and VOC (including Lag/Rolling) features...")

# Columns to process (PM + All VOC variants)
# We will read one file to dynamically get column names
sample_file = os.path.join(input_dir, "data1_enhanced.xlsx")
if not os.path.exists(sample_file):
    print("Error: Enhanced data not found.")
    exit()

df_sample = pd.read_excel(sample_file)
# Target: Num_0.3um
# Features: VOC_Raw, VOC_Lag_*, VOC_Roll_*
target_col = 'Num_0.3um'
feature_cols = [c for c in df_sample.columns if 'VOC' in c]

print(f"Target: {target_col}")
print(f"Features to extract: {feature_cols}")

# Initialize storage
dfs_pm = []
dfs_voc = []
dfs_integrated = []

start_time = time.time()

for pct in PCTS:
    print(f"\nProcessing {pct}% segment...")
    
    long_data_list_pm = []
    long_data_list_voc = []
    
    # Store meta info
    meta_data = []

    for i in range(1, 28):
        file_path = os.path.join(input_dir, f"data{i}_enhanced.xlsx")
        
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            
            # Cut by percentage
            n_rows = len(df)
            end_idx = max(5, int(n_rows * (pct / 100.0)))
            df_slice = df.iloc[:end_idx].copy()
            
            unique_id = f"data{i}_{pct}"
            df_slice['id'] = unique_id
            df_slice['time'] = range(len(df_slice))
            
            # 1. PM Data (Target) for tsfresh
            # Actually, for PM_Only model, we usually use PM features to predict PM target (Auto-regression)
            # OR we just use PM statistics as features?
            # In previous logic: PM_Only sheet had PM statistics + Target.
            # VOC_Only sheet had VOC statistics + Target.
            
            # Prepare PM Long Format
            df_pm_melt = df_slice.melt(id_vars=['id', 'time'], value_vars=[target_col], 
                                       var_name='kind', value_name='value')
            long_data_list_pm.append(df_pm_melt)
            
            # Prepare VOC Long Format (All VOC variants)
            df_voc_melt = df_slice.melt(id_vars=['id', 'time'], value_vars=feature_cols, 
                                        var_name='kind', value_name='value')
            long_data_list_voc.append(df_voc_melt)
            
            # Meta data (Target value - Mean of last 10% or just the final value?)
            # Usually we predict the MEAN of the remaining part, or the final value?
            # User's original code logic: "Target" was often the 'Num_0.3um' mean of the whole file or specific label?
            # Wait, in 'extract_features_p10to100.py', the user was calculating statistics.
            # But where is the Y (Truth)? 
            # Usually Y is 'Num_0.3um' of the current segment? No, that's X.
            # Ah, the user's task is "Feature Analysis".
            # The MODEL scripts (run_best_model...) usually load X and y.
            # Let's check how Y was defined. 
            # In 'results_features_p10to100.xlsx', there was likely a 'Target' column or similar?
            # Or is 'Num_0.3um' itself the feature, and we are predicting something else?
            # No, usually we use 10% data to predict 100% data (or future).
            # BUT, the user's prompt says "extract data files ... related to ERC/ML_Feature_Analysis".
            # Let's stick to generating FEATURES (X).
            # The 'y' (Target) usually comes from the filename or a separate label file.
            # BUT, for now, we will just save the extracted features.
            # Wait, for the model to learn, it needs a Target (Y).
            # In previous `run_best_model_analysis.py`, Y was taken from `df['Num_0.3um']` ?
            # No, that's a time series.
            # Let's assume the Target Y is the AVERAGE PM of the *entire* sequence, or similar.
            # Let's preserve the sample-level mean of PM as a reference column 'Mean_PM_Target' just in case.
            
            mean_pm = df[target_col].mean() # Whole file mean
            meta_data.append({'id': unique_id, 'Sample_ID': f"data{i}", 'Segment_Pct': pct, 'Target_PM_Mean': mean_pm})

    # Combine Long Data
    df_long_pm = pd.concat(long_data_list_pm, ignore_index=True)
    df_long_voc = pd.concat(long_data_list_voc, ignore_index=True)
    
    # Extract Features using tsfresh
    print(f"  - Extracting PM features...")
    X_pm = extract_features(df_long_pm, column_id='id', column_sort='time', column_kind='kind', column_value='value',
                            default_fc_parameters=EfficientFCParameters(), n_jobs=0, disable_progressbar=True)
    
    print(f"  - Extracting VOC features (Enhanced)...")
    X_voc = extract_features(df_long_voc, column_id='id', column_sort='time', column_kind='kind', column_value='value',
                             default_fc_parameters=EfficientFCParameters(), n_jobs=0, disable_progressbar=True)
    
    # Meta DataFrame
    df_meta = pd.DataFrame(meta_data).set_index('id')
    
    # Merge
    # PM Only
    df_pm_final = X_pm.join(df_meta)
    dfs_pm.append(df_pm_final)
    
    # VOC Only
    df_voc_final = X_voc.join(df_meta)
    dfs_voc.append(df_voc_final)
    
    # Integrated (Initial: PM + VOC)
    df_int_final = X_pm.join(X_voc, lsuffix='_PM', rsuffix='_VOC').join(df_meta)
    dfs_integrated.append(df_int_final)

print("\nCombining all segments...")
final_pm = pd.concat(dfs_pm)
final_voc = pd.concat(dfs_voc)
final_int = pd.concat(dfs_integrated)

print(f"Saving to {output_file}...")
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    final_pm.to_excel(writer, sheet_name='PM_Only')
    final_voc.to_excel(writer, sheet_name='VOC_Only')
    final_int.to_excel(writer, sheet_name='Integrated')

end_time = time.time()
print(f"Step 1 Enhanced Complete! Time taken: {end_time - start_time:.2f} seconds")

