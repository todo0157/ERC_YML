import pandas as pd
import numpy as np
import os
import time
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_dir = os.path.join(base_dir, "Enhanced_Data")
input_file = os.path.join(base_dir, "results_features_p10to100_enhanced.xlsx")

# Output Directory for Pickle Files
output_dir = os.path.join(base_dir, "Enhanced_Features_v2")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print("Starting Step 2 Enhanced (Pickle Version): Extracting Interaction (Prod, Ratio) features...")
print(f"Reading base features from: {input_file}")

try:
    # Read Excel once
    df_pm_base = pd.read_excel(input_file, sheet_name='PM_Only', index_col=0) 
    df_voc_base = pd.read_excel(input_file, sheet_name='VOC_Only', index_col=0)
    df_int_base = pd.read_excel(input_file, sheet_name='Integrated', index_col=0)
    
    # Save base files as pickle immediately to ensure consistency
    df_pm_base.to_pickle(os.path.join(output_dir, "PM_Only.pkl"))
    df_voc_base.to_pickle(os.path.join(output_dir, "VOC_Only.pkl"))
    print("Base PM/VOC files saved as Pickle.")
    
except Exception as e:
    print(f"Error loading base file: {e}")
    exit()

# Get enhanced VOC column names
sample_file = os.path.join(input_dir, "data1_enhanced.xlsx")
if not os.path.exists(sample_file):
    print("Error: Enhanced data not found.")
    exit()
df_sample = pd.read_excel(sample_file)
voc_cols = [c for c in df_sample.columns if 'VOC' in c]
target_col = 'Num_0.3um'

print(f"Interaction pairs: PM ({target_col}) x VOC Features ({len(voc_cols)} cols)")

start_time = time.time()
new_features_list = []

for pct in PCTS:
    print(f"\nProcessing {pct}% segment interactions...")
    long_data_list = []
    
    for i in range(1, 28):
        file_path = os.path.join(input_dir, f"data{i}_enhanced.xlsx")
        
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            
            end_idx = max(5, int(len(df) * (pct / 100.0)))
            df_slice = df.iloc[:end_idx].copy()
            unique_id = f"data{i}_{pct}"
            df_slice['id'] = unique_id
            df_slice['time'] = range(len(df_slice))
            
            # Generate Interaction Series
            pm_series = df_slice[target_col]
            
            for v_col in voc_cols:
                voc_series = df_slice[v_col]
                
                # Product
                prod_series = pm_series * voc_series
                df_slice[f"Prod_{v_col}"] = prod_series
                
                # Ratio (Handle div by zero)
                voc_safe = voc_series.replace(0, 1e-6)
                ratio_series = pm_series / voc_safe
                df_slice[f"Ratio_{v_col}"] = ratio_series
            
            # Melt only the NEW interaction columns
            new_cols = [c for c in df_slice.columns if c.startswith('Prod_') or c.startswith('Ratio_')]
            
            df_melt = df_slice.melt(id_vars=['id', 'time'], value_vars=new_cols, 
                                    var_name='kind', value_name='value')
            long_data_list.append(df_melt)
    
    # Concatenate & Extract
    if long_data_list:
        df_long = pd.concat(long_data_list, ignore_index=True)
        print(f"  - Extracting tsfresh features for interactions...")
        X_int_new = extract_features(df_long, column_id='id', column_sort='time', column_kind='kind', column_value='value',
                                     default_fc_parameters=EfficientFCParameters(), n_jobs=0, disable_progressbar=True)
        new_features_list.append(X_int_new)

print("\nCombining all new interaction features...")
if new_features_list:
    final_new_features = pd.concat(new_features_list)
    
    # Handle duplicate columns if any (tsfresh sometimes generates duplicates?)
    final_new_features = final_new_features.loc[:, ~final_new_features.columns.duplicated()]

    print("Merging with existing Integrated data...")
    # Merge based on index (id)
    df_int_updated = df_int_base.join(final_new_features, how='left')
    
    # Save Integrated as Pickle
    print(f"Saving Integrated.pkl to {output_dir}...")
    df_int_updated.to_pickle(os.path.join(output_dir, "Integrated.pkl"))
else:
    print("No new features extracted.")

end_time = time.time()
print(f"Step 2 Enhanced (Pickle) Complete! Time taken: {end_time - start_time:.2f} seconds")
