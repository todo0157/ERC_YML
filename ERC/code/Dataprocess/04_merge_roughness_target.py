import pandas as pd
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ERC
feature_dir = os.path.join(base_dir, "Enhanced_Features_v2")
quality_file = os.path.join(base_dir, "Printing_qualitydata.xlsx")
output_dir = os.path.join(base_dir, "Enhanced_Features_Roughness")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Starting Merge of Roughness Target...")

# 1. Load Quality Data (Target)
if not os.path.exists(quality_file):
    print(f"Error: Quality file not found at {quality_file}")
    exit()

df_quality = pd.read_excel(quality_file)
# Assume first column is ID and 'Roughness(nm)' is target
id_col = df_quality.columns[0]
target_col = 'Roughness(nm)'

if target_col not in df_quality.columns:
    print(f"Error: Target column '{target_col}' not found in quality file.")
    print(f"Columns found: {df_quality.columns}")
    exit()

# Clean ID
df_quality[id_col] = df_quality[id_col].astype(str).str.strip()
df_target = df_quality[[id_col, target_col]].rename(columns={id_col: 'Sample_ID', target_col: 'Target_Roughness'})
print(f"Loaded Quality Data: {len(df_target)} samples")
print(df_target.head())

# 2. Process Each Feature File
files = ["PM_Only.pkl", "VOC_Only.pkl", "Integrated.pkl"]

for fname in files:
    fpath = os.path.join(feature_dir, fname)
    if not os.path.exists(fpath):
        print(f"Skipping {fname} (not found)")
        continue
        
    print(f"\nProcessing {fname}...")
    try:
        df_feat = pd.read_pickle(fpath)
        
        # Check Sample_ID
        if 'Sample_ID' not in df_feat.columns:
            # Try to recover Sample_ID from index or other columns
            # In previous steps, Sample_ID was kept.
            print("  'Sample_ID' column missing. Checking index...")
            # If index is 'data1_10', we can extract 'data1'
            if isinstance(df_feat.index[0], str) and 'data' in df_feat.index[0]:
                df_feat['Sample_ID'] = [x.split('_')[0] for x in df_feat.index]
            else:
                print("  Cannot find Sample_ID. Skipping.")
                continue
                
        # Ensure ID format matches
        df_feat['Sample_ID'] = df_feat['Sample_ID'].astype(str).str.strip()
        
        # Merge with Target
        # Inner join to keep only samples with known roughness
        df_merged = pd.merge(df_feat, df_target, on='Sample_ID', how='inner')
        
        # Drop old target if exists
        if 'Target_PM_Mean' in df_merged.columns:
            df_merged.drop(columns=['Target_PM_Mean'], inplace=True)
            
        print(f"  Merged Shape: {df_merged.shape}")
        
        # Save to new folder
        save_path = os.path.join(output_dir, fname)
        df_merged.to_pickle(save_path)
        print(f"  Saved to {save_path}")
        
    except Exception as e:
        print(f"  Error processing {fname}: {e}")

print("\nMerge Completed. New files in 'Enhanced_Features_Roughness' folder.")

