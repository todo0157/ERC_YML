import os
import glob
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# =========================================================
# 1. 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ERC 폴더 기준
PM_DIR = os.path.join(BASE_DIR, "PM_timeresampling")
VOC_DIR = os.path.join(BASE_DIR, "VOC")
EXISTING_XLSX = os.path.join(BASE_DIR, "results_features_p10to100_tsfresh.xlsx")

PCTS = list(range(10, 101, 10))

COL_PM_MAP = {"Num_0.3um": "Num_0.3um"}
COL_VOC = "VOC"

# =========================================================
# 2. 메인 로직
# =========================================================
def main():
    print(f"=== [Step 2] Interaction (Prod, Ratio) Features Extraction ===")
    
    if not os.path.exists(EXISTING_XLSX):
        print(f"[ERROR] Step 1 File not found: {EXISTING_XLSX}")
        return
        
    print("Loading existing data...")
    try:
        df_pm_only = pd.read_excel(EXISTING_XLSX, sheet_name="PM_Only")
        df_voc_only = pd.read_excel(EXISTING_XLSX, sheet_name="VOC_Only")
        df_integrated = pd.read_excel(EXISTING_XLSX, sheet_name="Integrated")
        print(f"[Loaded] Integrated Shape: {df_integrated.shape}")
    except Exception as e:
        print(f"[ERROR] Excel load failed: {e}")
        return

    # --- Data Prep for Prod & Ratio ---
    pm_files = sorted(glob.glob(os.path.join(PM_DIR, "*.xlsx")))
    if not pm_files:
        pm_files = sorted(glob.glob(os.path.join(PM_DIR, "*.xls")))
    
    long_data_list = []
    
    print("Preparing interaction timeseries (Prod, Ratio)...")
    for pm_path in pm_files:
        filename = os.path.basename(pm_path)
        sample_id = filename.replace("_resampling", "").replace(".xlsx", "").replace(".xls", "")
        
        voc_path = os.path.join(VOC_DIR, f"{sample_id}.xlsx")
        if not os.path.exists(voc_path): voc_path = os.path.join(VOC_DIR, f"{sample_id}.xls")
        if not os.path.exists(voc_path): continue
            
        try:
            df_pm = pd.read_excel(pm_path) 
            df_voc = pd.read_excel(voc_path)
            
            # Rename
            renamed_pm = {}
            for raw, target in COL_PM_MAP.items():
                if raw in df_pm.columns: renamed_pm[raw] = target
            if not renamed_pm: continue
            df_pm = df_pm[list(renamed_pm.keys())].rename(columns=renamed_pm)
            
            voc_col = None
            for c in df_voc.columns:
                if "VOC" in str(c).upper(): voc_col = c; break
            if not voc_col: continue
            df_voc = df_voc[[voc_col]].rename(columns={voc_col: COL_VOC})
            
            min_len = min(len(df_pm), len(df_voc))
            df_full = pd.concat([df_pm.iloc[:min_len], df_voc.iloc[:min_len]], axis=1)
            
            for pct in PCTS:
                end_idx = max(5, int(min_len * (pct / 100.0)))
                df_slice = df_full.iloc[:end_idx].copy()
                df_slice['time'] = range(len(df_slice))
                
                unique_id = f"{sample_id}_{pct}"
                
                # --- Interaction Series ---
                # Prod
                df_slice['Prod_Num_0.3um_VOC'] = df_slice['Num_0.3um'] * df_slice['VOC']
                # Ratio
                df_slice['Ratio_Num_0.3um_VOC'] = df_slice['Num_0.3um'] / (df_slice['VOC'] + 1e-9)
                
                value_vars = ['Prod_Num_0.3um_VOC', 'Ratio_Num_0.3um_VOC']
                df_melt = df_slice.melt(id_vars=['time'], value_vars=value_vars, 
                                        var_name='kind', value_name='value')
                df_melt['id'] = unique_id
                long_data_list.append(df_melt)
                
        except Exception as e:
            print(f"[WARN] {sample_id}: {e}")

    # --- Feature Extraction ---
    if not long_data_list:
        print("No interaction data created.")
        return

    print(f"\n[tsfresh] Extracting features for Prod & Ratio...")
    df_long = pd.concat(long_data_list, ignore_index=True)
    
    new_features = extract_features(
        df_long, 
        column_id='id', 
        column_sort='time', 
        column_kind='kind', 
        column_value='value',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=0
    )
    new_features = new_features.fillna(0.0)
    print(f"[tsfresh] New Features Shape: {new_features.shape}")
    
    # --- Merge to Integrated ---
    print("Merging into Integrated sheet...")
    
    # Key Matching (unique_id = sample_id + "_" + percentage)
    df_integrated['unique_id_temp'] = df_integrated['sample_id'].astype(str) + "_" + df_integrated['percentage'].astype(str)
    new_features['unique_id_temp'] = new_features.index
    
    # 중복 컬럼 제외하고 병합
    cols_to_use = new_features.columns.difference(df_integrated.columns).tolist()
    if 'unique_id_temp' not in cols_to_use: cols_to_use.append('unique_id_temp')
    
    df_merged = pd.merge(df_integrated, new_features[cols_to_use], on='unique_id_temp', how='left')
    df_merged.drop(columns=['unique_id_temp'], inplace=True)
    df_merged.fillna(0.0, inplace=True)
    
    print(f"Updated Integrated Shape: {df_merged.shape}")
    
    # --- Save ---
    print(f"Saving to {EXISTING_XLSX}...")
    with pd.ExcelWriter(EXISTING_XLSX, engine="openpyxl") as writer:
        df_pm_only.to_excel(writer, index=False, sheet_name="PM_Only")
        df_voc_only.to_excel(writer, index=False, sheet_name="VOC_Only")
        df_merged.to_excel(writer, index=False, sheet_name="Integrated")
        
    print("[SUCCESS] Step 2 Complete.")

if __name__ == "__main__":
    main()

