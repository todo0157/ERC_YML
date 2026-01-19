import os
import glob
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# =========================================================
# 1. 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ERC 폴더 기준
PM_DIR = os.path.join(BASE_DIR, "PM_timeresampling")
VOC_DIR = os.path.join(BASE_DIR, "VOC")
OUT_XLSX = os.path.join(BASE_DIR, "results_features_p10to100_tsfresh.xlsx")

PCTS = list(range(10, 101, 10))

# 타겟 컬럼 정의
COL_PM_MAP = {"Num_0.3um": "Num_0.3um"}
COL_VOC = "VOC"

# =========================================================
# 2. 메인 로직
# =========================================================
def main():
    print(f"=== [Step 1] PM & VOC Basic Features Extraction ===")
    
    pm_files = sorted(glob.glob(os.path.join(PM_DIR, "*.xlsx")))
    if not pm_files:
        pm_files = sorted(glob.glob(os.path.join(PM_DIR, "*.xls")))
    
    long_data_list = []
    meta_info = [] # sample_id, percentage, unique_id 등 저장
    
    print(f"Loading {len(pm_files)} samples...")
    
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
            
            # Merge (Row by Row)
            min_len = min(len(df_pm), len(df_voc))
            df_full = pd.concat([df_pm.iloc[:min_len], df_voc.iloc[:min_len]], axis=1)
            
            for pct in PCTS:
                end_idx = max(5, int(min_len * (pct / 100.0)))
                df_slice = df_full.iloc[:end_idx].copy()
                df_slice['time'] = range(len(df_slice))
                
                unique_id = f"{sample_id}_{pct}"
                
                # PM & VOC 만 Melt
                value_vars = ['Num_0.3um', 'VOC']
                df_melt = df_slice.melt(id_vars=['time'], value_vars=value_vars, 
                                        var_name='kind', value_name='value')
                df_melt['id'] = unique_id
                
                long_data_list.append(df_melt)
                
                meta_info.append({
                    "unique_id": unique_id,
                    "sample_id": sample_id,
                    "percentage": pct,
                    "n_points": end_idx
                })
                
        except Exception as e:
            print(f"[WARN] {sample_id}: {e}")

    # --- Feature Extraction ---
    if not long_data_list:
        print("No data found.")
        return

    print(f"\n[tsfresh] Extracting features for PM & VOC...")
    df_long = pd.concat(long_data_list, ignore_index=True)
    
    extracted_features = extract_features(
        df_long, 
        column_id='id', 
        column_sort='time', 
        column_kind='kind', 
        column_value='value',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=0
    )
    extracted_features = extracted_features.fillna(0.0)
    print(f"[tsfresh] Shape: {extracted_features.shape}")
    
    # --- Split & Save ---
    final_pm = []
    final_voc = []
    final_int = [] # 초기에는 PM + VOC 피처만 가짐
    
    meta_df = pd.DataFrame(meta_info).set_index("unique_id")
    
    for unique_id in extracted_features.index:
        if unique_id not in meta_df.index: continue
        
        meta = meta_df.loc[unique_id].to_dict()
        row_base = meta.copy()
        feats = extracted_features.loc[unique_id].to_dict()
        
        # Split
        pm_feats = {k: v for k, v in feats.items() if "Num_0.3um" in k}
        voc_feats = {k: v for k, v in feats.items() if "VOC" in k}
        
        # 1. PM Only
        r_pm = row_base.copy()
        r_pm.update(pm_feats)
        final_pm.append(r_pm)
        
        # 2. VOC Only
        r_voc = row_base.copy()
        r_voc.update(voc_feats)
        final_voc.append(r_voc)
        
        # 3. Integrated (Initial: PM + VOC)
        r_int = row_base.copy()
        r_int.update(pm_feats)
        r_int.update(voc_feats)
        final_int.append(r_int)

    print(f"\nSaving to {OUT_XLSX}...")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(final_pm).to_excel(writer, index=False, sheet_name="PM_Only")
        pd.DataFrame(final_voc).to_excel(writer, index=False, sheet_name="VOC_Only")
        pd.DataFrame(final_int).to_excel(writer, index=False, sheet_name="Integrated")
        
    print("[SUCCESS] Step 1 Complete.")

if __name__ == "__main__":
    main()

