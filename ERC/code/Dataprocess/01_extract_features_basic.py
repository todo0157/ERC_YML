import os
import glob
import numpy as np
import pandas as pd
from itertools import combinations
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

# 타겟 컬럼 정의 (Num_0.3um, VOC 만 사용)
COL_PM_MAP = {
    "Num_0.3um": "Num_0.3um",
}
COL_VOC = "VOC"

# 상호작용 계산용 리스트
INTERACTION_COLS = ["Num_0.3um", "VOC"]

# =========================================================
# 2. 상호작용 피처 계산 함수 (Manual - Scalar)
# =========================================================
def calculate_interaction_features(df):
    interaction_feats = {}
    
    # Num_0.3um vs VOC 하나뿐이지만 확장성을 위해 loop 유지
    avail_cols = [c for c in INTERACTION_COLS if c in df.columns]
    
    for col1, col2 in combinations(avail_cols, 2):
        # 상관계수
        try:
            val = df[col1].corr(df[col2])
            interaction_feats[f"Corr_{col1}_{col2}"] = float(val) if not np.isnan(val) else 0.0
        except:
            interaction_feats[f"Corr_{col1}_{col2}"] = 0.0
            
        # 비율 평균
        try:
            ratio = df[col1] / (df[col2] + 1e-9)
            interaction_feats[f"RatioMean_{col1}_{col2}"] = float(ratio.mean())
        except:
            interaction_feats[f"RatioMean_{col1}_{col2}"] = 0.0
            
        # 곱의 평균 (Interaction Strength)
        try:
            prod = df[col1] * df[col2]
            interaction_feats[f"ProdMean_{col1}_{col2}"] = float(prod.mean())
        except:
            interaction_feats[f"ProdMean_{col1}_{col2}"] = 0.0

    return interaction_feats

# =========================================================
# 3. 메인 로직
# =========================================================
def main():
    print(f"PM Directory: {PM_DIR}")
    print(f"VOC Directory: {VOC_DIR}")
    
    pm_files = sorted(glob.glob(os.path.join(PM_DIR, "*.xlsx")))
    if not pm_files:
        pm_files = sorted(glob.glob(os.path.join(PM_DIR, "*.xls")))
    
    # tsfresh용 Long Format 데이터 수집 리스트
    # 구조: [id, time, variable, value]
    long_data_list = []
    
    # 메타 정보 (sample_id -> pct -> n_points)
    meta_info = []
    
    # 상호작용 피처 저장소 (sample_id_pct -> dict)
    interaction_store = {}
    
    # 샘플 처리
    for pm_path in pm_files:
        filename = os.path.basename(pm_path)
        sample_id = filename.replace("_resampling", "").replace(".xlsx", "").replace(".xls", "")
        
        voc_path = os.path.join(VOC_DIR, f"{sample_id}.xlsx")
        if not os.path.exists(voc_path):
            voc_path = os.path.join(VOC_DIR, f"{sample_id}.xls")
        
        if not os.path.exists(voc_path):
            print(f"[SKIP] VOC not found: {sample_id}")
            continue
            
        try:
            df_pm = pd.read_excel(pm_path) 
            df_voc = pd.read_excel(voc_path)
            
            # Rename
            renamed_pm = {}
            for raw, target in COL_PM_MAP.items():
                if raw in df_pm.columns: renamed_pm[raw] = target
            if not renamed_pm: raise ValueError("No valid PM col")
            df_pm = df_pm[list(renamed_pm.keys())].rename(columns=renamed_pm)
            
            voc_col = None
            for c in df_voc.columns:
                if "VOC" in str(c).upper(): voc_col = c; break
            if not voc_col: raise ValueError("No valid VOC col")
            df_voc = df_voc[[voc_col]].rename(columns={voc_col: COL_VOC})
            
            # Merge
            min_len = min(len(df_pm), len(df_voc))
            df_full = pd.concat([df_pm.iloc[:min_len], df_voc.iloc[:min_len]], axis=1)
            
            # 10% 단위로 데이터 자르고 Long Format으로 변환
            for pct in PCTS:
                end_idx = max(5, int(min_len * (pct / 100.0)))
                df_slice = df_full.iloc[:end_idx].copy()
                df_slice['time'] = range(len(df_slice))
                
                unique_id = f"{sample_id}_{pct}" # 임시 ID (tsfresh용)
                
                # 1. Manual 상호작용 피처 계산 (Scalar)
                int_feats = calculate_interaction_features(df_slice)
                interaction_store[unique_id] = int_feats
                
                # 2. 파생 시계열 생성 (Series) -> tsfresh 입력용
                # Product Series
                df_slice['Prod_Num_0.3um_VOC'] = df_slice['Num_0.3um'] * df_slice['VOC']
                # Ratio Series (0 division 방지)
                df_slice['Ratio_Num_0.3um_VOC'] = df_slice['Num_0.3um'] / (df_slice['VOC'] + 1e-9)
                # Diff Series (Num - VOC)
                df_slice['Diff_Num_0.3um_VOC'] = df_slice['Num_0.3um'] - df_slice['VOC']
                # Sum Series (Num + VOC)
                df_slice['Sum_Num_0.3um_VOC'] = df_slice['Num_0.3um'] + df_slice['VOC']
                
                # 3. Long Format 변환 (Melt)
                # 대상 컬럼: Num_0.3um, VOC, Prod_..., Ratio_..., Diff_..., Sum_...
                value_vars = ['Num_0.3um', 'VOC', 'Prod_Num_0.3um_VOC', 'Ratio_Num_0.3um_VOC',
                              'Diff_Num_0.3um_VOC', 'Sum_Num_0.3um_VOC']
                
                df_melt = df_slice.melt(id_vars=['time'], value_vars=value_vars, 
                                        var_name='kind', value_name='value')
                df_melt['id'] = unique_id
                
                long_data_list.append(df_melt)
                
                # 메타 정보 저장
                meta_info.append({
                    "unique_id": unique_id,
                    "sample_id": sample_id,
                    "percentage": pct,
                    "n_points": end_idx
                })
                
            print(f"[Loaded] {sample_id}")
            
        except Exception as e:
            print(f"[FAIL] {sample_id}: {e}")

    # --- tsfresh Feature Extraction ---
    if not long_data_list:
        print("No data to process.")
        return

    print("\n[tsfresh] Starting feature extraction (Efficient)... This may take a while.")
    df_long = pd.concat(long_data_list, ignore_index=True)
    
    # extract_features
    extracted_features = extract_features(
        df_long, 
        column_id='id', 
        column_sort='time', 
        column_kind='kind', 
        column_value='value',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=0 # 병렬처리 (0=All CPUs)
    )
    
    print(f"\n[tsfresh] Extraction Complete. Shape: {extracted_features.shape}")
    
    # --- Post Processing & Splitting ---
    extracted_features = extracted_features.fillna(0.0)
    
    # 데이터 정리
    final_pm = []
    final_voc = []
    final_int = []
    
    meta_df = pd.DataFrame(meta_info).set_index("unique_id")
    
    # extracted_features의 인덱스는 unique_id
    for unique_id in extracted_features.index:
        if unique_id not in meta_df.index: continue
        
        # Meta
        meta = meta_df.loc[unique_id].to_dict()
        row_base = meta.copy()
        
        # Features (dict로 변환)
        feats = extracted_features.loc[unique_id].to_dict()
        
        # 분리 로직
        # 상호작용 키워드
        int_keywords = ["Prod", "Ratio", "Diff", "Sum"]
        
        # 1. PM Only: 이름에 "Num_0.3um" 포함 AND 상호작용 키워드 미포함
        pm_feats = {k: v for k, v in feats.items() 
                   if "Num_0.3um" in k and not any(x in k for x in int_keywords)}
        
        # 2. VOC Only: 이름에 "VOC" 포함 AND 상호작용 키워드 미포함
        voc_feats = {k: v for k, v in feats.items() 
                    if "VOC" in k and not any(x in k for x in int_keywords)}
        
        # 3. Interaction (tsfresh derived)
        # 이름에 상호작용 키워드 포함
        int_tsfresh_feats = {k: v for k, v in feats.items() 
                            if any(x in k for x in int_keywords)}
        
        # 4. Interaction (Manual Scalar)
        int_manual = interaction_store.get(unique_id, {})
        
        # --- 시트별 저장 데이터 구성 ---
        
        # 1. PM Only Sheet
        r_pm = row_base.copy()
        r_pm.update(pm_feats)
        final_pm.append(r_pm)
        
        # 2. VOC Only Sheet
        r_voc = row_base.copy()
        r_voc.update(voc_feats)
        final_voc.append(r_voc)
        
        # 3. Integrated Sheet (PM + VOC + Int_tsfresh + Int_manual)
        r_int = row_base.copy()
        r_int.update(pm_feats)
        r_int.update(voc_feats)
        r_int.update(int_tsfresh_feats)
        r_int.update(int_manual)
        final_int.append(r_int)

    # Save
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(final_pm).to_excel(writer, index=False, sheet_name="PM_Only")
        pd.DataFrame(final_voc).to_excel(writer, index=False, sheet_name="VOC_Only")
        pd.DataFrame(final_int).to_excel(writer, index=False, sheet_name="Integrated")
        
    print(f"\n[SUCCESS] Saved to {OUT_XLSX}")
    # 개수 확인용 샘플
    print(f"Features Count Per Sample:")
    print(f"  PM Only: {len(pm_feats)}")
    print(f"  VOC Only: {len(voc_feats)}")
    print(f"  Interaction (tsfresh): {len(int_tsfresh_feats)}")
    print(f"  Interaction (Manual): {len(int_manual)}")
    print(f"  Total Integrated: {len(r_int) - len(row_base)}") # 메타 제외

if __name__ == "__main__":
    main()
