import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_PATH = os.path.join(BASE_DIR, "results_features_p10to100_tsfresh.xlsx")
TARGET_PATH = os.path.join(BASE_DIR, "Printing_qualitydata.xlsx")
OUT_DIR = os.path.join(BASE_DIR, "Comparative_Analysis_FixedK")
os.makedirs(OUT_DIR, exist_ok=True)

PCTS = list(range(10, 101, 10))
CASES = ["VOC_Only", "PM_Only", "Integrated"] # VOC 먼저 실행해야 함

# =========================================================
# 2. 모델 정의
# =========================================================
def get_models():
    return {
        "SVR": SVR(kernel='rbf', C=100, epsilon=0.1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

# =========================================================
# 3. 데이터 로드
# =========================================================
def load_all_datasets():
    print("Loading datasets...")
    datasets = {}
    
    if not os.path.exists(TARGET_PATH) or not os.path.exists(FEATURE_PATH):
        print("[ERROR] Files not found.")
        return {}

    df_target = pd.read_excel(TARGET_PATH)
    df_target.rename(columns={df_target.columns[0]: 'sample_id'}, inplace=True)
    df_target['sample_id'] = df_target['sample_id'].astype(str).str.strip()
    target_df = df_target[['sample_id', 'Roughness(nm)']]
    
    for case in CASES:
        try:
            df_feat = pd.read_excel(FEATURE_PATH, sheet_name=case)
            df_feat['sample_id'] = df_feat['sample_id'].astype(str).str.strip()
            df_merged = pd.merge(df_feat, target_df, on='sample_id', how='inner')
            datasets[case] = df_merged
            print(f"  [{case}] Loaded: {len(df_merged)} rows, {df_merged.shape[1]} cols")
        except Exception as e:
            print(f"  [ERROR] {case}: {e}")
            
    return datasets

# =========================================================
# 4. 분석 로직 (VOC 기준 K 고정)
# =========================================================
def run_analysis(datasets):
    # This function is deprecated/replaced by logic inside main()
    pass

def analyze_single_case(case_name, df_data, fixed_k_map=None):
    results = []
    feature_details = []
    
    drop_cols = ['sample_id', 'percentage', 'n_points', 'Roughness(nm)']
    feature_cols = [c for c in df_data.columns if c not in drop_cols]
    
    print(f"  Running {case_name}...")
    
    for pct in PCTS:
        df_sub = df_data[df_data['percentage'] == pct].copy()
        if len(df_sub) < 5: continue
        
        X_raw = df_sub[feature_cols].values
        y = df_sub['Roughness(nm)'].values
        
        # Preprocess
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_imputed = imputer.fit_transform(X_raw)
        
        # Remove constant
        from sklearn.feature_selection import VarianceThreshold
        try:
            sel = VarianceThreshold(threshold=0)
            X_imputed = sel.fit_transform(X_imputed)
            support = sel.get_support()
            curr_feats = [feature_cols[i] for i in range(len(feature_cols)) if support[i]]
        except:
            continue
            
        # 1. SelectKBest (Top 200 filtering)
        k_filter = min(200, X_imputed.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k_filter)
        try:
            X_sel = selector.fit_transform(X_imputed, y)
            sel_mask = selector.get_support()
            curr_feats = [curr_feats[i] for i in range(len(curr_feats)) if sel_mask[i]]
        except:
            continue
            
        X_sc = scaler.fit_transform(X_sel)
        
        # Models Loop
        models = get_models()
        for m_name, model in models.items():
            
            # Determine K to use
            if fixed_k_map is None:
                # VOC Case: Search Best K
                target_k_list = np.linspace(0, len(curr_feats), 9, dtype=int)[1:]
                target_k_list = sorted(list(set(target_k_list)))
                if 0 in target_k_list: target_k_list.remove(0)
            else:
                # Other Case: Use Fixed K
                if (pct, m_name) in fixed_k_map:
                    fixed_k = fixed_k_map[(pct, m_name)]
                    # 만약 현재 피처 개수보다 fixed_k가 크면 최대치로 조정
                    target_k_list = [min(fixed_k, len(curr_feats))]
                else:
                    target_k_list = [min(20, len(curr_feats))] # Fallback
            
            # 2. Permutation Importance
            try:
                model.fit(X_sc, y)
                r = permutation_importance(model, X_sc, y, n_repeats=3, random_state=42, n_jobs=1)
                ranks = sorted(zip(range(len(curr_feats)), r.importances_mean), key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, imp in ranks]
            except:
                top_indices = list(range(len(curr_feats)))
                
            # 3. CV Evaluation
            best_re = float('inf')
            best_k = -1
            
            loo = LeaveOneOut()
            
            for k in target_k_list:
                sel_idx = top_indices[:k]
                X_final = X_sc[:, sel_idx]
                
                re_vals = []
                for train_idx, test_idx in loo.split(X_final):
                    m_clone = get_models()[m_name]
                    m_clone.fit(X_final[train_idx], y[train_idx])
                    pred = max(0, m_clone.predict(X_final[test_idx])[0])
                    re = abs(y[test_idx][0] - pred) / (y[test_idx][0] + 1e-9)
                    re_vals.append(re)
                
                mean_re = np.mean(re_vals)
                if mean_re < best_re:
                    best_re = mean_re
                    best_k = k
                    
            results.append({
                "Case": case_name,
                "Model": m_name,
                "Percentage": pct,
                "Best_RE": best_re,
                "Best_K": best_k
            })
            
            # Save Feature Details (Top K features with importance)
            if best_k > 0:
                final_top_indices = top_indices[:best_k]
                for rank, idx in enumerate(final_top_indices, 1):
                    feat_name = curr_feats[idx]
                    # 중요도 값을 가져오기 위해 ranks 리스트 활용
                    # ranks는 (idx, importance) 튜플 리스트임
                    imp_val = 0.0
                    for r_idx, r_imp in ranks:
                        if r_idx == idx:
                            imp_val = r_imp
                            break
                    
                    feature_details.append({
                        "Case": case_name,
                        "Model": m_name,
                        "Percentage": pct,
                        "Rank": rank,
                        "Feature_Name": feat_name,
                        "Importance": imp_val
                    })
            
            # Log only for VOC search or final result
            if fixed_k_map is None:
                print(f"    [VOC] {pct}% {m_name}: Found Best K={best_k} (RE={best_re:.4f})")
            else:
                print(f"    [{case_name}] {pct}% {m_name}: Fixed K={best_k} -> RE={best_re:.4f}")

    return results, feature_details

# =========================================================
# 5. 실행 및 저장
# =========================================================
def main():
    datasets = load_all_datasets()
    if not datasets: return
    
    # Analyze all and collect results + details
    all_results = []
    all_feature_details = []
    
    # 1. Run Analysis
    # run_analysis 함수를 수정하여 feature_details도 반환받도록 처리해야 함
    # 하지만 run_analysis 함수 내부에서 analyze_single_case를 호출하므로 구조 수정 필요
    
    # 편의상 run_analysis 로직을 여기서 풀어서 작성
    voc_best_k_map = {} 
    
    print("\n>>> Phase 1: Analyzing VOC_Only to find Best K...")
    res_voc, det_voc = analyze_single_case("VOC_Only", datasets["VOC_Only"], fixed_k_map=None)
    all_results.extend(res_voc)
    all_feature_details.extend(det_voc)
    
    for res in res_voc:
        voc_best_k_map[(res['Percentage'], res['Model'])] = res['Best_K']
        
    print("\n>>> Phase 2: Analyzing PM & Integrated with Fixed K...")
    for case in ["PM_Only", "Integrated"]:
        if case in datasets:
            res, det = analyze_single_case(case, datasets[case], fixed_k_map=voc_best_k_map)
            all_results.extend(res)
            all_feature_details.extend(det)
            
    df_res = pd.DataFrame(all_results)
    df_det = pd.DataFrame(all_feature_details)
    
    # Save Excel
    summary = df_res.groupby(['Model', 'Case'])['Best_RE'].agg(['mean', 'std']).reset_index()
    summary.columns = ['Model', 'Case', 'Mean_Accuracy', 'Stability']
    summary = summary.sort_values(['Model', 'Mean_Accuracy'])
    
    out_file = os.path.join(OUT_DIR, "FixedK_Analysis_Result_WithFeatures.xlsx")
    with pd.ExcelWriter(out_file) as writer:
        df_res.to_excel(writer, sheet_name="Detail", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df_det.to_excel(writer, sheet_name="Feature_Details", index=False)
        
    print("\n[Summary]")
    print(summary)
    print(f"\n[Saved] Feature details to {out_file}")
    
    # Visualization (Existing Code)

    for model in ["SVR", "ExtraTrees"]:
        plt.figure(figsize=(10, 6))
        df_m = df_res[df_res['Model'] == model]
        if df_m.empty: continue
        
        colors = {"VOC_Only": "red", "PM_Only": "blue", "Integrated": "black"}
        markers = {"VOC_Only": "^", "PM_Only": "s", "Integrated": "o"}
        
        for case in CASES:
            df_c = df_m[df_m['Case'] == case]
            if df_c.empty: continue
            plt.plot(df_c['Percentage'], df_c['Best_RE'], 
                     label=case, color=colors[case], marker=markers[case])
            
            # Show K (Only for VOC to show baseline)
            if case == "VOC_Only":
                for _, row in df_c.iterrows():
                    plt.text(row['Percentage'], row['Best_RE'], f"K={int(row['Best_K'])}", 
                             fontsize=8, va='bottom')

        plt.title(f"[{model}] Fixed Feature Count (Based on VOC Best K)")
        plt.xlabel("Percentage")
        plt.ylabel("Mean RE")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"Fig_{model}_FixedK.png"))
        plt.close()

if __name__ == "__main__":
    main()

