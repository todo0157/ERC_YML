import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_PATH = os.path.join(BASE_DIR, "results_features_p10to100.xlsx")
TARGET_PATH = os.path.join(BASE_DIR, "Printing_qualitydata.xlsx")
OUT_DIR = os.path.join(BASE_DIR, "Comparative_Analysis")
os.makedirs(OUT_DIR, exist_ok=True)

PCTS = list(range(10, 101, 10))
CASES = ["Integrated", "PM_Only", "VOC_Only"]

# =========================================================
# 2. 모델 정의
# =========================================================
def get_models():
    return {
        "SVR": SVR(kernel='rbf', C=100, epsilon=0.1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

# =========================================================
# 3. 데이터 로드 (3개 시트)
# =========================================================
def load_all_datasets():
    print("Loading datasets from Excel...")
    datasets = {}
    
    # Label Load
    df_target = pd.read_excel(TARGET_PATH)
    df_target.rename(columns={df_target.columns[0]: 'sample_id'}, inplace=True)
    df_target['sample_id'] = df_target['sample_id'].astype(str).str.strip()
    target_df = df_target[['sample_id', 'Roughness(nm)']]
    
    # Feature Sheets Load
    for case in CASES:
        try:
            df_feat = pd.read_excel(FEATURE_PATH, sheet_name=case)
            df_feat['sample_id'] = df_feat['sample_id'].astype(str).str.strip()
            
            # Merge
            df_merged = pd.merge(df_feat, target_df, on='sample_id', how='inner')
            datasets[case] = df_merged
            print(f"  [{case}] Loaded: {len(df_merged)} rows, {df_merged.shape[1]} cols")
            
        except Exception as e:
            print(f"  [ERROR] Failed to load sheet '{case}': {e}")
            
    return datasets

# =========================================================
# 4. 분석 로직 (Best RE 탐색)
# =========================================================
def analyze_case(case_name, df_data):
    results = []
    
    # 피처 컬럼 식별
    drop_cols = ['sample_id', 'percentage', 'n_points', 'Roughness(nm)']
    feature_cols = [c for c in df_data.columns if c not in drop_cols]
    total_features = len(feature_cols)
    
    # 동적 구간 (5 steps)
    if total_features < 5:
        feature_steps = list(range(1, total_features + 1))
    else:
        feature_steps = np.linspace(0, total_features, 6, dtype=int)[1:]
        feature_steps = sorted(list(set(feature_steps)))
        if 0 in feature_steps: feature_steps.remove(0)
    
    print(f"\n>>> Analyzing [{case_name}] (Features: {total_features})")
    
    # Loop Percentage
    for pct in PCTS:
        df_sub = df_data[df_data['percentage'] == pct].copy()
        if len(df_sub) < 5: continue
        
        X_raw = df_sub[feature_cols].values
        y = df_sub['Roughness(nm)'].values
        
        # Preprocess
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(imputer.fit_transform(X_raw))
        
        # Models Loop
        models = get_models()
        for m_name, model in models.items():
            # 1. Importance
            try:
                model.fit(X_sc, y)
                r = permutation_importance(model, X_sc, y, n_repeats=5, random_state=42, n_jobs=1)
                importances = r.importances_mean
                ranks = sorted(zip(range(len(feature_cols)), importances), key=lambda x: x[1], reverse=True)
                top_indices_full = [idx for idx, imp in ranks]
            except:
                top_indices_full = list(range(len(feature_cols)))
            
            # 2. Find Best K
            best_re = float('inf')
            best_k = -1
            
            loo = LeaveOneOut()
            
            for k in feature_steps:
                k_real = min(k, len(feature_cols))
                sel_idx = top_indices_full[:k_real]
                X_sel = X_sc[:, sel_idx]
                
                re_list = []
                for train_idx, test_idx in loo.split(X_sel):
                    X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]
                    
                    m_clone = get_models()[m_name]
                    m_clone.fit(X_tr, y_tr)
                    pred = max(0, m_clone.predict(X_te)[0])
                    
                    re_val = abs(y_te[0] - pred) / (y_te[0] + 1e-9)
                    re_list.append(re_val)
                
                mean_re = np.mean(re_list)
                if mean_re < best_re:
                    best_re = mean_re
                    best_k = k_real
            
            results.append({
                "Case": case_name,
                "Model": m_name,
                "Percentage": pct,
                "Best_RE": best_re,
                "Best_K": best_k
            })
            
    return results

# =========================================================
# 5. 시각화 및 결과 저장
# =========================================================
def main():
    datasets = load_all_datasets()
    if not datasets: return
    
    all_results = []
    
    # 1. Run Analysis
    for case in CASES:
        if case in datasets:
            res = analyze_case(case, datasets[case])
            all_results.extend(res)
            
    df_res = pd.DataFrame(all_results)
    
    # 2. Save Excel
    # Summary (Mean RE, Std RE per Model/Case)
    summary = df_res.groupby(['Model', 'Case'])['Best_RE'].agg(['mean', 'std']).reset_index()
    summary.rename(columns={'mean': 'Mean_Accuracy(Lower Better)', 'std': 'Stability(Lower Better)'}, inplace=True)
    summary = summary.sort_values(['Model', 'Mean_Accuracy(Lower Better)'])
    
    with pd.ExcelWriter(os.path.join(OUT_DIR, "Comparative_Analysis_Result.xlsx")) as writer:
        df_res.to_excel(writer, sheet_name="Detail", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        
    print(f"\n[Saved] Excel Report")
    print(summary)
    
    # 3. Visualization
    # Common Style
    colors = {"Integrated": "black", "PM_Only": "blue", "VOC_Only": "red"}
    markers = {"Integrated": "o", "PM_Only": "s", "VOC_Only": "^"}
    
    # (1) Fig 1 & 2: Per Model Comparison
    for model in ["SVR", "ExtraTrees"]:
        plt.figure(figsize=(10, 6))
        df_m = df_res[df_res['Model'] == model]
        
        # Y-limit setting
        min_y, max_y = df_m['Best_RE'].min(), df_m['Best_RE'].max()
        margin = (max_y - min_y) * 0.1
        if margin == 0: margin = 0.01
        
        for case in CASES:
            df_c = df_m[df_m['Case'] == case]
            if df_c.empty: continue
            
            plt.plot(df_c['Percentage'], df_c['Best_RE'], 
                     color=colors[case], marker=markers[case], label=case, linewidth=2)
            
            # Text Annotation
            offset = margin * 0.05 if case == "Integrated" else (-margin * 0.05 if case == "PM_Only" else 0)
            for _, row in df_c.iterrows():
                plt.text(row['Percentage'], row['Best_RE'] + offset, f"N={int(row['Best_K'])}", 
                         color=colors[case], fontsize=8, ha='center', va='bottom' if offset >=0 else 'top')
        
        plt.title(f"[{model}] Comparison: Integrated vs PM vs VOC (Best RE)")
        plt.xlabel("Process Completion (%)")
        plt.ylabel("Best Mean Relative Error (RE)")
        plt.ylim(max(0, min_y - margin), max_y + margin)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(PCTS)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"Fig_{model}_Comparison.png"))
        plt.close()
        
    # (2) Fig 3: Best of Best Comparison
    # 각 모델별로 가장 Mean RE가 낮은 Case를 선정
    best_case_svr = summary[summary['Model'] == "SVR"].iloc[0]['Case']
    best_case_et = summary[summary['Model'] == "ExtraTrees"].iloc[0]['Case']
    
    plt.figure(figsize=(10, 6))
    
    # SVR Best Line
    df_svr_best = df_res[(df_res['Model'] == "SVR") & (df_res['Case'] == best_case_svr)]
    plt.plot(df_svr_best['Percentage'], df_svr_best['Best_RE'], 
             color='blue', marker='o', label=f"SVR Best ({best_case_svr})", linewidth=2.5)
             
    # ET Best Line
    df_et_best = df_res[(df_res['Model'] == "ExtraTrees") & (df_res['Case'] == best_case_et)]
    plt.plot(df_et_best['Percentage'], df_et_best['Best_RE'], 
             color='green', marker='s', label=f"ExtraTrees Best ({best_case_et})", linewidth=2.5)
    
    # Y-limit
    all_best_re = pd.concat([df_svr_best['Best_RE'], df_et_best['Best_RE']])
    min_y, max_y = all_best_re.min(), all_best_re.max()
    margin = (max_y - min_y) * 0.1
    if margin == 0: margin = 0.01
    
    plt.title(f"Model Comparison (Best Case Selected)")
    plt.xlabel("Process Completion (%)")
    plt.ylabel("Best Mean Relative Error (RE)")
    plt.ylim(max(0, min_y - margin), max_y + margin)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(PCTS)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig_Model_Best_Comparison.png"))
    plt.close()
    
    print("\n[SUCCESS] All plots and files generated.")

if __name__ == "__main__":
    main()

