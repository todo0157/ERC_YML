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
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_PATH = os.path.join(BASE_DIR, "results_features_p10to100_tsfresh.xlsx")
TARGET_PATH = os.path.join(BASE_DIR, "Printing_qualitydata.xlsx")
OUT_DIR = os.path.join(BASE_DIR, "Comparative_Analysis_TSFRESH")
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
    if not os.path.exists(TARGET_PATH):
        print(f"[ERROR] Target file not found: {TARGET_PATH}")
        return {}

    df_target = pd.read_excel(TARGET_PATH)
    df_target.rename(columns={df_target.columns[0]: 'sample_id'}, inplace=True)
    df_target['sample_id'] = df_target['sample_id'].astype(str).str.strip()
    target_df = df_target[['sample_id', 'Roughness(nm)']]
    
    # Feature Sheets Load
    if not os.path.exists(FEATURE_PATH):
        print(f"[ERROR] Feature file not found: {FEATURE_PATH}")
        return {}
        
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
# 4. 분석 로직 (Best RE 탐색 & 예측값 반환)
# =========================================================
def analyze_case(case_name, df_data):
    results = []
    predictions = {} # (pct, model) -> (y_true, y_pred)
    
    # 선택된 피처 컬럼 식별
    drop_cols = ['sample_id', 'percentage', 'n_points', 'Roughness(nm)']
    feature_cols = [c for c in df_data.columns if c not in drop_cols]
    
    total_features = len(feature_cols)
    
    # print(f"\n>>> Analyzing [{case_name}] (Features: {total_features})") # 로그 과다 방지
    
    # Loop Percentage
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
        
        # [Safety] Remove constant features
        from sklearn.feature_selection import VarianceThreshold
        try:
            sel_var = VarianceThreshold(threshold=0)
            X_imputed = sel_var.fit_transform(X_imputed)
            support = sel_var.get_support()
            feature_cols_local = [feature_cols[i] for i in range(len(feature_cols)) if support[i]]
        except ValueError:
            print(f"    [Skip] All features are constant at {pct}%")
            continue
            
        # 1단계: F-test로 상위 200개 선정 (고속 필터링)
        k_first = min(200, X_imputed.shape[1])
        if k_first == 0: continue
        
        selector = SelectKBest(score_func=f_regression, k=k_first)
        try:
            X_selected = selector.fit_transform(X_imputed, y)
        except Exception as e:
            print(f"    [Error] SelectKBest failed: {e}")
            continue
        
        # 선택된 피처 리스트
        selected_mask = selector.get_support()
        current_features = [feature_cols_local[i] for i in range(len(feature_cols_local)) if selected_mask[i]]
        
        # 스케일링
        X_sc = scaler.fit_transform(X_selected)
        
        # Models Loop
        models = get_models()
        for m_name, model in models.items():
            # 2단계: Permutation Importance
            try:
                model.fit(X_sc, y)
                r = permutation_importance(model, X_sc, y, n_repeats=3, random_state=42, n_jobs=1)
                importances = r.importances_mean
                ranks = sorted(zip(range(len(current_features)), importances), key=lambda x: x[1], reverse=True)
                top_indices_local = [idx for idx, imp in ranks]
            except Exception as e:
                print(f"    [Warning] Importance calc failed: {e}")
                top_indices_local = list(range(len(current_features)))
            
            # 3단계: Top K 탐색
            n_curr = len(current_features)
            steps = np.linspace(0, n_curr, 9, dtype=int)[1:]
            steps = sorted(list(set(steps)))
            if 0 in steps: steps.remove(0)
            
            best_re = float('inf')
            best_k = -1
            
            loo = LeaveOneOut()
            
            # Find Best K
            for k in steps:
                sel_idx_local = top_indices_local[:k]
                X_final = X_sc[:, sel_idx_local]
                
                re_list = []
                for train_idx, test_idx in loo.split(X_final):
                    X_tr, X_te = X_final[train_idx], X_final[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]
                    
                    m_clone = get_models()[m_name]
                    m_clone.fit(X_tr, y_tr)
                    pred = max(0, m_clone.predict(X_te)[0])
                    
                    re_val = abs(y_te[0] - pred) / (y_te[0] + 1e-9)
                    re_list.append(re_val)
                
                mean_re = np.mean(re_list)
                if mean_re < best_re:
                    best_re = mean_re
                    best_k = k
            
            # Save Predictions for Best K (for Stacking)
            sel_idx_local = top_indices_local[:best_k]
            X_final_best = X_sc[:, sel_idx_local]
            
            y_pred_best = []
            for train_idx, test_idx in loo.split(X_final_best):
                X_tr, X_te = X_final_best[train_idx], X_final_best[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                
                m_clone = get_models()[m_name]
                m_clone.fit(X_tr, y_tr)
                pred = max(0, m_clone.predict(X_te)[0])
                y_pred_best.append(pred)
            
            predictions[(pct, m_name)] = (y, np.array(y_pred_best), sel_idx_local) # Save features indices too if needed
            
            results.append({
                "Case": case_name,
                "Model": m_name,
                "Percentage": pct,
                "Best_RE": best_re,
                "Best_K": best_k
            })
            print(f"    [{pct}%] {m_name}: Best RE={best_re:.4f} (Top {best_k})")
            
    return results, predictions

# =========================================================
# 5-1. Stacking (Late Fusion) 수행
# =========================================================
def perform_late_fusion(predictions_pm, predictions_voc):
    print("\n>>> Performing Late Fusion (Stacking)...")
    stacking_results = []
    
    # Stacking Model: Ridge Regression (Simple but effective for ensembling)
    meta_model = Ridge(alpha=1.0)
    loo = LeaveOneOut()
    
    for pct in PCTS:
        for m_name in ["SVR", "ExtraTrees"]:
            if (pct, m_name) not in predictions_pm or (pct, m_name) not in predictions_voc:
                continue
                
            y_true, y_pred_pm, _ = predictions_pm[(pct, m_name)]
            _, y_pred_voc, _ = predictions_voc[(pct, m_name)]
            
            # Input for Meta Model: [Pred_PM, Pred_VOC]
            X_stack = np.column_stack([y_pred_pm, y_pred_voc])
            y = y_true
            
            re_list = []
            
            # Meta Model CV
            for train_idx, test_idx in loo.split(X_stack):
                X_tr, X_te = X_stack[train_idx], X_stack[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                
                meta_model.fit(X_tr, y_tr)
                pred = max(0, meta_model.predict(X_te)[0])
                
                re_val = abs(y_te[0] - pred) / (y_te[0] + 1e-9)
                re_list.append(re_val)
                
            mean_re = np.mean(re_list)
            
            stacking_results.append({
                "Case": "Integrated_LateFusion",
                "Model": m_name,
                "Percentage": pct,
                "Best_RE": mean_re,
                "Best_K": 2 # PM + VOC (2 inputs)
            })
            print(f"    [{pct}%] {m_name} (Stacking): RE={mean_re:.4f}")
            
    return stacking_results

# =========================================================
# 5-2. Residual Learning (PM + VOC Residual) 수행
# =========================================================
def perform_residual_learning(predictions_pm, df_voc):
    print("\n>>> Performing Residual Learning (PM + VOC Correction)...")
    residual_results = []
    
    # VOC Data Prep (동일하게 전처리 필요)
    drop_cols = ['sample_id', 'percentage', 'n_points', 'Roughness(nm)']
    feature_cols = [c for c in df_voc.columns if c not in drop_cols]
    
    loo = LeaveOneOut()
    
    for pct in PCTS:
        df_sub = df_voc[df_voc['percentage'] == pct].copy()
        if len(df_sub) < 5: continue
        
        # VOC Features
        X_voc_raw = df_sub[feature_cols].values
        # Preprocess VOC
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        X_voc_raw = np.nan_to_num(X_voc_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_voc_imp = imputer.fit_transform(X_voc_raw)
        
        # Feature Selection for VOC (Residual 예측용)
        # 여기서는 빠르게 Top 50개만 사용 (과적합 방지)
        k_res = min(50, X_voc_imp.shape[1])
        
        for m_name in ["SVR", "ExtraTrees"]:
            if (pct, m_name) not in predictions_pm: continue
            
            y_true, y_pred_pm, _ = predictions_pm[(pct, m_name)]
            
            # 1. 잔차 계산 (Target for VOC)
            residuals = y_true - y_pred_pm
            
            # 2. VOC Feature Selection (Target이 Residual임)
            selector = SelectKBest(score_func=f_regression, k=k_res)
            try:
                X_voc_sel = selector.fit_transform(X_voc_imp, residuals)
                X_voc_sc = scaler.fit_transform(X_voc_sel)
            except:
                continue
                
            # 3. Residual Learning CV
            re_list = []
            
            # Residual 예측 모델 (복잡하면 오히려 독이 되므로 SVR 고정 혹은 가벼운 모델 추천)
            # 여기서는 메인 모델과 동일한 구조 사용하되 파라미터 조정
            if m_name == "SVR":
                res_model = SVR(kernel='rbf', C=10, epsilon=0.01) # 조금 더 유연하게
            else:
                res_model = ExtraTreesRegressor(n_estimators=50, max_depth=5, random_state=42) # 과적합 방지
            
            for train_idx, test_idx in loo.split(X_voc_sc):
                # Split Data
                X_tr, X_te = X_voc_sc[train_idx], X_voc_sc[test_idx]
                r_tr = residuals[train_idx] # Train on Residuals
                
                # PM Prediction (이미 구해둠)
                pm_pred_val = y_pred_pm[test_idx][0]
                true_val = y_true[test_idx][0]
                
                # Fit & Predict Residual
                res_model.fit(X_tr, r_tr)
                res_pred = res_model.predict(X_te)[0]
                
                # Final Prediction = PM_Pred + Residual_Pred
                final_pred = max(0, pm_pred_val + res_pred)
                
                re_val = abs(true_val - final_pred) / (true_val + 1e-9)
                re_list.append(re_val)
            
            mean_re = np.mean(re_list)
            
            residual_results.append({
                "Case": "Integrated_Residual",
                "Model": m_name,
                "Percentage": pct,
                "Best_RE": mean_re,
                "Best_K": k_res # VOC feature count
            })
            print(f"    [{pct}%] {m_name} (Residual): RE={mean_re:.4f}")
            
    return residual_results

# =========================================================
# 6. 메인 실행 및 저장
# =========================================================
def main():
    datasets = load_all_datasets()
    if not datasets: return
    
    all_results = []
    all_predictions = {} # case -> predictions dict
    
    # 1. Run Analysis for Basic Cases
    for case in CASES:
        if case in datasets:
            res, preds = analyze_case(case, datasets[case])
            all_results.extend(res)
            all_predictions[case] = preds
            
    # 2. Run Late Fusion
    if "PM_Only" in all_predictions and "VOC_Only" in all_predictions:
        res_stack = perform_late_fusion(all_predictions["PM_Only"], all_predictions["VOC_Only"])
        all_results.extend(res_stack)
        
    # 3. Run Residual Learning (NEW)
    if "PM_Only" in all_predictions and "VOC_Only" in datasets:
        res_resid = perform_residual_learning(all_predictions["PM_Only"], datasets["VOC_Only"])
        all_results.extend(res_resid)
    
    df_res = pd.DataFrame(all_results)
    
    # 4. Save Excel
    summary = df_res.groupby(['Model', 'Case'])['Best_RE'].agg(['mean', 'std']).reset_index()
    summary.rename(columns={'mean': 'Mean_Accuracy(Lower Better)', 'std': 'Stability(Lower Better)'}, inplace=True)
    summary = summary.sort_values(['Model', 'Mean_Accuracy(Lower Better)'])
    
    out_file = os.path.join(OUT_DIR, "Comparative_Analysis_TSFRESH_Result_v3.xlsx")
    with pd.ExcelWriter(out_file) as writer:
        df_res.to_excel(writer, sheet_name="Detail", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        
    print(f"\n[Saved] Excel Report to {out_file}")
    print(summary)
    
    # 5. Visualization
    colors = {
        "Integrated": "black", 
        "PM_Only": "blue", 
        "VOC_Only": "red",
        "Integrated_LateFusion": "purple",
        "Integrated_Residual": "green" # New Color
    }
    markers = {
        "Integrated": "o", 
        "PM_Only": "s", 
        "VOC_Only": "^",
        "Integrated_LateFusion": "*",
        "Integrated_Residual": "D" # Diamond
    }
    
    # (1) Per Model Comparison
    for model in ["SVR", "ExtraTrees"]:
        plt.figure(figsize=(10, 6))
        df_m = df_res[df_res['Model'] == model]
        if df_m.empty: continue

        min_y, max_y = df_m['Best_RE'].min(), df_m['Best_RE'].max()
        margin = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 0.01
        
        # Plot Order
        plot_cases = ["Integrated", "PM_Only", "VOC_Only", "Integrated_LateFusion", "Integrated_Residual"]
        
        for case in plot_cases:
            df_c = df_m[df_m['Case'] == case]
            if df_c.empty: continue
            
            plt.plot(df_c['Percentage'], df_c['Best_RE'], 
                     color=colors.get(case, "gray"), 
                     marker=markers.get(case, "x"), 
                     label=case, linewidth=2, markersize=8)
            
            # Text Annotation (Top K)
            if "Integrated" not in case: # Too crowded
                offset = margin * 0.05 if case == "PM_Only" else -margin * 0.05
                for _, row in df_c.iterrows():
                    plt.text(row['Percentage'], row['Best_RE'] + offset, f"N={int(row['Best_K'])}", 
                             color=colors.get(case, "black"), fontsize=8, ha='center')
        
        plt.title(f"[{model}] Comparison: Integrated (Residual) vs PM vs VOC")
        plt.xlabel("Process Completion (%)")
        plt.ylabel("Best Mean Relative Error (RE)")
        plt.ylim(max(0, min_y - margin), max_y + margin)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(PCTS)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"Fig_{model}_Comparison_Residual.png"))
        plt.close()
        
    # (2) Best of Best Comparison
    best_rows = []
    for m in ["SVR", "ExtraTrees"]:
        df_m = df_res[df_res['Model'] == m]
        best_case = df_m.groupby('Case')['Best_RE'].mean().idxmin()
        best_rows.append((m, best_case))
        
    plt.figure(figsize=(10, 6))
    
    for m, best_case in best_rows:
        df_best = df_res[(df_res['Model'] == m) & (df_res['Case'] == best_case)]
        c = 'blue' if m == "SVR" else 'green'
        lbl = f"{m} Best ({best_case})"
        plt.plot(df_best['Percentage'], df_best['Best_RE'], 
                 color=c, marker='o', label=lbl, linewidth=2.5)
                 
    plt.title(f"Model Comparison (Best Case Selected)")
    plt.xlabel("Process Completion (%)")
    plt.ylabel("Best Mean Relative Error (RE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(PCTS)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig_Model_Best_Comparison_Residual.png"))
    plt.close()
    
    print("\n[SUCCESS] All plots and files generated.")

if __name__ == "__main__":
    main()
