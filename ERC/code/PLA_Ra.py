import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


# =========================================================
# 1) 설정
# =========================================================
INPUT_XLSX = r"features_pm_rich_p10to100.xlsx"
SHEET_NAME = None

LABEL_COL = "Ra (um)"          # ✅ 라벨 헤더 정확히
PCTS = list(range(10, 101, 10))

# “중요 feature만” 비교용 K (원하면 5/10/20 등 바꿔도 됨)
FOCUS_K = 10

# “자동 BestK per pct”를 뽑기 위한 후보 K들
TOPK_CANDIDATES = [5, 10, 20, 40]

# Permutation importance
PERM_REPEATS = 80
RANDOM_STATE = 0
TOPN_IMPORT_FIG = 25

OUT_DIR = "ML_ONLY_p10to100_results_TOP"
OUT_XLSX = "results_ONLY_p10to100_TOP_compare.xlsx"


# =========================================================
# 2) 유틸
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_read_excel(path, sheet_name=None):
    obj = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if isinstance(obj, dict):
        if len(obj) == 0:
            raise ValueError("엑셀에서 읽을 시트가 없습니다.")
        return obj[list(obj.keys())[0]]
    return obj

def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def re_mean_percent(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    eps = 1e-12
    return float(np.mean(np.abs(y - yhat) / (np.abs(y) + eps)) * 100.0)

def baseline_metrics(y):
    mu = float(np.mean(y))
    yhat = np.full_like(y, mu, dtype=float)
    return mu, rmse(y, yhat), re_mean_percent(y, yhat)

def clean_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.select_dtypes(include=[np.number]).copy()
    nan_ratio = X.isna().mean()
    X = X.loc[:, nan_ratio <= 0.5].copy()
    X = X.apply(lambda col: col.fillna(col.median()), axis=0)
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1].copy()
    return X

def select_only_pct_columns(X_all: pd.DataFrame, pct: int) -> pd.DataFrame:
    """
    pX only feature 선택:
    - 일반: ..._p10_...
    - CC:   CC_p10_...
    """
    key1 = f"_p{pct}_"
    key2 = f"CC_p{pct}_"
    cols = []
    for c in X_all.columns:
        s = str(c)
        if (key1 in s) or (s.startswith(key2)) or (key2 in s):
            cols.append(c)
    return X_all[cols].copy()

def topk_features(cols, scores, k):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    cols_sorted = np.array(cols)[order]
    return list(cols_sorted[:min(k, len(cols_sorted))])

def save_importance_bar(cols, scores, title, outpath, topn=25, xlabel="perm importance (MSE increase)"):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    cols_sorted = np.array(cols)[order][:min(topn, len(cols))]
    scores_sorted = scores[order][:min(topn, len(cols))]
    cols_sorted = cols_sorted[::-1]
    scores_sorted = scores_sorted[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(cols_sorted, scores_sorted)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_compare_lines(pcts, y_full, y_top, baseline, ylabel, title, outpath, label_top="TopK"):
    plt.figure()
    plt.plot(pcts, y_full, marker="o", color="black", label="Aerosol feature")
    plt.plot(pcts, y_top, marker="o", color="red", label=label_top)
    plt.axhline(baseline, linestyle="--", color="#4C92C3", label="Baseline (mean-only)")
    plt.xlabel("Percentage of build time (0 ~ pct %)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, edgecolor="gray", facecolor="white", fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =========================================================
# 3) 모델 (소샘플 안정형)
# =========================================================
MODEL = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=0, max_iter=30000))
])


# =========================================================
# 4) 메인
# =========================================================
def main():
    ensure_dir(OUT_DIR)
    fig_dir = os.path.join(OUT_DIR, "figures")
    imp_dir = os.path.join(fig_dir, "importance_by_pct")
    ensure_dir(fig_dir)
    ensure_dir(imp_dir)

    df = safe_read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]

    if LABEL_COL not in df.columns:
        ra_like = [c for c in df.columns if "ra" in str(c).lower()]
        raise ValueError(f"라벨 컬럼 '{LABEL_COL}' 없음. ra 후보: {ra_like[:10]}")

    # y
    y_series = pd.to_numeric(df[LABEL_COL], errors="coerce")
    valid = np.isfinite(y_series.values)
    df = df.loc[valid, :].copy()
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(float).values

    if len(y) < 5:
        raise ValueError("유효 샘플 수가 너무 적음(>=5 필요).")

    # sample_id
    if "sample_id" in df.columns:
        sample_id = df["sample_id"].astype(str).reset_index(drop=True)
    else:
        sample_id = pd.Series(np.arange(len(y))).astype(str)

    # X_all
    X_all = df.drop(columns=[LABEL_COL], errors="ignore")
    X_all = clean_feature_matrix(X_all)

    base_mean, base_rmse, base_re = baseline_metrics(y)
    print(f"Baseline: mean={base_mean:.4f} um | RMSE={base_rmse:.4f} um | RE={base_re:.2f}%")
    print(f"samples={len(y)}, all_numeric_features={X_all.shape[1]}")

    cv = LeaveOneOut()

    # 저장용
    pred_df = pd.DataFrame({"sample_id": sample_id, "Ra_true": y})
    metrics_rows = []
    imp_rows = []
    selected_rows = []

    used_pcts = []

    rmse_full_line = []
    re_full_line = []

    rmse_focus_line = []
    re_focus_line = []

    rmse_bestk_line = []
    re_bestk_line = []
    bestk_used_line = []

    for pct in PCTS:
        Xp = select_only_pct_columns(X_all, pct)
        Xp = clean_feature_matrix(Xp)

        if Xp.shape[1] < 3:
            print(f"⚠️ p{pct}: feature 부족({Xp.shape[1]}) -> skip")
            continue

        used_pcts.append(pct)

        # 1) FULL LOOCV
        yhat_full = cross_val_predict(MODEL, Xp, y, cv=cv)
        r_full = rmse(y, yhat_full)
        re_full = re_mean_percent(y, yhat_full)
        rmse_full_line.append(r_full)
        re_full_line.append(re_full)

        pred_df[f"Ra_pred_FULL_p{pct}"] = yhat_full

        # 2) Importance (fit on all -> permutation)
        MODEL.fit(Xp, y)
        perm = permutation_importance(
            MODEL, Xp, y,
            n_repeats=PERM_REPEATS,
            random_state=RANDOM_STATE,
            scoring="neg_mean_squared_error"
        )
        perm_imp = perm.importances_mean  # MSE 증가량 (클수록 중요)

        # 저장(롱포맷)
        for f, s in zip(Xp.columns, perm_imp):
            imp_rows.append({"pct": pct, "feature": f, "perm_MSE_increase": float(s)})

        # importance figure
        save_importance_bar(
            Xp.columns, perm_imp,
            title=f"Permutation importance (ONLY p{pct})",
            outpath=os.path.join(imp_dir, f"importance_perm_p{pct}.png"),
            topn=TOPN_IMPORT_FIG
        )

        # 3) FOCUS_K로 “중요 feature만” LOOCV
        K = min(FOCUS_K, Xp.shape[1])
        colsK = topk_features(Xp.columns, perm_imp, K)
        Xk = Xp[colsK].copy()
        Xk = clean_feature_matrix(Xk)

        yhat_k = cross_val_predict(MODEL, Xk, y, cv=cv)
        r_k = rmse(y, yhat_k)
        re_k = re_mean_percent(y, yhat_k)

        rmse_focus_line.append(r_k)
        re_focus_line.append(re_k)

        pred_df[f"Ra_pred_Top{FOCUS_K}_p{pct}"] = yhat_k

        for rank, f in enumerate(colsK, start=1):
            selected_rows.append({"pct": pct, "mode": f"Top{FOCUS_K}", "K": K, "rank": rank, "feature": f})

        # 4) BestK per pct (TOPK_CANDIDATES 중 RMSE 최소)
        best = None
        for kk in TOPK_CANDIDATES:
            kk2 = min(kk, Xp.shape[1])
            cols = topk_features(Xp.columns, perm_imp, kk2)
            Xkk = clean_feature_matrix(Xp[cols].copy())
            if Xkk.shape[1] < 3:
                continue
            yhat = cross_val_predict(MODEL, Xkk, y, cv=cv)
            rr = rmse(y, yhat)
            rre = re_mean_percent(y, yhat)
            if (best is None) or (rr < best["RMSE_um"]):
                best = {"K": kk2, "RMSE_um": rr, "RE_%": rre}

        if best is None:
            rmse_bestk_line.append(np.nan)
            re_bestk_line.append(np.nan)
            bestk_used_line.append(np.nan)
        else:
            rmse_bestk_line.append(best["RMSE_um"])
            re_bestk_line.append(best["RE_%"])
            bestk_used_line.append(best["K"])

        # metrics row
        metrics_rows.append({
            "pct": pct,
            "nfeat_full": int(Xp.shape[1]),
            "RMSE_full_um": r_full,
            "RE_full_%": re_full,
            f"RMSE_top{FOCUS_K}_um": r_k,
            f"RE_top{FOCUS_K}_%": re_k,
            "bestK_perPct": (best["K"] if best else np.nan),
            "RMSE_bestK_um": (best["RMSE_um"] if best else np.nan),
            "RE_bestK_%": (best["RE_%"] if best else np.nan),
        })

        print(f"p{pct:03d} | FULL RMSE={r_full:.4f}, Top{FOCUS_K} RMSE={r_k:.4f}, bestK={bestk_used_line[-1]}")

    if not used_pcts:
        raise RuntimeError("사용 가능한 pct가 없습니다. 컬럼명에 _p10_ 같은 형태가 있는지 확인하세요.")

    # ===== Figure: FULL vs TopK(FOCUS_K)
    plot_compare_lines(
        used_pcts, rmse_full_line, rmse_focus_line, base_rmse,
        ylabel="RMSE (um)",
        title=f"RMSE: Aerosol feature vs Top{FOCUS_K} important features (pX only) + Baseline",
        outpath=os.path.join(fig_dir, f"RMSE_FULL_vs_Top{FOCUS_K}.png"),
        label_top=f"Top{FOCUS_K} (perm-selected)"
    )
    plot_compare_lines(
        used_pcts, re_full_line, re_focus_line, base_re,
        ylabel="RE (%)",
        title=f"RE: FULL vs Top{FOCUS_K} important features (pX only) + Baseline",
        outpath=os.path.join(fig_dir, f"RE_FULL_vs_Top{FOCUS_K}.png"),
        label_top=f"Top{FOCUS_K} (perm-selected)"
    )

    # ===== Figure: FULL vs BestK per pct
    plot_compare_lines(
        used_pcts, rmse_full_line, rmse_bestk_line, base_rmse,
        ylabel="RMSE (um)",
        title="RMSE: Aerosol feature vs BestK + Baseline",
        outpath=os.path.join(fig_dir, "RMSE_FULL_vs_BestK_perPct.png"),
        label_top="BestK"
    )
    plot_compare_lines(
        used_pcts, re_full_line, re_bestk_line, base_re,
        ylabel="RE (%)",
        title="RE: Aerosol feature vs BestK + Baseline",
        outpath=os.path.join(fig_dir, "RE_FULL_vs_BestK_perPct.png"),
        label_top="BestK"
    )

    # ===== 저장 Excel
    out_path = os.path.join(OUT_DIR, OUT_XLSX)
    metrics_df = pd.DataFrame(metrics_rows)
    imp_df = pd.DataFrame(imp_rows)
    sel_df = pd.DataFrame(selected_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame([{
            "input_file": INPUT_XLSX,
            "label_col": LABEL_COL,
            "samples": len(y),
            "baseline_mean_um": base_mean,
            "baseline_RMSE_um": base_rmse,
            "baseline_RE_%": base_re,
            "model": "ElasticNet(scaled)",
            "FOCUS_K": FOCUS_K,
            "TOPK_CANDIDATES": str(TOPK_CANDIDATES)
        }]).to_excel(writer, index=False, sheet_name="baseline_meta")

        metrics_df.to_excel(writer, index=False, sheet_name="metrics")
        imp_df.to_excel(writer, index=False, sheet_name="importance_long"[:31])
        sel_df.to_excel(writer, index=False, sheet_name="selected_features"[:31])
        pred_df.to_excel(writer, index=False, sheet_name="predictions"[:31])

    print("\n완료!")
    print(" - 엑셀:", out_path)
    print(" - figures:", fig_dir)
    print(" - importance:", imp_dir)


if __name__ == "__main__":
    main()
