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
# ABS Ra Prediction (PM + VOC 결합 특징)
#
# - 기존 코드는 유지
# - PM 특징 파일 + VOC 특징 파일을 sample_id로 병합하여 예측
# - pct별(p10~p100) 특징만 선택하여 LOOCV + permutation importance 수행
#
# 추천 입력(안정성):
# - PM: features_pm_rich_p10to100_corr_robust.xlsx (ratio 폭발 방지 버전)
# - VOC: features_voc_rich_p10to100.xlsx
# =========================================================


# =========================================================
# 1) 설정
# =========================================================
PM_INPUT_XLSX = r"ERC/PM/features_pm_rich_p10to100_corr_robust.xlsx"
PM_SHEET_NAME = "features"

VOC_INPUT_XLSX = r"ERC/VOC/features_voc_rich_p10to100.xlsx"
VOC_SHEET_NAME = "features"

# 정답지(라벨)
LABEL_XLSX = r"ERC/Printing_qualitydata.xlsx"
LABEL_ORIGIN_COL = "Roughness(nm)"
LABEL_COL = "Ra (um)"

PCTS = list(range(10, 101, 10))
FOCUS_K = 10

# Permutation importance
PERM_REPEATS = 80
RANDOM_STATE = 0
TOPN_IMPORT_FIG = 25

# 출력(기존 결과와 분리)
OUT_DIR = "ABS_Ra_p10to100_results_PM_VOC"
OUT_XLSX = "results_ABS_p10to100_compare_PM_VOC.xlsx"

# (옵션) 결합 feature에서 outlier 영향 완화용 winsorize
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99


# =========================================================
# 2) 유틸
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_read_excel(path, sheet_name=None):
    if not os.path.exists(path):
        base_name = os.path.basename(path)
        if os.path.exists(base_name):
            path = base_name
        else:
            raise FileNotFoundError(
                f"엑셀 파일을 찾을 수 없습니다: {path}\n전처리 코드를 먼저 실행했는지 확인하세요."
            )

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
    """
    - numeric만 사용
    - NaN 50% 초과 컬럼 제거
    - 결측 median 대체
    - 열별 winsorize(1~99%)로 outlier 영향 완화 (옵션)
    - 상수 컬럼 제거
    """
    X = X.select_dtypes(include=[np.number]).copy()

    nan_ratio = X.isna().mean()
    X = X.loc[:, nan_ratio <= 0.5].copy()
    X = X.apply(lambda col: col.fillna(col.median()), axis=0)

    if X.shape[0] >= 8 and X.shape[1] >= 1:
        ql = X.quantile(WINSOR_Q_LOW)
        qh = X.quantile(WINSOR_Q_HIGH)
        X = X.clip(lower=ql, upper=qh, axis=1)

    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1].copy()
    return X


def select_only_pct_columns(X_all: pd.DataFrame, pct: int) -> pd.DataFrame:
    """
    PM/VOC 모두 컬럼명에 '_p{pct}_'가 포함되므로 이를 기준으로 선택.
    (PM의 CC feature도 'CC_p{pct}_' 형태라 자연스럽게 포함됨)
    """
    key = f"_p{pct}_"
    cols = [c for c in X_all.columns if key in str(c)]
    return X_all[cols].copy()


def topk_features(cols, scores, k):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    cols_sorted = np.array(cols)[order]
    return list(cols_sorted[: min(k, len(cols_sorted))])


def save_importance_bar(cols, scores, title, outpath, topn=25, xlabel="Permutation Importance (MSE Increase)"):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    n_show = min(topn, len(cols))
    cols_sorted = np.array(cols)[order][:n_show][::-1]
    scores_sorted = scores[order][:n_show][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(cols_sorted, scores_sorted, color="salmon")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_compare_lines(pcts, y_full, y_top, baseline, ylabel, title, outpath, label_top="TopK"):
    plt.figure(figsize=(8, 5))
    plt.plot(pcts, y_full, marker="o", color="black", label="Full Features (PM+VOC)")
    plt.plot(pcts, y_top, marker="o", color="red", label=label_top)
    plt.axhline(baseline, linestyle="--", color="blue", label="Baseline (Mean)")
    plt.xlabel("Percentage of build time (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =========================================================
# 3) 모델
# =========================================================
MODEL = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=30000)),
    ]
)


# =========================================================
# 4) 메인
# =========================================================
def main():
    ensure_dir(OUT_DIR)
    fig_dir = os.path.join(OUT_DIR, "figures")
    imp_dir = os.path.join(fig_dir, "importance_by_pct")
    ensure_dir(fig_dir)
    ensure_dir(imp_dir)

    # --- Load PM/VOC feature tables
    print(f"Loading PM features : {PM_INPUT_XLSX}")
    df_pm = safe_read_excel(PM_INPUT_XLSX, sheet_name=PM_SHEET_NAME)
    df_pm.columns = [str(c).strip() for c in df_pm.columns]

    print(f"Loading VOC features: {VOC_INPUT_XLSX}")
    df_voc = safe_read_excel(VOC_INPUT_XLSX, sheet_name=VOC_SHEET_NAME)
    df_voc.columns = [str(c).strip() for c in df_voc.columns]

    # --- Prefix feature columns to avoid collisions
    key_cols = ["sample_id"]
    for k in key_cols:
        if k not in df_pm.columns or k not in df_voc.columns:
            raise ValueError(f"Both feature files must contain '{k}' column.")

    def _prefix_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = df.copy()
        keep = {"sample_id", "n_points"}
        ren = {}
        for c in df.columns:
            if c in keep:
                continue
            ren[c] = f"{prefix}{c}"
        return df.rename(columns=ren)

    df_pm2 = _prefix_features(df_pm, "PM__")
    df_voc2 = _prefix_features(df_voc, "VOC__")

    # --- Merge PM+VOC by sample_id
    df_feat = pd.merge(df_pm2, df_voc2, on="sample_id", how="inner", suffixes=("", ""))
    if df_feat.empty:
        raise ValueError("PM/VOC feature merge is empty. Check sample_id alignment.")

    # --- Load label & merge
    print(f"Loading labels     : {LABEL_XLSX}")
    df_label = pd.read_excel(LABEL_XLSX, engine="openpyxl")
    id_col_label = df_label.columns[0]
    df = pd.merge(df_feat, df_label[[id_col_label, LABEL_ORIGIN_COL]], left_on="sample_id", right_on=id_col_label, how="inner")
    if df.empty:
        print("[Error] Merge result is empty. (sample_id vs label id mismatch)")
        return

    df[LABEL_COL] = df[LABEL_ORIGIN_COL] / 1000.0
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()
    y = df[LABEL_COL].values
    if len(y) < 5:
        print(f"[Error] Not enough samples (n={len(y)})")
        return

    print(f"[Success] Merged: {len(df)} samples matched.")

    # --- Build X
    drop_cols = [LABEL_COL, "sample_id", "n_points", "PM__n_points", "VOC__n_points", id_col_label, LABEL_ORIGIN_COL]
    X_all = df.drop(columns=drop_cols, errors="ignore")
    X_all = clean_feature_matrix(X_all)

    base_mu, base_rmse, base_re = baseline_metrics(y)
    print(f"Baseline: RMSE={base_rmse:.4f}, RE={base_re:.2f}%")

    cv = LeaveOneOut()
    metrics_list, importance_list, selection_list = [], [], []
    prediction_df = pd.DataFrame({"sample_id": df["sample_id"].astype(str).values, "Ra_true": y})

    line_full_rmse, line_full_re = [], []
    line_focus_rmse, line_focus_re = [], []
    valid_pcts = []

    for pct in PCTS:
        Xp = select_only_pct_columns(X_all, pct)
        Xp = clean_feature_matrix(Xp)
        if Xp.shape[1] < 3:
            print(f"p{pct:03d} | skip (not enough features: {Xp.shape[1]})")
            continue

        valid_pcts.append(pct)

        # 1) Full LOOCV
        yhat_full = cross_val_predict(MODEL, Xp, y, cv=cv)
        r_full, re_full = rmse(y, yhat_full), re_mean_percent(y, yhat_full)
        line_full_rmse.append(r_full)
        line_full_re.append(re_full)
        prediction_df[f"Ra_pred_Full_p{pct}"] = yhat_full

        # 2) Importance
        MODEL.fit(Xp, y)
        perm = permutation_importance(
            MODEL, Xp, y,
            n_repeats=PERM_REPEATS, random_state=RANDOM_STATE,
            scoring="neg_mean_squared_error",
        )
        perm_scores = perm.importances_mean
        for feat, score in zip(Xp.columns, perm_scores):
            importance_list.append({"pct": pct, "feature": feat, "importance": float(score)})
        save_importance_bar(
            Xp.columns, perm_scores,
            f"ABS Importance (PM+VOC) - p{pct}%",
            os.path.join(imp_dir, f"importance_p{pct}.png"),
            topn=TOPN_IMPORT_FIG,
        )

        # 3) TopK LOOCV
        k_focus = min(FOCUS_K, Xp.shape[1])
        top_feats = topk_features(Xp.columns, perm_scores, k_focus)
        Xk = clean_feature_matrix(Xp[top_feats].copy())
        yhat_focus = cross_val_predict(MODEL, Xk, y, cv=cv)
        r_focus, re_focus = rmse(y, yhat_focus), re_mean_percent(y, yhat_focus)
        line_focus_rmse.append(r_focus)
        line_focus_re.append(re_focus)
        prediction_df[f"Ra_pred_Top{FOCUS_K}_p{pct}"] = yhat_focus

        for r, f in enumerate(top_feats, 1):
            selection_list.append({"pct": pct, "rank": r, "feature": f})

        metrics_list.append({
            "pct": pct, "n_feat": Xp.shape[1],
            "RMSE_full": r_full, "RE_full": re_full,
            f"RMSE_top{FOCUS_K}": r_focus, f"RE_top{FOCUS_K}": re_focus,
        })
        print(f"p{pct:03d} | Full RMSE: {r_full:.4f} | Top{FOCUS_K} RMSE: {r_focus:.4f}")

    if valid_pcts:
        plot_compare_lines(valid_pcts, line_full_rmse, line_focus_rmse, base_rmse, "RMSE (um)",
                           "ABS Ra Prediction (PM+VOC) - RMSE", os.path.join(fig_dir, "RMSE_Comparison.png"))
        plot_compare_lines(valid_pcts, line_full_re, line_focus_re, base_re, "RE (%)",
                           "ABS Ra Prediction (PM+VOC) - RE (%)", os.path.join(fig_dir, "RE_Comparison.png"))

    final_out = os.path.join(OUT_DIR, OUT_XLSX)
    with pd.ExcelWriter(final_out, engine="openpyxl") as writer:
        pd.DataFrame(metrics_list).to_excel(writer, index=False, sheet_name="metrics")
        pd.DataFrame(importance_list).to_excel(writer, index=False, sheet_name="importance_all")
        pd.DataFrame(selection_list).to_excel(writer, index=False, sheet_name="top_features")
        prediction_df.to_excel(writer, index=False, sheet_name="predictions")

    print(f"\n[Success] Result folder: {OUT_DIR}")


if __name__ == "__main__":
    main()


