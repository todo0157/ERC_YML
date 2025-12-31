from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_XLSX = Path("ERC/PM/features_pm_rich_p10to100_corr.xlsx")
FEATURE_SHEET = "features"
LABEL_XLSX = Path("ERC/Printing_qualitydata.xlsx")
LABEL_ORIGIN_COL = "Roughness(nm)"
LABEL_COL = "Ra (um)"

RESULT_XLSX = Path("ABS_Ra_p10to100_results_corr/results_ABS_p10to100_compare_corr.xlsx")

PCT = 10
FOCUS_K = 10

MODEL = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=0, max_iter=30000)),
    ]
)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def re_mean_percent(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    eps = 1e-12
    return float(np.mean(np.abs(y - yhat) / (np.abs(y) + eps)) * 100.0)


def clean_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.select_dtypes(include=[np.number]).copy()
    nan_ratio = X.isna().mean()
    X = X.loc[:, nan_ratio <= 0.5].copy()
    X = X.apply(lambda col: col.fillna(col.median()), axis=0)
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1].copy()
    return X


def select_only_pct_columns(X_all: pd.DataFrame, pct: int) -> pd.DataFrame:
    key1 = f"_p{pct}_"
    key2 = f"CC_p{pct}_"
    cols = []
    for c in X_all.columns:
        s = str(c)
        if (key1 in s) or (s.startswith(key2)) or (key2 in s):
            cols.append(c)
    return X_all[cols].copy()


def finite_report(df: pd.DataFrame, name: str) -> dict:
    arr = df.to_numpy(dtype=float, copy=False)
    return {
        f"{name}.shape": df.shape,
        f"{name}.nan": int(np.isnan(arr).sum()),
        f"{name}.inf": int(np.isinf(arr).sum()),
        f"{name}.finite_min": float(np.nanmin(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else math.nan,
        f"{name}.finite_max": float(np.nanmax(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else math.nan,
        f"{name}.absmax": float(np.nanmax(np.abs(arr[np.isfinite(arr)]))) if np.isfinite(arr).any() else math.nan,
    }


def main() -> None:
    print("=== Inputs ===")
    print("FEATURE_XLSX:", FEATURE_XLSX, "exists:", FEATURE_XLSX.exists())
    print("LABEL_XLSX  :", LABEL_XLSX, "exists:", LABEL_XLSX.exists())
    print("RESULT_XLSX :", RESULT_XLSX, "exists:", RESULT_XLSX.exists())
    print("PCT:", PCT)
    print()

    df_feat = pd.read_excel(FEATURE_XLSX, sheet_name=FEATURE_SHEET, engine="openpyxl")
    df_feat.columns = [str(c).strip() for c in df_feat.columns]

    df_label = pd.read_excel(LABEL_XLSX, engine="openpyxl")
    id_col_label = df_label.columns[0]
    df = pd.merge(
        df_feat,
        df_label[[id_col_label, LABEL_ORIGIN_COL]],
        left_on="sample_id",
        right_on=id_col_label,
        how="inner",
    )
    df[LABEL_COL] = df[LABEL_ORIGIN_COL] / 1000.0
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()

    print("=== Merge ===")
    print("merged samples:", len(df))
    print("sample_id head:", df["sample_id"].astype(str).head(5).tolist())
    print()

    drop_cols = [LABEL_COL, "sample_id", "n_points", id_col_label, LABEL_ORIGIN_COL]
    X_all = df.drop(columns=drop_cols, errors="ignore")
    X_all = clean_feature_matrix(X_all)

    Xp = select_only_pct_columns(X_all, PCT)
    Xp = clean_feature_matrix(Xp)
    y = df[LABEL_COL].to_numpy(dtype=float)
    sample_ids = df["sample_id"].astype(str).to_numpy()

    print("=== Feature matrix ===")
    print("X_all shape:", X_all.shape)
    print("Xp(pct) shape:", Xp.shape)
    rep = finite_report(Xp, "Xp")
    for k in sorted(rep.keys()):
        print(f"{k}: {rep[k]}")
    print()

    # Compare with saved predictions (optional)
    if RESULT_XLSX.exists():
        pred_saved = pd.read_excel(RESULT_XLSX, sheet_name="predictions", engine="openpyxl")
        col_full = f"Ra_pred_Full_p{PCT}"
        if col_full in pred_saved.columns:
            # align by sample_id
            m = pd.merge(
                pd.DataFrame({"sample_id": sample_ids, "Ra_true": y}),
                pred_saved[["sample_id", col_full]],
                on="sample_id",
                how="left",
            )
            print("=== Saved predictions check ===")
            print("saved pred col:", col_full)
            print("saved pred nan:", int(m[col_full].isna().sum()))
            print("saved pred min/max:", float(m[col_full].min()), float(m[col_full].max()))
            print("saved RMSE:", rmse(m["Ra_true"].to_numpy(), m[col_full].to_numpy()))
            print()

    # Recompute LOOCV predictions
    print("=== Recompute LOOCV (Full features) ===")
    cv = LeaveOneOut()
    yhat_full = cross_val_predict(MODEL, Xp, y, cv=cv)
    print("yhat_full min/max:", float(np.min(yhat_full)), float(np.max(yhat_full)))
    print("RMSE_full:", rmse(y, yhat_full))
    print("RE_full(%):", re_mean_percent(y, yhat_full))

    abs_err = np.abs(y - yhat_full)
    worst_i = int(np.argmax(abs_err))
    print()
    print("=== Worst sample (Full) ===")
    print("worst sample_id:", sample_ids[worst_i])
    print("y_true:", float(y[worst_i]))
    print("y_pred:", float(yhat_full[worst_i]))
    print("abs_err:", float(abs_err[worst_i]))
    print()

    # Inspect row magnitudes and non-finite issues
    row = Xp.iloc[worst_i]
    row_arr = row.to_numpy(dtype=float, copy=False)
    print("=== Worst sample feature diagnostics ===")
    print("row nan:", int(np.isnan(row_arr).sum()), "row inf:", int(np.isinf(row_arr).sum()))
    finite_row = row_arr[np.isfinite(row_arr)]
    print("row finite min/max:", float(np.min(finite_row)), float(np.max(finite_row)))
    print("row absmax:", float(np.max(np.abs(finite_row))))

    # Show top absolute features for that sample
    s_abs = row.abs().sort_values(ascending=False)
    print()
    print("Top 20 |feature value| for worst sample:")
    for k, v in s_abs.head(20).items():
        print(f"  {k}: {v}")

    # Find globally extreme columns (helps find 'ratio' explosions etc.)
    col_absmax = Xp.abs().max(axis=0).sort_values(ascending=False)
    print()
    print("Top 30 columns by global abs(max):")
    for k, v in col_absmax.head(30).items():
        print(f"  {k}: {v}")

    # Fold-level explanation: fit on train, inspect coefficients and contributions for worst sample
    print()
    print("=== Fold contribution analysis (worst sample) ===")
    mask = np.ones(len(y), dtype=bool)
    mask[worst_i] = False
    X_train = Xp.iloc[mask].copy()
    y_train = y[mask]
    X_test = Xp.iloc[[worst_i]].copy()

    MODEL.fit(X_train, y_train)
    y_pred = float(MODEL.predict(X_test)[0])
    print("refit pred for worst sample:", y_pred)

    scaler: StandardScaler = MODEL.named_steps["scaler"]
    enet: ElasticNet = MODEL.named_steps["model"]
    x_scaled = scaler.transform(X_test)[0]
    coefs = enet.coef_
    contrib = x_scaled * coefs
    idx = np.argsort(np.abs(contrib))[::-1][:20]
    cols = np.array(Xp.columns)
    print("Top 20 contributions (|x_scaled * coef|):")
    for j in idx:
        print(
            f"  {cols[j]}: contrib={contrib[j]: .6g}, x_scaled={x_scaled[j]: .6g}, coef={coefs[j]: .6g}"
        )

    # Compare with Top10 from saved excel (if available)
    if RESULT_XLSX.exists():
        top = pd.read_excel(RESULT_XLSX, sheet_name="top_features", engine="openpyxl")
        top10 = top[top["pct"] == PCT].sort_values("rank")["feature"].tolist()[:FOCUS_K]
        if top10:
            Xk = clean_feature_matrix(Xp[top10].copy())
            yhat_k = cross_val_predict(MODEL, Xk, y, cv=cv)
            print()
            print("=== Top10 re-run (sanity) ===")
            print("Top10 features:")
            for f in top10:
                print(" ", f)
            print("yhat_top10 min/max:", float(np.min(yhat_k)), float(np.max(yhat_k)))
            print("RMSE_top10:", rmse(y, yhat_k))
            print("RE_top10(%):", re_mean_percent(y, yhat_k))


if __name__ == "__main__":
    main()


