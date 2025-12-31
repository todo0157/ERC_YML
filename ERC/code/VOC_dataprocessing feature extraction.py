import os
import glob
import numpy as np
import pandas as pd

# =========================================================
# 1) 설정 (VOC 환경에 맞게 수정)
# =========================================================
# repo 루트에서 실행해도 VOC 폴더를 정확히 가리키도록 고정
DATA_DIR = os.path.join("ERC", "VOC")      # raw 엑셀 폴더
SHEET_NAME = "VOC"     # 시트명 (실제 엑셀 시트명에 맞게 "Sheet1" 등으로 수정 가능)
COL_VOC = "VOC"        # 사용할 VOC 컬럼명

# 누적 구간(10% 단위)
PCTS = list(range(10, 101, 10))

# 피크/threshold 관련
THRESH_K = 2.0  # threshold = mean + K*std
ROLL_WIN = 10   # rolling std 계산 창 크기(포인트 수)

# 출력 파일명
OUT_XLSX = os.path.join(DATA_DIR, "features_voc_rich_p10to100.xlsx")


# =========================================================
# 2) 기본 통계 함수들
# =========================================================
def _safe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x

def _skewness(x):
    x = _safe(x)
    if len(x) < 5:
        return np.nan
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        return 0.0
    return float(np.mean((x - mu) ** 3) / (sd ** 3))

def _kurtosis_excess(x):
    x = _safe(x)
    if len(x) < 5:
        return np.nan
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        return -3.0
    return float(np.mean((x - mu) ** 4) / (sd ** 4) - 3.0)

def base_stats(x):
    x = _safe(x)
    if len(x) < 5:
        return dict(max=np.nan, median=np.nan, mean=np.nan, min=np.nan,
                    std=np.nan, skew=np.nan, kurt=np.nan, range=np.nan)
    return {
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "std": float(np.std(x, ddof=1)),
        "skew": _skewness(x),
        "kurt": _kurtosis_excess(x),
        "range": float(np.max(x) - np.min(x)),
    }

def percentile_stats(x):
    x = _safe(x)
    if len(x) < 5:
        return dict(p90=np.nan, p95=np.nan, p99=np.nan, iqr=np.nan)
    p25, p75 = np.percentile(x, [25, 75])
    return {
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "iqr": float(p75 - p25),
    }

# =========================================================
# 3) 시계열(트렌드/이벤트/변동성) feature
# =========================================================
def trend_features(x):
    x = _safe(x)
    n = len(x)
    if n < 8:
        return dict(slope=np.nan, curvature=np.nan, rank_corr=np.nan)

    t = np.arange(n, dtype=float)

    try:
        slope = float(np.polyfit(t, x, 1)[0])
    except Exception:
        slope = np.nan

    try:
        curvature = float(np.polyfit(t, x, 2)[0])
    except Exception:
        curvature = np.nan

    try:
        rx = pd.Series(x).rank(method="average").values
        rt = pd.Series(t).rank(method="average").values
        rank_corr = float(np.corrcoef(rt, rx)[0, 1])
    except Exception:
        rank_corr = np.nan

    return dict(slope=slope, curvature=curvature, rank_corr=rank_corr)

def diff_features(x):
    x = _safe(x)
    if len(x) < 6:
        return dict(dmean=np.nan, dstd=np.nan, dmax=np.nan, dmin=np.nan, zcr=np.nan)
    d = np.diff(x)
    sign = np.sign(d)
    zcr = np.mean(sign[1:] * sign[:-1] < 0) if len(sign) > 2 else np.nan
    return {
        "dmean": float(np.mean(d)),
        "dstd": float(np.std(d, ddof=1)) if len(d) > 2 else float(np.std(d)),
        "dmax": float(np.max(d)),
        "dmin": float(np.min(d)),
        "zcr": float(zcr),
    }

def autocorr_lag1(x):
    x = _safe(x)
    if len(x) < 6:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    if np.std(x0) == 0 or np.std(x1) == 0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])

def rolling_std_mean(x, win=10):
    x = _safe(x)
    if len(x) < win + 2:
        return np.nan
    s = pd.Series(x)
    r = s.rolling(win).std()
    return float(np.nanmean(r.values))

def peak_features(x, k=2.0):
    x = _safe(x)
    n = len(x)
    if n < 8:
        return dict(peak_cnt=np.nan, peak_mean=np.nan, peak_max=np.nan,
                    frac_above=np.nan, excess_auc=np.nan, auc=np.nan)

    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 2 else float(np.std(x))
    thr = mu + k * sd

    above = x > thr
    frac_above = float(np.mean(above))
    auc = float(np.sum(x))
    excess_auc = float(np.sum(np.maximum(0.0, x - thr)))

    peaks = []
    for i in range(1, n - 1):
        if x[i] > thr and x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append(x[i])

    if len(peaks) == 0:
        return dict(peak_cnt=0.0, peak_mean=0.0, peak_max=0.0,
                    frac_above=frac_above, excess_auc=excess_auc, auc=auc)

    peaks = np.asarray(peaks, dtype=float)
    return dict(
        peak_cnt=float(len(peaks)),
        peak_mean=float(np.mean(peaks)),
        peak_max=float(np.max(peaks)),
        frac_above=frac_above,
        excess_auc=excess_auc,
        auc=auc
    )


# =========================================================
# 6) 단일 파일 처리 (p10~p100 누적, VOC 채널 처리)
# =========================================================
def process_one_file(path: str):
    sample_id = os.path.splitext(os.path.basename(path))[0]

    # 시트 읽기
    try:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")
    except Exception:
        # 시트명이 다를 경우 첫 번째 시트 시도
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

    # 헤더 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    # 컬럼 체크
    if COL_VOC not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {COL_VOC} / 실제 컬럼: {list(df.columns)}")

    # 필요한 컬럼만 + 결측 제거
    df = df[[COL_VOC]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n < 50:
        raise ValueError(f"데이터가 너무 짧음: {n} rows")

    out = {"sample_id": sample_id, "n_points": int(n)}

    x_voc_all = df[COL_VOC].astype(float).values

    for pct in PCTS:
        end = int(np.floor(n * (pct / 100.0)))
        end = max(end, 10)
        
        # VOC 채널 데이터 슬라이싱
        x = x_voc_all[:end]
        tag = "VOC"

        # 특징량 추출
        bs = base_stats(x)
        ps = percentile_stats(x)
        tr = trend_features(x)
        dfc = diff_features(x)
        pk = peak_features(x, k=THRESH_K)
        ac1 = autocorr_lag1(x)
        rstd = rolling_std_mean(x, win=ROLL_WIN)

        out.update({f"{tag}_p{pct}_{k}": v for k, v in bs.items()})
        out.update({f"{tag}_p{pct}_{k}": v for k, v in ps.items()})
        out.update({f"{tag}_p{pct}_{k}": v for k, v in tr.items()})
        out.update({f"{tag}_p{pct}_{k}": v for k, v in dfc.items()})
        out.update({f"{tag}_p{pct}_{k}": v for k, v in pk.items()})
        out[f"{tag}_p{pct}_autocorr1"] = ac1
        out[f"{tag}_p{pct}_rollstd_mean"] = rstd

        # 추가 변동성 지표
        eps = 1e-12
        mu = float(np.mean(_safe(x))) if len(_safe(x)) else np.nan
        sd = float(np.std(_safe(x), ddof=1)) if len(_safe(x)) > 2 else float(np.std(_safe(x))) if len(_safe(x)) else np.nan
        out[f"{tag}_p{pct}_cv"] = float(sd / (abs(mu) + eps)) if np.isfinite(mu) and np.isfinite(sd) else np.nan
        out[f"{tag}_p{pct}_fano"] = float((sd**2) / (abs(mu) + eps)) if np.isfinite(mu) and np.isfinite(sd) else np.nan

    return out


# =========================================================
# 7) 전체 폴더 처리 -> 엑셀 저장
# =========================================================
def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.xls*")))
    # 결과 파일 등 "원시 데이터"가 아닌 엑셀은 제외(불필요한 FAIL 방지)
    out_basename = os.path.basename(OUT_XLSX).lower()
    filtered = []
    for p in paths:
        bn = os.path.basename(p).lower()
        if bn == out_basename:
            continue
        if bn.startswith("features_"):
            continue
        if "charts" in bn:
            continue
        filtered.append(p)
    paths = filtered
    if not paths:
        print(f"⚠️ 경고: '{DATA_DIR}' 폴더에 엑셀 파일이 없습니다.")
        return

    rows = []
    fails = []

    for p in paths:
        try:
            r = process_one_file(p)
            rows.append(r)
            print(f"OK   - {r['sample_id']} (n={r['n_points']})")
        except Exception as e:
            fails.append((os.path.basename(p), str(e)))
            print(f"FAIL - {os.path.basename(p)} / {e}")

    if not rows:
        print("❌ 추출된 데이터가 없습니다.")
        return

    feat_df = pd.DataFrame(rows)

    # 보기 좋게 앞쪽 정렬
    front = ["sample_id", "n_points"]
    front = [c for c in front if c in feat_df.columns]
    other = [c for c in feat_df.columns if c not in front]
    feat_df = feat_df[front + other]

    # 저장(실패 로그도 같이)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        feat_df.to_excel(writer, index=False, sheet_name="features")
        if fails:
            pd.DataFrame(fails, columns=["file", "error"]).to_excel(writer, index=False, sheet_name="fails")

    # Windows 콘솔(cp949)에서 이모지 출력이 깨질 수 있어 ASCII로만 출력
    print(f"\n[Saved] {OUT_XLSX}")
    print("shape:", feat_df.shape)
    if fails:
        print(f"[Warning] Failed files: {len(fails)} (check 'fails' sheet)")


if __name__ == "__main__":
    main()

