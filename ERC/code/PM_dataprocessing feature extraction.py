import os
import glob
import numpy as np
import pandas as pd

# =========================================================
# 1) 설정(여기만 너 환경에 맞게 수정)
# =========================================================
DATA_DIR = r"C:\Users\Gyuman\Desktop\ERC Aerosol\PLA\PLA Aerosol data"  # raw 엑셀 27개 폴더
SHEET_NAME = "PM"   # 시트명 (다르면 "Sheet1" 등으로 변경)
COL_PM03 = "Num_0.3um"
COL_PM05 = "Num_0.5um"

# 누적 구간(10% 단위)
PCTS = list(range(10, 101, 10))

# 피크/threshold 관련
THRESH_K = 2.0  # threshold = mean + K*std
ROLL_WIN = 10   # rolling std 계산 창 크기(포인트 수)

# 출력 파일명
OUT_XLSX = os.path.join(DATA_DIR, "features_pm_rich_p10to100.xlsx")


# =========================================================
# 2) 기본 통계(기존 feature 포함)
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
    """
    시간축이 없으니 index(0..n-1)를 시간으로 간주
    slope(1차), curvature(2차항), rank_corr(스피어만 유사)
    """
    x = _safe(x)
    n = len(x)
    if n < 8:
        return dict(slope=np.nan, curvature=np.nan, rank_corr=np.nan)

    t = np.arange(n, dtype=float)

    # slope
    try:
        slope = float(np.polyfit(t, x, 1)[0])
    except Exception:
        slope = np.nan

    # curvature(2차항)
    try:
        curvature = float(np.polyfit(t, x, 2)[0])
    except Exception:
        curvature = np.nan

    # rank correlation (Spearman 유사) : rank(t) vs rank(x)
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
    # zero-crossing rate of diff sign (변동 방향이 얼마나 자주 바뀌는지)
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
    """
    매우 단순/안전한 peak 정의:
    threshold = mean + k*std 를 넘는 '로컬 최대' 개수
    """
    x = _safe(x)
    n = len(x)
    if n < 8:
        return dict(peak_cnt=np.nan, peak_mean=np.nan, peak_max=np.nan,
                    frac_above=np.nan, excess_auc=np.nan, auc=np.nan)

    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 2 else float(np.std(x))
    thr = mu + k * sd

    # above threshold fraction
    above = x > thr
    frac_above = float(np.mean(above))

    # AUC(시간간격 동일 가정)
    auc = float(np.sum(x))

    # excess AUC = sum(max(0, x - thr))
    excess_auc = float(np.sum(np.maximum(0.0, x - thr)))

    # local peaks
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
# 4) 두 채널 조합 feature (PM0.3 vs PM0.5)
# =========================================================
def cross_channel_features(x03, x05, k=2.0):
    x03 = _safe(x03)
    x05 = _safe(x05)
    n = min(len(x03), len(x05))
    if n < 8:
        return {}

    x03 = x03[:n]
    x05 = x05[:n]
    eps = 1e-12

    ratio = x03 / (x05 + eps)
    frac = x03 / (x03 + x05 + eps)
    diff = x03 - x05

    out = {}

    # 상관
    try:
        out["corr_03_05"] = float(np.corrcoef(x03, x05)[0, 1])
    except Exception:
        out["corr_03_05"] = np.nan

    # ratio / frac / diff 기본 통계(간단히)
    for name, arr in [("ratio", ratio), ("frac03", frac), ("diff03_05", diff)]:
        bs = base_stats(arr)
        ps = percentile_stats(arr)
        out.update({f"{name}_{k}": v for k, v in bs.items()})
        out.update({f"{name}_{k}": v for k, v in ps.items()})

    # 동시 스파이크(둘 다 mean+k*std 초과 비율)
    mu03, sd03 = float(np.mean(x03)), float(np.std(x03, ddof=1))
    mu05, sd05 = float(np.mean(x05)), float(np.std(x05, ddof=1))
    thr03 = mu03 + k * sd03
    thr05 = mu05 + k * sd05
    both = (x03 > thr03) & (x05 > thr05)
    out["simul_spike_frac"] = float(np.mean(both))

    return out


# =========================================================
# 5) 파일명 파싱 (층두께_노즐온도_속도)
# =========================================================
# =========================================================
# 6) 단일 파일 처리 (p10~p100 누적)
# =========================================================
def process_one_file(path: str):
    sample_id = os.path.splitext(os.path.basename(path))[0]

    # 시트 읽기: 시트명이 다르면 에러 -> 그때 SHEET_NAME 바꾸면 됨
    df = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")

    # 헤더 공백 제거(혹시 모를 상황 대비)
    df.columns = [str(c).strip() for c in df.columns]

    # 컬럼 체크
    for c in [COL_PM03, COL_PM05]:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {c} / 실제 컬럼: {list(df.columns)}")

    # 필요한 컬럼만 + 결측 제거
    df = df[[COL_PM03, COL_PM05]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n < 50:
        raise ValueError(f"데이터가 너무 짧음: {n} rows")

    out = {"sample_id": sample_id, "n_points": int(n)}

    x03_all = df[COL_PM03].astype(float).values
    x05_all = df[COL_PM05].astype(float).values

    for pct in PCTS:
        end = int(np.floor(n * (pct / 100.0)))
        end = max(end, 10)
        x03 = x03_all[:end]
        x05 = x05_all[:end]

        # PM03 / PM05 각각 feature (기존+신규 통합)
        for tag, x in [("PM03", x03), ("PM05", x05)]:
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

            # 변동성 지표 몇 개 더(가볍게)
            eps = 1e-12
            mu = float(np.mean(_safe(x))) if len(_safe(x)) else np.nan
            sd = float(np.std(_safe(x), ddof=1)) if len(_safe(x)) > 2 else float(np.std(_safe(x))) if len(_safe(x)) else np.nan
            out[f"{tag}_p{pct}_cv"] = float(sd / (abs(mu) + eps)) if np.isfinite(mu) and np.isfinite(sd) else np.nan
            out[f"{tag}_p{pct}_fano"] = float((sd**2) / (abs(mu) + eps)) if np.isfinite(mu) and np.isfinite(sd) else np.nan

        # 두 채널 조합 feature (ratio/corr/동시스파이크 등)
        cc = cross_channel_features(x03, x05, k=THRESH_K)
        out.update({f"CC_p{pct}_{k}": v for k, v in cc.items()})

    return out


# =========================================================
# 7) 전체 폴더 처리 -> 엑셀 저장
# =========================================================
def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.xls*")))
    if not paths:
        raise FileNotFoundError(f"엑셀 파일이 없음: {DATA_DIR}")

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

    feat_df = pd.DataFrame(rows)

    # 보기 좋게 앞쪽 정렬
    front = ["sample_id", "n_points"]
    front = [c for c in front if c in feat_df.columns]
    other = [c for c in feat_df.columns if c not in front]
    feat_df = feat_df[front + other]

    # 저장(실패 로그도 같이)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        feat_df.to_excel(writer, index=False, sheet_name="features")
        if fails:
            pd.DataFrame(fails, columns=["file", "error"]).to_excel(writer, index=False, sheet_name="fails")

    print(f"\n✅ 저장 완료: {OUT_XLSX}")
    print("shape:", feat_df.shape)
    if fails:
        print(f"⚠️ 실패 파일: {len(fails)}개 (엑셀 'fails' 시트 확인)")


if __name__ == "__main__":
    main()
