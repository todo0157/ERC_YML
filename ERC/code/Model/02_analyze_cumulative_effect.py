import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pm_dir = os.path.join(base_dir, "PM_timeresampling")
voc_dir = os.path.join(base_dir, "VOC")
output_dir = os.path.join(base_dir, "Cumulative_Analysis")
plot_dir = os.path.join(output_dir, "Plots")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Parameters
ROLLING_WINDOWS = [10, 30, 60, 120, 180, 240, 300]  # Windows in seconds
LAG_RANGE = 300  # Shift range to check with rolling features

results = []

print(f"Starting Cumulative Effect Analysis...")

for i in range(1, 28):
    pm_file = os.path.join(pm_dir, f"data{i}_resampling.xlsx")
    voc_file = os.path.join(voc_dir, f"data{i}.xlsx")
    sample_id = f"data{i}"
    
    if os.path.exists(pm_file) and os.path.exists(voc_file):
        try:
            # Load
            df_pm = pd.read_excel(pm_file)
            df_voc = pd.read_excel(voc_file)
            
            # Columns
            pm_col = next((c for c in df_pm.columns if "0.3" in str(c)), df_pm.columns[0])
            voc_col = next((c for c in df_voc.columns if "VOC" in str(c).upper()), df_voc.columns[0])
            
            # Align
            min_len = min(len(df_pm), len(df_voc))
            pm_series = df_pm[pm_col].iloc[:min_len].reset_index(drop=True)
            voc_series = df_voc[voc_col].iloc[:min_len].reset_index(drop=True)
            
            sample_res = {'Sample_ID': sample_id}
            best_r_corr = 0
            best_r_win = 0
            best_r_lag = 0
            
            # Iterate through rolling windows
            for win in ROLLING_WINDOWS:
                # Calculate Rolling Mean of VOC
                # rolling(window=win).mean() computes mean of current and previous (win-1) values
                voc_rolling = voc_series.rolling(window=win, min_periods=1).mean()
                
                # Check correlation with Lag for this rolling feature
                lags = range(0, LAG_RANGE + 1) # Only positive lag (Past affects Future)
                corrs = []
                
                for lag in lags:
                    shifted_rolling = voc_rolling.shift(lag)
                    mask = ~np.isnan(pm_series) & ~np.isnan(shifted_rolling)
                    if np.sum(mask) > win:
                        c = np.corrcoef(pm_series[mask], shifted_rolling[mask])[0, 1]
                    else:
                        c = np.nan
                    corrs.append(c)
                
                # Find best lag for this window
                # We expect Negative correlation
                # Use min() because stronger relationship is more negative
                # If we want magnitude, use abs(). Let's stick to min for negative relationship.
                valid_corrs = [c for c in corrs if not np.isnan(c)]
                if valid_corrs:
                    # Find strongest NEGATIVE correlation
                    min_corr = np.min(valid_corrs)
                    min_idx = np.argmin(valid_corrs)
                    
                    sample_res[f'Win_{win}_BestCorr'] = min_corr
                    sample_res[f'Win_{win}_BestLag'] = lags[min_idx]
                    
                    if min_corr < best_r_corr: # Looking for lower (more negative) value
                        best_r_corr = min_corr
                        best_r_win = win
                        best_r_lag = lags[min_idx]
            
            sample_res['Overall_Best_Window'] = best_r_win
            sample_res['Overall_Best_Lag'] = best_r_lag
            sample_res['Overall_Best_Corr'] = best_r_corr
            
            results.append(sample_res)
            print(f"{sample_id}: Best Window={best_r_win}s, Lag={best_r_lag}s, Corr={best_r_corr:.4f}")
            
        except Exception as e:
            print(f"Error {sample_id}: {e}")

# Save
df_res = pd.DataFrame(results)
out_path = os.path.join(output_dir, "Cumulative_Analysis_Results.xlsx")
df_res.to_excel(out_path, index=False)
print(f"Saved to {out_path}")

