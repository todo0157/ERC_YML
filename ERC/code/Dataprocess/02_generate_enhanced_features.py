import pandas as pd
import numpy as np
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pm_dir = os.path.join(base_dir, "PM_timeresampling")
voc_dir = os.path.join(base_dir, "VOC")
output_dir = os.path.join(base_dir, "Enhanced_Data")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters determined from analysis
# We will create features around the "Best" range (4-5 minutes lag, 3-4 minutes window)
# To be robust, we will generate a set of features.

LAGS = [240, 300]         # 4 min, 5 min (Shift only)
WINDOWS = [180, 240]      # 3 min, 4 min (Rolling only)
COMBINATIONS = [          # (Lag, Window) combinations
    (240, 180),           # Lag 4m, Win 3m
    (240, 240),           # Lag 4m, Win 4m
    (300, 180),           # Lag 5m, Win 3m
    (300, 240)            # Lag 5m, Win 4m
]

print("Starting Enhanced Feature Generation...")

for i in range(1, 28):
    pm_file = os.path.join(pm_dir, f"data{i}_resampling.xlsx")
    voc_file = os.path.join(voc_dir, f"data{i}.xlsx")
    
    if os.path.exists(pm_file) and os.path.exists(voc_file):
        try:
            # Load Data
            df_pm = pd.read_excel(pm_file)
            df_voc = pd.read_excel(voc_file)
            
            # Identify columns
            pm_col = next((c for c in df_pm.columns if "0.3" in str(c)), df_pm.columns[0])
            voc_col = next((c for c in df_voc.columns if "VOC" in str(c).upper()), df_voc.columns[0])
            
            # 1. Align Basic Data (PM and VOC)
            # Truncate to shorter length
            min_len = min(len(df_pm), len(df_voc))
            df_enhanced = pd.DataFrame()
            
            # Base Target and Features
            df_enhanced['Time_Index'] = range(min_len)
            df_enhanced['Num_0.3um'] = df_pm[pm_col].iloc[:min_len].values
            df_enhanced['VOC_Raw'] = df_voc[voc_col].iloc[:min_len].values
            
            # To perform shift/rolling correctly, we need the full VOC series first, 
            # then truncate AFTER processing to match PM's indices.
            # But here, we assume PM[t] corresponds to VOC[t] physically (recorded at same time).
            # So calculating features on the full VOC and then truncating is correct.
            
            voc_series = df_voc[voc_col] # Use full VOC data for calculation
            
            # 2. Generate Lag Features (Shifted VOC)
            for lag in LAGS:
                # shift(lag): value at t comes from t-lag.
                feat_name = f"VOC_Lag_{lag}s"
                shifted = voc_series.shift(lag)
                df_enhanced[feat_name] = shifted.iloc[:min_len].values
            
            # 3. Generate Rolling Features (Moving Average)
            for win in WINDOWS:
                feat_name = f"VOC_Roll_{win}s"
                # rolling mean of current and past (win-1)
                rolling = voc_series.rolling(window=win, min_periods=1).mean()
                df_enhanced[feat_name] = rolling.iloc[:min_len].values
            
            # 4. Generate Combination Features (Shifted Rolling Mean)
            # "Mean of 3 mins, from 4 mins ago"
            for lag, win in COMBINATIONS:
                feat_name = f"VOC_Roll_{win}s_Lag_{lag}s"
                # First roll, then shift
                rolling = voc_series.rolling(window=win, min_periods=1).mean()
                shifted_rolling = rolling.shift(lag)
                df_enhanced[feat_name] = shifted_rolling.iloc[:min_len].values

            # 5. Clean up NaNs
            # Shifting introduces NaNs at the beginning.
            # Options: Drop rows, or Fill (Backfill/Forwardfill).
            # Since we need to match PM data length for later analysis, 
            # we should be careful. 
            # If we drop rows, we lose PM targets.
            # Strategy: Fill NaNs with the first valid observation (Backfill) 
            # to preserve data length, or fill with 0/Mean.
            # Given these are time series, Backfill (bfill) is reasonable for the start.
            
            df_enhanced = df_enhanced.fillna(method='bfill').fillna(method='ffill')
            
            # Handle any remaining NaNs (if full column is NaN) by filling with 0
            df_enhanced = df_enhanced.fillna(0)

            # Save
            output_file = os.path.join(output_dir, f"data{i}_enhanced.xlsx")
            df_enhanced.to_excel(output_file, index=False)
            
            if i % 5 == 0:
                print(f"Processed up to data{i}...")
                
        except Exception as e:
            print(f"Error processing data{i}: {e}")

print(f"Enhanced data generation complete. Saved to {output_dir}")
print("Features created:")
print(f"- Lags: {LAGS}")
print(f"- Rolling Windows: {WINDOWS}")
print(f"- Combinations: {COMBINATIONS}")

