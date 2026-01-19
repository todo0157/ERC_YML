import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set Korean font for plots
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pm_dir = os.path.join(base_dir, "PM_timeresampling")
voc_dir = os.path.join(base_dir, "VOC")
output_dir = os.path.join(base_dir, "Lag_Analysis")
plot_dir = os.path.join(output_dir, "Plots")

# Create output directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Parameters
LAG_RANGE = 300  # Check lag from -300s to +300s

results = []

print(f"Starting Cross-Correlation Analysis (Lag: -{LAG_RANGE} ~ +{LAG_RANGE})...")

for i in range(1, 28):
    pm_file = os.path.join(pm_dir, f"data{i}_resampling.xlsx")
    voc_file = os.path.join(voc_dir, f"data{i}.xlsx")
    
    sample_id = f"data{i}"
    
    if os.path.exists(pm_file) and os.path.exists(voc_file):
        try:
            # Load data
            df_pm = pd.read_excel(pm_file)
            df_voc = pd.read_excel(voc_file)
            
            # Identify columns
            pm_col = next((c for c in df_pm.columns if "0.3" in str(c)), df_pm.columns[0])
            voc_col = next((c for c in df_voc.columns if "VOC" in str(c).upper()), df_voc.columns[0])
            
            # Align data lengths (truncate to min length)
            min_len = min(len(df_pm), len(df_voc))
            pm_series = df_pm[pm_col].iloc[:min_len].reset_index(drop=True)
            voc_series = df_voc[voc_col].iloc[:min_len].reset_index(drop=True)
            
            lags = range(-LAG_RANGE, LAG_RANGE + 1)
            correlations = []
            
            for lag in lags:
                # Shift VOC data
                # If lag > 0 (positive), VOC is shifted down (t -> t+k).
                # Correlation(PM[t], VOC[t-k]) means we are checking if past VOC affects current PM.
                # In pandas shift(k): index i gets value from i-k.
                # So df['A'].shift(k) at index i is the value that was at i-k.
                # This aligns Current PM with Past VOC.
                
                shifted_voc = voc_series.shift(lag)
                
                # Calculate correlation ignoring NaNs created by shift
                valid_mask = ~np.isnan(pm_series) & ~np.isnan(shifted_voc)
                
                if np.sum(valid_mask) > 10: # Ensure enough data points
                    corr = np.corrcoef(pm_series[valid_mask], shifted_voc[valid_mask])[0, 1]
                else:
                    corr = np.nan
                
                correlations.append(corr)
            
            # Find Best Lag (Max Absolute Correlation)
            # We want to find the strongest relationship, whether positive or negative.
            # However, user hypothesis is VOC -> PM, so we expect negative correlation?
            # Or positive? If VOC converts to PM, VOC goes down, PM goes up? 
            # -> That would be concurrent negative correlation.
            # Or "High VOC caused High PM later"? -> Positive correlation with lag?
            # Let's just find the strongest correlation.
            
            # Replace NaNs with 0 for max search
            corrs_no_nan = [c if not np.isnan(c) else 0 for c in correlations]
            
            # Find index of max absolute correlation
            best_idx = np.argmax(np.abs(corrs_no_nan))
            best_lag = lags[best_idx]
            best_corr = correlations[best_idx]
            zero_lag_corr = correlations[lags.index(0)]
            
            results.append({
                'Sample_ID': sample_id,
                'Best_Lag': best_lag,
                'Max_Correlation': best_corr,
                'Zero_Lag_Correlation': zero_lag_corr,
                'Improvement': abs(best_corr) - abs(zero_lag_corr) if not np.isnan(best_corr) and not np.isnan(zero_lag_corr) else 0
            })
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(lags, correlations, label='Correlation')
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Zero Lag')
            plt.axvline(x=best_lag, color='r', linestyle='--', alpha=0.8, label=f'Best Lag ({best_lag}s)')
            plt.scatter([best_lag], [best_corr], color='red', s=100)
            
            plt.title(f'Cross-Correlation: {sample_id} (PM vs VOC Shifted)\nBest Lag: {best_lag}s (Corr: {best_corr:.4f})')
            plt.xlabel('Lag (Seconds) - Positive means VOC precedes PM (VOC leads)')
            plt.ylabel('Pearson Correlation Coefficient')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_path = os.path.join(plot_dir, f"{sample_id}_lag_analysis.png")
            plt.savefig(plot_path)
            plt.close()
            
            print(f"{sample_id}: Best Lag = {best_lag}s, Corr = {best_corr:.4f} (Zero Lag: {zero_lag_corr:.4f})")
            
        except Exception as e:
            print(f"Error processing {sample_id}: {str(e)}")
            results.append({
                'Sample_ID': sample_id,
                'Best_Lag': None,
                'Max_Correlation': None,
                'Zero_Lag_Correlation': None,
                'Improvement': None,
                'Note': str(e)
            })

# Save Results to Excel
df_results = pd.DataFrame(results)
output_path = os.path.join(output_dir, "Lag_Analysis_Results.xlsx")
df_results.to_excel(output_path, index=False)

print(f"\nAnalysis Complete. Results saved to {output_path}")
print(f"Plots saved to {plot_dir}")

