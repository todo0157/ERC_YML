import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_dir = os.path.join(base_dir, "..", "Enhanced_Features_Roughness") # Changed path
output_dir = os.path.join(base_dir, "..", "Comparative_Analysis_Roughness") # New output dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
TARGET_PCT = 90
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3
TOP_K = 20 

print(f"Starting SVR Tuning for Roughness ({TARGET_PCT}% Segment)...")

# Load Data (Integrated)
try:
    df_int_all = pd.read_pickle(os.path.join(feature_dir, "Integrated.pkl"))
except Exception as e:
    print(f"Error loading pickle: {e}")
    exit()

# Filter Data
if 'Segment_Pct' in df_int_all.columns:
    data = df_int_all[df_int_all['Segment_Pct'] == TARGET_PCT].copy()
else:
    print("Segment_Pct column missing.")
    exit()

if data.empty:
    print(f"No data for {TARGET_PCT}% segment.")
    exit()

# Prepare X, y (Target_Roughness)
y = data['Target_Roughness'].values
meta_cols = ['Segment_Pct', 'Target_Roughness', 'Sample_ID']
X = data.drop(columns=[c for c in meta_cols if c in data.columns]).select_dtypes(include=[np.number]).values

# Handle NaNs
X = np.where(np.isinf(X), np.nan, X)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Feature Selection (Simulate Fixed K)
var_mask = np.var(X, axis=0) > 0
X = X[:, var_mask]
selector = SelectKBest(f_regression, k=min(TOP_K, X.shape[1]))
X_selected = selector.fit_transform(X, y)

print(f"Data Shape: {X_selected.shape}")

# RE Metric
def mean_relative_error(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe))

# --- Baseline SVR (Standard, C=1.0) ---
print("\nEvaluating Baseline SVR (Standard, C=1.0)...")
scaler_base = StandardScaler()
X_base = scaler_base.fit_transform(X_selected)
model_base = SVR(C=1.0, epsilon=0.1)
rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

scores_base = []
for train_idx, test_idx in rkf.split(X_base):
    model_base.fit(X_base[train_idx], y[train_idx])
    pred = model_base.predict(X_base[test_idx])
    scores_base.append(mean_relative_error(y[test_idx], pred))
baseline_re = np.mean(scores_base)
print(f"Baseline Mean RE: {baseline_re:.4f}")

# --- Tuning Strategy 1: RobustScaler + GridSearch (No Log) ---
print("\nEvaluating Strategy 1 (Robust + GridSearch, No Log)...")
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X_selected)

param_grid = {
    'C': [1, 10, 100, 1000, 5000],
    'epsilon': [0.01, 0.1, 1.0, 5.0, 10.0], # Roughness values are larger (hundreds), so eps can be larger
    'gamma': ['scale', 'auto', 0.1, 0.01]
}

grid1 = GridSearchCV(SVR(), param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)

scores_str1 = []
for train_idx, test_idx in rkf.split(X_robust):
    grid1.fit(X_robust[train_idx], y[train_idx])
    pred = grid1.best_estimator_.predict(X_robust[test_idx])
    scores_str1.append(mean_relative_error(y[test_idx], pred))
str1_re = np.mean(scores_str1)
print(f"Strategy 1 Mean RE: {str1_re:.4f} (Best Params: {grid1.best_params_})")

# --- Tuning Strategy 2: RobustScaler + Log(y) + GridSearch ---
print("\nEvaluating Strategy 2 (Robust + Log(y) + GridSearch)...")
y_log = np.log1p(y)

# For Log space, epsilon should be small again
param_grid_log = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.001, 0.01, 0.05, 0.1], 
    'gamma': ['scale', 'auto', 0.1, 0.01]
}
grid2 = GridSearchCV(SVR(), param_grid_log, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)

scores_str2 = []
for train_idx, test_idx in rkf.split(X_robust):
    grid2.fit(X_robust[train_idx], y_log[train_idx])
    pred_log = grid2.best_estimator_.predict(X_robust[test_idx])
    pred = np.expm1(pred_log)
    scores_str2.append(mean_relative_error(y[test_idx], pred))
str2_re = np.mean(scores_str2)
print(f"Strategy 2 Mean RE: {str2_re:.4f} (Best Params: {grid2.best_params_})")

# --- Comparison Plot ---
plt.figure(figsize=(10, 6))
labels = ['Baseline', 'Robust+NoLog', 'Robust+Log']
values = [baseline_re, str1_re, str2_re]

bars = plt.bar(labels, values, color=['gray', 'orange', 'blue'])
plt.ylabel('Mean Relative Error (RE)')
plt.title(f'SVR Tuning for Roughness ({TARGET_PCT}%)')
plt.ylim(0, max(values)*1.2)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')

save_path = os.path.join(output_dir, "SVR_Tuning_Roughness.png")
plt.savefig(save_path)
print(f"Saved plot to {save_path}")

