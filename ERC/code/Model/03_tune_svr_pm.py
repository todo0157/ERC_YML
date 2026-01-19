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
feature_dir = os.path.join(base_dir, "..", "Enhanced_Features_v2")
output_dir = os.path.join(base_dir, "..", "Comparative_Analysis_Enhanced_v2")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
TARGET_PCT = 90  # Focus on 90% segment for tuning
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3
TOP_K = 20  # Use fixed K=20 as found in previous analysis

print(f"Starting SVR Tuning for {TARGET_PCT}% Segment...")

# Load Data (Integrated only)
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

# Prepare X, y
y = data['Target_PM_Mean'].values
meta_cols = ['Segment_Pct', 'Target_PM_Mean', 'Sample_ID']
X = data.drop(columns=[c for c in meta_cols if c in data.columns]).select_dtypes(include=[np.number]).values

# Handle NaNs/Infs
X = np.where(np.isinf(X), np.nan, X)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Feature Selection (Simulate Fixed K Step)
# 1. Variance Threshold (Simple check)
var_mask = np.var(X, axis=0) > 0
X = X[:, var_mask]

# 2. SelectKBest (Top 20) to match the Fixed K scenario
selector = SelectKBest(f_regression, k=min(TOP_K, X.shape[1]))
X_selected = selector.fit_transform(X, y)

print(f"Data Shape: {X_selected.shape}")

# Define Scorer (Negative Mean Absolute Percentage Error for GridSearch)
# We want to minimize RE, but GridSearch maximizes score. So use neg_mean_absolute_percentage_error
# Note: sklearn's MAPE might differ slightly from our RE formula (epsilon handling).
# Let's define custom scorer.

def mean_relative_error(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe))

# Make scorer (greater_is_better=False because we want to minimize error)
re_scorer = make_scorer(mean_relative_error, greater_is_better=False)

# --- Baseline SVR (Current Settings) ---
print("\nEvaluating Baseline SVR (C=1.0, eps=0.1, StandardScaler)...")
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


# --- Tuned SVR Strategy ---

# Strategy A: Log Transform y + RobustScaler + GridSearch
print("\nEvaluating Tuned SVR (Log(y) + RobustScaler + GridSearch)...")

# 1. Log Transform y
y_log = np.log1p(y)

# 2. Robust Scaler
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X_selected)

# 3. Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.001, 0.01, 0.05, 0.1],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
}

# Use inner CV for GridSearch
grid = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    scoring='neg_mean_squared_error', # Optimized for MSE in log space usually works well
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Outer Loop Evaluation
scores_tuned = []
best_params_list = []

for i, (train_idx, test_idx) in enumerate(rkf.split(X_robust)):
    X_train, X_test = X_robust[train_idx], X_robust[test_idx]
    y_train_log, y_test = y_log[train_idx], y[test_idx] # Train on log, Test on original
    
    # Fit Grid Search
    grid.fit(X_train, y_train_log)
    best_model = grid.best_estimator_
    
    # Predict (Log space)
    pred_log = best_model.predict(X_test)
    # Inverse Transform
    pred = np.expm1(pred_log)
    
    re = mean_relative_error(y_test, pred)
    scores_tuned.append(re)
    
    if i == 0: # Print best params for the first fold only
        print(f"Fold 1 Best Params: {grid.best_params_}")

tuned_re = np.mean(scores_tuned)
print(f"Tuned Mean RE: {tuned_re:.4f}")

# --- Comparison Plot ---
plt.figure(figsize=(8, 6))
labels = ['Baseline (Standard, C=1.0)', 'Tuned (Robust, Log(y), GridSearch)']
values = [baseline_re, tuned_re]
colors = ['gray', 'blue']

bars = plt.bar(labels, values, color=colors)
plt.ylabel('Mean Relative Error (RE)')
plt.title(f'SVR Performance Improvement ({TARGET_PCT}% Segment)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.ylim(0, max(values) * 1.2)
save_path = os.path.join(output_dir, "SVR_Tuning_Comparison.png")
plt.savefig(save_path)
print(f"Saved comparison plot to: {save_path}")

