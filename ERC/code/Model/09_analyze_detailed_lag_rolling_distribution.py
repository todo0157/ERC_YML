import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_file = os.path.join(base_dir, "07_Comparative_Analysis_Roughness", "Top_Features_Roughness_DynamicK.xlsx")
output_dir = os.path.join(base_dir, "07_Comparative_Analysis_Roughness")

print(f"Loading features from {input_file}...")
try:
    df = pd.read_excel(input_file, sheet_name='Top_Features_List')
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Integrated 모델만 분석
df = df[df['Case'] == 'Integrated'].copy()

# ---------------------------------------------------------
# 상세 분류 함수 정의
# ---------------------------------------------------------
def classify_fine_grained(row):
    base_type = row['Type']
    feat_name = str(row['Feature'])
    
    has_lag = 'Lag' in feat_name
    has_roll = 'Roll' in feat_name
    
    # 1. VOC_Enhanced 세분화
    if base_type == 'VOC_Enhanced':
        if has_lag and has_roll:
            return 'VOC (Rolling+Lag)'
        elif has_roll:
            return 'VOC (Rolling Only)'
        elif has_lag:
            return 'VOC (Lag Only)'
        else:
            return 'VOC (Other)' # 예외 케이스
            
    # 2. Interaction 세분화
    elif base_type == 'Interaction':
        if has_lag and has_roll:
            return 'Interaction (Rolling+Lag)'
        elif has_roll:
            return 'Interaction (Rolling Only)'
        elif has_lag:
            return 'Interaction (Lag Only)'
        else:
            return 'Interaction (Sync)'
            
    # 3. 나머지는 그대로 (PM_Raw, VOC_Raw)
    else:
        return base_type

# 분류 적용
df['Fine_Type'] = df.apply(classify_fine_grained, axis=1)

# ---------------------------------------------------------
# 시각화 함수
# ---------------------------------------------------------
def plot_fine_distribution(model_name):
    subset = df[df['Model'] == model_name]
    
    # 구간별, 타입별 개수 집계
    counts = subset.groupby(['Segment_Pct', 'Fine_Type']).size().unstack(fill_value=0)
    
    # 정렬 순서 지정 (논리적 순서)
    desired_order = [
        'PM_Raw', 
        'VOC_Raw', 
        'VOC (Lag Only)', 'VOC (Rolling Only)', 'VOC (Rolling+Lag)',
        'Interaction (Sync)', 'Interaction (Lag Only)', 'Interaction (Rolling Only)', 'Interaction (Rolling+Lag)'
    ]
    # 실제 데이터에 있는 컬럼만 필터링
    cols_to_use = [c for c in desired_order if c in counts.columns]
    counts = counts[cols_to_use]
    
    # 비율로 변환 (Stacked 100%)
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    
    # 색상 팔레트 지정
    colors = {
        'PM_Raw': '#808080',            # Gray
        'VOC_Raw': '#C0C0C0',           # Silver
        
        'VOC (Lag Only)': '#87CEEB',    # SkyBlue
        'VOC (Rolling Only)': '#1E90FF',# DodgerBlue
        'VOC (Rolling+Lag)': '#000080', # Navy
        
        'Interaction (Sync)': '#F5DEB3',        # Wheat
        'Interaction (Lag Only)': '#FA8072',    # Salmon
        'Interaction (Rolling Only)': '#FF4500',# OrangeRed
        'Interaction (Rolling+Lag)': '#8B0000'  # DarkRed
    }
    # 현재 데이터에 있는 색상만 추출
    plot_colors = [colors[c] for c in cols_to_use]
    
    # Plot
    plt.figure(figsize=(14, 8))
    ax = counts_pct.plot(kind='bar', stacked=True, color=plot_colors, width=0.7, edgecolor='black', figsize=(14, 8))
    
    plt.title(f'Feature Distribution by Type (Lag vs Rolling) - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Process Percentage (%)', fontsize=12)
    plt.ylabel('Percentage of Top K Features (%)', fontsize=12)
    plt.legend(title='Detailed Feature Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 값 표시
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', fontsize=8, color='white', padding=3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"Feature_Distribution_Lag_vs_Rolling_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")

# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
models = df['Model'].unique()
for m in models:
    plot_fine_distribution(m)
    
# 데이터 저장 (확인용)
summary_file = os.path.join(output_dir, "Feature_Distribution_Lag_vs_Rolling_Summary.xlsx")
with pd.ExcelWriter(summary_file) as writer:
    for m in models:
        subset = df[df['Model'] == m]
        counts = subset.groupby(['Segment_Pct', 'Fine_Type']).size().unstack(fill_value=0)
        counts.to_excel(writer, sheet_name=m)
print(f"Summary data saved to {summary_file}")

