import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Feature Type 세분화 함수
def classify_detailed_type(row):
    base_type = row['Type']
    feature_name = row['Feature']
    
    if base_type == 'Interaction':
        if 'Lag' in str(feature_name):
            return 'Interaction (Time-Lagged)'
        else:
            return 'Interaction (Sync)'
    else:
        return base_type

# 상세 타입 분류 적용
df['Detailed_Type'] = df.apply(classify_detailed_type, axis=1)

# 구간별, 모델별, 상세 타입별 집계
summary = df.groupby(['Segment_Pct', 'Model', 'Detailed_Type']).size().reset_index(name='Count')

# Pivot Table 생성 (각 구간/모델 별로 5개 타입의 개수)
pivot_df = summary.pivot_table(index=['Segment_Pct', 'Model'], 
                             columns='Detailed_Type', 
                             values='Count', 
                             fill_value=0).reset_index()

# 컬럼 순서 지정 (그래프 쌓는 순서 고려)
desired_order = ['PM_Raw', 'VOC_Raw', 'VOC_Enhanced', 'Interaction (Sync)', 'Interaction (Time-Lagged)']
# 실제 데이터에 존재하는 컬럼만 선택
existing_cols = [c for c in desired_order if c in pivot_df.columns]
pivot_df = pivot_df[['Segment_Pct', 'Model'] + existing_cols]

# 엑셀 저장
output_excel = os.path.join(output_dir, "Feature_Distribution_Detailed.xlsx")
pivot_df.to_excel(output_excel, index=False)
print(f"Detailed analysis saved to {output_excel}")

# 시각화 함수
def plot_detailed_distribution(model_name, data, columns):
    plt.figure(figsize=(14, 7))
    
    segments = data['Segment_Pct']
    bottom = pd.Series([0] * len(data), index=data.index)
    
    # 색상 지정
    colors = {
        'PM_Raw': '#1f77b4',       # 파랑
        'VOC_Raw': '#2ca02c',      # 초록
        'VOC_Enhanced': '#98df8a', # 연두
        'Interaction (Sync)': '#ff7f0e',        # 주황
        'Interaction (Time-Lagged)': '#9467bd'  # 보라 (강조)
    }
    
    for col in columns:
        values = data[col]
        plt.bar(segments, values, bottom=bottom, label=col, color=colors.get(col, '#333333'), width=6, alpha=0.9)
        
        # 비율 텍스트 표시 (값이 있을 때만)
        for i, val in enumerate(values):
            if val > 0:
                total = data.iloc[i][columns].sum()
                pct = (val / total) * 100
                if pct >= 5:  # 5% 이상일 때만 표시
                    y_pos = bottom.iloc[i] + val / 2
                    plt.text(segments.iloc[i], y_pos, f"{pct:.0f}%", 
                             ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        
        bottom += values

    plt.xlabel('Process Segment (%)', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title(f'Feature Type Distribution (Detailed): {model_name}\nIntegrated Model (Roughness Prediction)', fontsize=15)
    plt.xticks(segments, fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    output_png = os.path.join(output_dir, f"Feature_Distribution_Detailed_{model_name}.png")
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")
    plt.close()

# 모델별 그래프 생성
models = pivot_df['Model'].unique()
for model in models:
    model_data = pivot_df[pivot_df['Model'] == model]
    plot_detailed_distribution(model, model_data, existing_cols)

