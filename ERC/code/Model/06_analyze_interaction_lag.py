import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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

# 통합 모델의 Interaction 피처만 필터링
# Case 컬럼이 'Integrated'이고 Type이 'Interaction'인 것
df_int = df[(df['Case'] == 'Integrated') & (df['Type'] == 'Interaction')].copy()

if df_int.empty:
    print("No interaction features found for Integrated case.")
    # 빈 파일이라도 생성하여 에러 방지
    exit()

# Lag vs Sync 분류 함수
def classify_interaction(feature_name):
    # 'Lag' 문자열이 포함되어 있으면 시간 차가 있는 변수와의 상호작용
    if 'Lag' in str(feature_name):
        return 'Time-Lagged'
    else:
        return 'Sync (Immediate)'

df_int['Sub_Type'] = df_int['Feature'].apply(classify_interaction)

# 구간별, 모델별, 서브타입별 집계
summary = df_int.groupby(['Segment_Pct', 'Model', 'Sub_Type']).size().reset_index(name='Count')

# Pivot table 생성 (구간/모델 별로 Time-Lagged와 Sync 개수 나열)
pivot_df = summary.pivot_table(index=['Segment_Pct', 'Model'], 
                             columns='Sub_Type', 
                             values='Count', 
                             fill_value=0).reset_index()

# 컬럼이 없을 경우 0으로 채움
if 'Time-Lagged' not in pivot_df.columns:
    pivot_df['Time-Lagged'] = 0
if 'Sync (Immediate)' not in pivot_df.columns:
    pivot_df['Sync (Immediate)'] = 0

# 총합 및 비율 계산
pivot_df['Total_Interaction'] = pivot_df['Time-Lagged'] + pivot_df['Sync (Immediate)']

# 0으로 나누기 방지
pivot_df['Lag_Ratio'] = pivot_df.apply(
    lambda row: row['Time-Lagged'] / row['Total_Interaction'] if row['Total_Interaction'] > 0 else 0, axis=1
)
pivot_df['Sync_Ratio'] = pivot_df.apply(
    lambda row: row['Sync (Immediate)'] / row['Total_Interaction'] if row['Total_Interaction'] > 0 else 0, axis=1
)

# 엑셀 저장
output_excel = os.path.join(output_dir, "Interaction_Lag_Analysis.xlsx")
pivot_df.to_excel(output_excel, index=False)
print(f"Analysis saved to {output_excel}")

# 시각화 (SVR, ExtraTrees)
models = pivot_df['Model'].unique()

for model in models:
    model_data = pivot_df[pivot_df['Model'] == model]
    
    if model_data.empty:
        continue
        
    plt.figure(figsize=(12, 6))
    
    segments = model_data['Segment_Pct']
    lagged = model_data['Time-Lagged']
    sync = model_data['Sync (Immediate)']
    
    # 누적 막대 그래프 그리기
    # 1. Time-Lagged (하단)
    p1 = plt.bar(segments, lagged, label='Time-Lagged (Async)', color='#7B68EE', alpha=0.8, width=6)
    # 2. Sync (상단)
    p2 = plt.bar(segments, sync, bottom=lagged, label='Sync (Immediate)', color='#FF8C00', alpha=0.8, width=6)
    
    plt.xlabel('Process Segment (%)', fontsize=12)
    plt.ylabel('Number of Interaction Features', fontsize=12)
    plt.title(f'Interaction Feature Breakdown: Time-Lagged vs Sync ({model})\nIntegrated Model (Dynamic K)', fontsize=14)
    plt.xticks(segments, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 비율 텍스트 추가
    for i, (idx, row) in enumerate(model_data.iterrows()):
        total = row['Total_Interaction']
        if total > 0:
            # Lagged 비율 표시 (보라색 막대 중간)
            if row['Time-Lagged'] > 0:
                plt.text(row['Segment_Pct'], row['Time-Lagged']/2, 
                         f"{row['Lag_Ratio']*100:.0f}%", 
                         ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
            # Sync 비율 표시 (주황색 막대 중간)
            if row['Sync (Immediate)'] > 0:
                plt.text(row['Segment_Pct'], row['Time-Lagged'] + row['Sync (Immediate)']/2, 
                         f"{row['Sync_Ratio']*100:.0f}%", 
                         ha='center', va='center', color='black', fontweight='bold', fontsize=9)
    
    output_png = os.path.join(output_dir, f"Interaction_Breakdown_{model}.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_png}")
    plt.close()

