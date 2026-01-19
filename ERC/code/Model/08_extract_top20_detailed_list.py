import pandas as pd
import os

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

# Integrated 모델만 필터링
df_int = df[df['Case'] == 'Integrated'].copy()

# 상세 타입 분류 함수 (이전과 동일)
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

df_int['Detailed_Type'] = df_int.apply(classify_detailed_type, axis=1)

# 결과를 저장할 리스트
result_frames = []

# 각 구간(Segment)과 모델(Model)별로 상위 20개 추출
segments = sorted(df_int['Segment_Pct'].unique())
models = df_int['Model'].unique()

for model in models:
    for seg in segments:
        # 해당 모델, 해당 구간 데이터 필터링
        subset = df_int[(df_int['Model'] == model) & (df_int['Segment_Pct'] == seg)].copy()
        
        # 중요도(Importance) 기준 내림차순 정렬 (이미 되어있을 수 있지만 확실하게)
        subset = subset.sort_values(by='Rank', ascending=True)
        
        # 상위 20개만 자르기 (또는 전체가 20개 안되면 전체)
        top20 = subset.head(20).copy()
        
        # 필요한 컬럼만 선택 및 정렬
        top20 = top20[['Segment_Pct', 'Model', 'Rank', 'Feature', 'Importance', 'Detailed_Type']]
        result_frames.append(top20)

# 전체 결과 병합
if result_frames:
    final_df = pd.concat(result_frames, ignore_index=True)
    
    # 엑셀 저장
    output_excel = os.path.join(output_dir, "Top20_Features_Detailed_List.xlsx")
    final_df.to_excel(output_excel, index=False)
    print(f"Top 20 features list saved to {output_excel}")
else:
    print("No data found to extract.")

