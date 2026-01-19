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

# 상세 타입 분류 함수
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

# 모델별로 데이터 분리
models = df_int['Model'].unique()
output_excel = os.path.join(output_dir, "All_Top_Features_Detailed_List.xlsx")

with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    for model in models:
        # 해당 모델의 데이터만 추출
        model_df = df_int[df_int['Model'] == model].copy()
        
        # 보기 좋게 정렬: Segment -> Rank 순
        model_df = model_df.sort_values(by=['Segment_Pct', 'Rank'])
        
        # 필요한 컬럼만 선택
        model_df = model_df[['Segment_Pct', 'Rank', 'Feature', 'Importance', 'Detailed_Type']]
        
        # 시트 이름 저장 (SVR, ExtraTrees)
        model_df.to_excel(writer, sheet_name=model, index=False)
        print(f"Sheet '{model}' added with {len(model_df)} rows.")

print(f"All Top features list saved to {output_excel}")

