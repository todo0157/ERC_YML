
import pandas as pd
import os

# Define paths
base_dir = "ERC/Comparative_Analysis_FixedK"
input_file = os.path.join(base_dir, "FixedK_Analysis_Result_WithFeatures.xlsx")
output_file = os.path.join(base_dir, "Interaction_Feature_Analysis.xlsx")

# Load data
try:
    df = pd.read_excel(input_file, sheet_name='Feature_Details')
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit(1)

# Filter for Integrated case
df_integrated = df[df['Case'] == 'Integrated'].copy()

# Define interaction keywords
interaction_keywords = ['Prod', 'Ratio', 'Diff', 'Sum', 'Corr']

# Helper function to check if a feature is an interaction feature
def is_interaction(feature_name):
    # Ensure feature_name is a string to avoid errors
    if not isinstance(feature_name, str):
        return False
    return any(keyword in feature_name for keyword in interaction_keywords)

# Apply helper function
df_integrated['Is_Interaction'] = df_integrated['Feature_Name'].apply(is_interaction)

# --- Create Summary_Ratio Sheet ---
summary_data = []

# Group by Model and Percentage
for (model, percentage), group in df_integrated.groupby(['Model', 'Percentage']):
    total_features = len(group)
    interaction_count = group['Is_Interaction'].sum()
    ratio = interaction_count / total_features if total_features > 0 else 0
    
    summary_data.append({
        'Model': model,
        'Percentage': percentage,
        'Total_Features': total_features,
        'Interaction_Count': interaction_count,
        'Ratio': ratio
    })

df_summary = pd.DataFrame(summary_data)
# Sort for better readability
df_summary = df_summary.sort_values(by=['Model', 'Percentage'])


# --- Create Detailed_Interaction_Features Sheet ---
# Filter only interaction features
df_details = df_integrated[df_integrated['Is_Interaction']].copy()

# Select relevant columns
df_details = df_details[['Model', 'Percentage', 'Rank', 'Feature_Name', 'Importance']]

# Sort
df_details = df_details.sort_values(by=['Model', 'Percentage', 'Rank'])


# --- Save to Excel ---
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='Summary_Ratio', index=False)
    df_details.to_excel(writer, sheet_name='Detailed_Interaction_Features', index=False)

print(f"Successfully created {output_file}")
print("Summary preview:")
print(df_summary.head())

