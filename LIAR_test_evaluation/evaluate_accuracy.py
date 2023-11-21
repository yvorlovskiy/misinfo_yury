import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(df, category):
    # Convert 'label' and 'gpt-answer' columns to string type
    df['label'] = df['label'].astype(str)
    df['gpt-answer'] = df['gpt-answer'].astype(str)

    # Filter out rows where 'gpt-answer' is not '0.5' (as a string)
    confident_df = df[df['gpt-answer'].isin(['1.0', '0.0'])]

    # Extract label and predicted values
    y_true = confident_df['label'].str[0]
    y_pred = confident_df['gpt-answer'].str[0]

    # Calculate Macro F1 Score
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100

    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100

    print(f"Category {category}, Macro F1 Score: {macro_f1:.2f}%, Accuracy: {accuracy:.2f}%.")


def filter_confident(df):
    """Filter the DataFrame for confident answers."""
    return df[df['gpt-answer'].isin(['1.0', '0.0'])]

# Reading the data
df_no_context = pd.read_json('data/LIAR_new_uncertainty_quantification_hard_impossible_cases_no_context_gpt-4-0613_temp0.5.jsonl', lines=True)
df_context_results_1 = pd.read_json('data/corrected_merged_impossible_and_hard_cases_with_labels.jsonl', lines=True)


# Print dataframe (optional)


# Calculate Macro F1
calculate_metrics(df_no_context, 'no context')
calculate_metrics(df_context_results_1, 'politfact article]')

# Apply the filter
df_no_context_confident = filter_confident(df_no_context)
df_context_confident = filter_confident(df_context_results_1)

print('rows in politfact article]', len(df_context_confident))

print('rows in no context', len(df_no_context_confident))

common_statements = set(df_no_context_confident['statement']).intersection(df_context_confident['statement'])
print('prediction for both', len(common_statements))

# Step 2: Filter each DataFrame to keep only rows with common 'statement' values
df_no_context_confident_common = df_no_context_confident[df_no_context_confident['statement'].isin(common_statements)]
df_context_confident_common = df_context_confident[df_context_confident['statement'].isin(common_statements)]

# Perform an inner merge on the 'statement' column
merged_df = df_no_context_confident_common.merge(df_context_confident_common[['statement', 'gpt-answer']], on='statement', how='inner', suffixes=('_no_context', '_context_1'))

# Save the merged DataFrame to a CSV file
merged_df.to_csv('out/common_confident_predictions.csv', index=False)



df_context_confident_common.to_csv('out/context_common.csv')
df_no_context_confident_common.to_csv('out/no_context_common.csv')



calculate_metrics(df_no_context_confident_common, 'interesection, no context')
calculate_metrics(df_context_confident_common, 'interesection, politfact article]')
