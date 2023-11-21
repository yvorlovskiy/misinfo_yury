import pandas as pd

# Read data from JSON file
df = pd.read_json('data/classification_LIAR_test_categories_12567_gpt-4-0613_temp0.5.jsonl', lines = True)

# Mapping dictionary
dct = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7
}

# Apply mapping to the 'gpt-answer' column


df['gpt-answer-category'] = df['gpt-answer_y']
df['gpt-answer-category'] = df['gpt-answer-category'].apply(lambda x: x[:1])
df['gpt-answer-category'] = df['gpt-answer-category'].apply(lambda x: dct.get(x, x))

df['label'] = (df['label'] > 2).astype(int)
# Filter and save to CSV based on conditions

df.to_csv('out/full_df.csv')

df.drop('gpt-answer_x', axis=1, inplace=True)
df.drop('gpt-answer_y', axis=1, inplace=True)
df.drop('gpt-message_x', axis=1, inplace=True)
df.drop('gpt-message_y', axis=1, inplace=True)




df[df['gpt-answer-category'] == 1].to_csv('out/df_1.csv', index=False)
df[df['gpt-answer-category'] == 2].to_csv('out/df_2.csv', index=False)
df[df['gpt-answer-category'] == 6].to_csv('out/df_6.csv', index=False)
df[df['gpt-answer-category'] == 5].to_csv('out/df_5.csv', index=False)
df[df['gpt-answer-category'] == 7].to_csv('out/df_7.csv', index=False)

df[df['gpt-answer-category'] == 1].to_json('out/LIAR_test_category_1.jsonl', orient='records', lines=True)
df[df['gpt-answer-category'] == 2].to_json('out/LIAR_test_category_2.jsonl', orient='records', lines=True)
df[df['gpt-answer-category'] == 6].to_json('out/LIAR_test_category_6.jsonl', orient='records', lines=True)
df[df['gpt-answer-category'] == 5].to_json('out/LIAR_test_category_5.jsonl', orient='records', lines=True)
df[df['gpt-answer-category'] == 7].to_json('out/LIAR_test_category_7.jsonl', orient='records', lines=True)