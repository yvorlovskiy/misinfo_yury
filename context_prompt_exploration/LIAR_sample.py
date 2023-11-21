import pandas as pd 
import json 
import csv


def sample_rows(df, n):
    # Ensure the input n is valid
    if n <= 0:
        raise ValueError("n should be a positive integer")
    
    # Split the dataframe based on the 'possibility' column values 'i' and 'p'
    df_i = df[df['possibility'] == 'i']
    df_p = df[df['possibility'] == 'p']
    
    # Randomly sample n rows from each subset
    sample_i = df_i.sample(n, replace=False)  # replace=True allows re-sampling
    sample_p = df_p.sample(n, replace=True)
    
    # Concatenate the samples and return the result
    result_df = pd.concat([sample_i, sample_p])
    return result_df

def sample_rows_excel(file_path, possibility, n):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Define a function to convert a string to lowercase only if it's not empty
    def to_lower(s):
        if pd.notna(s):
            return s.lower()
        return s
    
    # Convert the 'possibility' column to lowercase if not empty and filter rows
    filtered_df = df[df['new_possibility'].apply(to_lower) == possibility.lower()]
    
    # Randomly sample n rows from the filtered DataFrame
    sampled_rows = filtered_df.sample(n, random_state=1)

    sampled_rows.to_csv('LIAR_new_samples_for_categorization_possible.csv', index=False)

def sample_by_case(df, cases):
    df['case'] = df['new_possibility_yury'] + df['new_possibility_Anne'] + df['new_possibility_Camille']
    df['case'] = df['case'].str.lower()
    filtered_df = df[df['case'].isin(cases)]
    return filtered_df

def csv_to_jsonl(csv_filename, jsonl_filename):
    with open(csv_filename, 'r') as csv_file, open(jsonl_filename, 'w') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            jsonl_file.write(json.dumps(row) + '\n')




df = pd.read_csv('politfact.csv')
#sample_rows_excel('data/LIAR_new_three_labels.xlsx', 'i', 97)

#csv_to_jsonl('LIAR_new_samples_for_categorization_possible.csv', 'LIAR_new_samples_for_categorization_possible.jsonl')

impossible_cases = sample_by_case(df, ['iii', 'iih', 'ihi', 'ihh', 'hii', 'hih', 'hhi', 'hhh'])

impossible_cases.to_json('impossible_and_hard_cases.jsonl',  orient='records', lines = True)
print(impossible_cases)
print(len(impossible_cases))
