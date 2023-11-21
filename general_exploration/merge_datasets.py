import pandas as pd 
import json
import re
# Initialize an empty list to store the parsed rows
data = []

def extract_statement(content):
    """
    Extracts the statement from the content string, strips it of all quotation marks, 
    and removes the instruction part.
    """
    # Extract the statement using regex
    pattern_imps = r'Consider the following statement:\s*(.*?)(?:\s*Provide letter I if impossible,)'
    pattern_judge = r''
    match = re.search(pattern_imps, content, re.DOTALL)
    if match:
        statement = match.group(1).strip()
    else:
        statement = content  # Fallback if regex fails
    
    # Remove all types of quotation marks
    statement = statement.replace('“', '').replace('”', '').replace('"', '')
    
    return statement

# Open the JSONL file and parse each line
with open('data/LIAR_New_impossibility_judgement_gpt-4-0613_temp0.5_rawoutput.jsonl', 'r') as file:
    for line in file:
        # Load the JSON data from the line
        json_line = json.loads(line)
        
        # Extract the required data
        row = {
            'id': json_line[0],
            'text': extract_statement(json_line[1]['messages'][0]['content']),
            'temperature': json_line[1]['temperature'],
            'model': json_line[1]['model'],
            'max_tokens': json_line[1]['max_tokens'],
            'gpt-message': json_line[2]['choices'][0]['message']['role'],
            'gpt_answer': json_line[2]['choices'][0]['message']['content']
        }
        
        # Append the extracted row to the data list
        data.append(row)

# Convert the list of dictionaries to a DataFrame
experiment_df = pd.DataFrame(data)



experiment_df.to_csv('out/impossibility_judgement_experiment_df.csv', index = False)
experiment_df['gpt-answer'] = experiment_df.gpt_answer.apply(lambda x: x.lower() if type(x) == str else x)
impossiblity_df = pd.read_csv('data/politifact_new_labeling.tsv', sep='\t', quoting=3).apply(lambda x: x.str.replace('["“”]', '') if x.dtype == 'object' else x).assign(possibility=lambda df: df.possibility.str.lower())
impossiblity_df['possibility'] = impossiblity_df.possibility.apply(lambda x: x.lower() if type(x) == str else x)



combined_df = experiment_df.merge(impossiblity_df, how='left', left_on='text', right_on='statement', suffixes=('', '_impossibility'))
impossiblity_df = combined_df[combined_df.possibility == 'i']
possible_and_unsure_df = combined_df[(combined_df['gpt-answer'] == '0.5') & (combined_df['possibility'] == 'p')]




impossiblity_df.to_csv('out/impossibility_df.csv', index = False)
possible_and_unsure_df.to_csv('out/possible_and_unsure_df.csv', index = False)
combined_df.to_csv('out/impossibility_judgement_combined_df.csv', index = False)

def evaluate_accuracy(df):
    # Check for correct predictions
    df['correct'] = df['gpt-answer'] == df['possibility']
    
    # Overall accuracy
    total_accuracy = df['correct'].mean()

    # Accuracy for possible statements
    possible_df = df[df['possibility'] == 'p']
    if not possible_df.empty:
        possible_accuracy = possible_df['correct'].mean()
    else:
        possible_accuracy = None

    # Accuracy for impossible statements
    impossible_df = df[df['possibility'] == 'i']
    if not impossible_df.empty:
        impossible_accuracy = impossible_df['correct'].mean()
    else:
        impossible_accuracy = None

    return total_accuracy, possible_accuracy, impossible_accuracy

total_accuracy, possible_accuracy, impossible_accuracy = evaluate_accuracy(combined_df)

print(f"Total Accuracy: {total_accuracy*100:.2f}%")
if possible_accuracy is not None:
    print(f"Accuracy on Possible Statements: {possible_accuracy*100:.2f}%")
else:
    print("No possible statements in the dataset.")

if impossible_accuracy is not None:
    print(f"Accuracy on Impossible Statements: {impossible_accuracy*100:.2f}%")
else:
    print("No impossible statements in the dataset.")


