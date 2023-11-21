import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def display_confusion_matrices(df, ground_truth_columns, categories = 7):
    # Convert None values to "NoneLabel" placeholder
    df = df.fillna("NoneLabel")
    
    # Adjust labels list to include the placeholder
    labels_list = [str(i) for i in range(1, categories + 1)] + ["NoneLabel"]
    
    # Initialize total confusion matrix
    cm_total = np.zeros((categories + 1, categories + 1))
    
    # Calculate the aggregated confusion matrix using your method
    for i, gt_label in enumerate(labels_list):
        for j, pred_label in enumerate(labels_list):
            count = df[df[ground_truth_columns].astype(str).apply(lambda x: x.str.contains(gt_label)).any(axis=1) & df['gpt-answer'].astype(str).str.contains(pred_label)].shape[0]
            cm_total[i, j] = count
    
    # Display individual confusion matrices for each column
    for column in ground_truth_columns:
        cm_column = np.zeros((categories + 1, categories + 1))
        
        for i, gt_label in enumerate(labels_list):
            for j, pred_label in enumerate(labels_list):
                count = df[df[column].astype(str).str.contains(gt_label) & df['gpt-answer'].astype(str).str.contains(pred_label)].shape[0]
                cm_column[i, j] = count
        
        # Display the confusion matrix for the current column
        disp_column = ConfusionMatrixDisplay(cm_column, display_labels=labels_list)
        disp_column.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {column}")
        plt.show()
    
    # Display the aggregated confusion matrix
    disp_total = ConfusionMatrixDisplay(cm_total, display_labels=labels_list)
    disp_total.plot(cmap=plt.cm.Blues)
    plt.title("Aggregated Confusion Matrix")
    plt.show()



def print_normalized_confusion_matrix(matrix, category):
    # Normalize the matrix across columns
    col_sums = matrix.sum(axis=0)
    normalized_matrix = matrix.astype('float') / col_sums[np.newaxis, :]
    normalized_matrix = np.round(normalized_matrix, 2)  # round to 2 decimal places

    # Extract normalized values
    TN, FP, FN, TP = normalized_matrix.ravel()
    
    # Print the matrix with formatting
    print(f"Normalized Confusion Matrix for Category {category}:")
    print("-" * 50)
    print("\t\tPredicted: POSITIVE\tPredicted: NEGATIVE")
    print(f"Actual: POSITIVE\tTP: {TP}\tFN: {FN}")
    print(f"Actual: NEGATIVE\tFP: {FP}\tTN: {TN}")
    print("-" * 50)
    print()


def evaluate_binary_classification(df: pd.DataFrame, cat: int) -> dict:
    condition = df[['category_yury', 'missing_info_Anne', 'missing_info_Camille']].eq(cat).any(axis=1)
    total_positives = condition.sum()
    total_negatives = (~condition).sum()

    true_positives = df[condition & (df['gpt-answer'] == 1)].shape[0]
    false_positives = df[~condition & (df['gpt-answer'] == 1)].shape[0]
    true_negatives = df[~condition & (df['gpt-answer'] == 0)].shape[0]
    false_negatives = df[condition & (df['gpt-answer'] == 0)].shape[0]

    print(f"Normalized TP: {true_positives / total_positives if total_positives else 0}")
    print(f"Normalized FP: {false_positives / total_negatives if total_negatives else 0}")
    print(f"Normalized TN: {true_negatives / total_negatives if total_negatives else 0}")
    print(f"Normalized FN: {false_negatives / total_positives if total_positives else 0}")

    return {
        "TP": true_positives,
        "FP": false_positives,
        "TN": true_negatives,
        "FN": false_negatives
    }


def cat_accuracy_prediction_sample(df, columns_to_check, categories = 7):
    df['accuracy'] = df.apply(lambda row : str(row['gpt-answer']) in str(row['category_yury']) or  str(row['gpt-answer']) in str(row['missing_info_Anne']) or str(row['gpt-answer']) in str(row['missing_info_Camille']), axis = 1)
    print('accuracy:', df['accuracy'].sum() / (len(df) - 1))
    predictions = []
    for i in range(1, categories + 1):
        # Convert the integer i to its string representation
        str_i = str(i)
        
        # Filter rows based on category value in the predicted column ('gpt-answer')
        mask = df['gpt-answer'].astype(str).str.contains(str_i)
        cat = df[mask]
        predictions.append(cat)
        
        # Calculate and print accuracy
        accuracy = cat['accuracy'].sum() / (len(cat))
        print(f'prediction sample accuracy category {i}: {accuracy:.2f} with {len(cat)} rows')
    return predictions

def cat_accuracy_label_sample(df, columns_to_check, categories = 7):
    df['accuracy'] = df.apply(lambda row : str(row['gpt-answer']) in str(row['category_yury']) or  str(row['gpt-answer']) in str(row['missing_info_Anne']) or str(row['gpt-answer']) in str(row['missing_info_Camille']), axis = 1)
    print('accuracy:', df['accuracy'].sum() / (len(df) - 1))
    for i in range(1, categories + 1):
        # Convert the integer i to its string representation
        str_i = str(i)
        
        # Filter rows based on category value
        mask = df[columns_to_check].astype(str).apply(lambda col: col.str.contains(str_i)).any(axis=1)
        cat = df[mask]
        
        # Calculate and print accuracy
        accuracy = cat['accuracy'].sum() / (len(cat))
        print(f'ground truth sample accuracy category {i}: {accuracy:.2f} with {len(cat)} rows')

def cat_confusion_matrix(df, columns_to_check, categories=7):
    results = {}

    for i in range(1, categories + 1):
        str_i = str(i)
        
        # Determine ground truth based on the presence of str_i in any of the specified columns
        ground_truth_mask = df[columns_to_check].astype(str).apply(lambda col: col.str.contains(str_i)).any(axis=1)
        
        # Predicted as category str_i
        predicted_mask = df['gpt-answer'].astype(str).str.contains(str_i)
        
        # Find false positives and false negatives
        false_positives = df[~ground_truth_mask & predicted_mask]
        false_negatives = df[ground_truth_mask & ~predicted_mask]
        
        # Store in the results dictionary
        results[str_i] = {
            'false_positives': false_positives[['statement', 'gpt-answer', 'case'] + columns_to_check],
            'false_negatives': false_negatives[['statement', 'gpt-answer', 'case'] + columns_to_check]
        }
        
        # You can still generate and print the confusion matrix if needed
        matrix = confusion_matrix(ground_truth_mask, predicted_mask)
        print_normalized_confusion_matrix(matrix, str_i)

    return results


def print_most_common_labels(category, columns_to_check):
    # Count the frequency of each label in each specified column
    category = category[category['accuracy'] == False]
    label_counts = {}
    for col in columns_to_check:
        counts = category[col].value_counts()
        label_counts[col] = counts

    # Print the most common labels for each column
    for col, counts in label_counts.items():
        print(f"Most common labels in {col}:")
        print(counts.head())  # Adjust the number in 'head()' to print more or fewer top labels
        print("-" * 50)


df = pd.read_json('data/categorization_CoT_prompt_categories_most_critical_example_gpt-4-0613_temp0.5.jsonl', lines = True)

categories_dict = {'A': 1, 
                'B': 2, 
                'C': 3, 
                'D': 4, 
                'E': 5, 
                'F': 6, 
                'G': 7, }



# Cleaning up GPT response
df.to_csv('out/categorization_CoT_prompt_cat_12567.csv')
df['gpt-answer'] = df['gpt-answer'].apply(lambda x: x[-1])

df['gpt-answer'] = df['gpt-answer'].apply(lambda x: categories_dict.get(x, x))


df.to_csv('out/categorization_CoT_prompt_cat_12567_sliced.csv')

# Saving dataframe


columns_to_check = ['category_yury', 'missing_info_Anne', 'missing_info_Camille']

#df.apply(pd.to_numeric, errors = 'coerce', downcast = 'integer')

df_filtered_cats = df.copy()


df_filtered_cats[columns_to_check] = df_filtered_cats[columns_to_check].replace([4.0, str(4), 4], 7) #if not all categories are present 

cat_accuracy_label_sample(df_filtered_cats, columns_to_check = columns_to_check)
cat_accuracy_prediction_sample(df_filtered_cats, columns_to_check = columns_to_check)
misclassifications = cat_confusion_matrix(df_filtered_cats, columns_to_check= columns_to_check)

print(misclassifications)

# Initialize an empty series to hold the combined value counts
combined_value_counts = pd.Series(dtype=int)

# Loop through the keys '1' to '7' and add the value counts
for i in range(1, 8):
    key = str(i)  # Convert the integer to a string to match the dictionary keys
    value_counts = misclassifications[key]['false_negatives']['case'].value_counts()
    combined_value_counts = combined_value_counts.add(value_counts, fill_value=0)

# Now 'combined_value_counts' holds the total counts of all cases across the series
sorted_combined_value_counts = combined_value_counts.sort_values(ascending=False)

# Now 'sorted_combined_value_counts' holds the sorted total counts of all cases
print(sorted_combined_value_counts)


display_confusion_matrices(df_filtered_cats, columns_to_check)


