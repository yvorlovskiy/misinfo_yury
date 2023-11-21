
import pandas as pd 
# Correcting the file names and reloading the data accordingly

# Correct file paths
file_paths_corrected = {
    "LIAR_train_saved_til_13412": 'LIAR_train_saved_til_13412.jsonl',
    "LIAR_train_index_3900_to_6500": 'LIAR_train_index_3900_to_6500_id_6419.jsonl',
    "LIAR_train_saved_til_index_10239": 'LIAR_train_saved_til_index_10239_id_1155.jsonl',
    "LIAR_train_saved_til_index_8800": 'LIAR_train_saved_til_index_8800_id_48.jsonl',
}

# Load the data from each corrected file into pandas DataFrames
df_13412_corrected = pd.read_json(file_paths_corrected["LIAR_train_saved_til_13412"], lines=True)
df_3900_to_6500_corrected = pd.read_json(file_paths_corrected["LIAR_train_index_3900_to_6500"], lines=True)
df_10239_corrected = pd.read_json(file_paths_corrected["LIAR_train_saved_til_index_10239"], lines=True)
df_6500_to_8800_corrected = pd.read_json(file_paths_corrected["LIAR_train_saved_til_index_8800"], lines=True)

# Slicing the DataFrames as per the new instructions
# Dates and labels until row number 3900 from LIAR_train_saved_til_13412.jsonl
df_part1_corrected = df_13412_corrected.loc[:3899]

# Dates and labels from 3900 to 6500 from LIAR_train_index_3900_to_6500.jsonl
df_part2_corrected = df_3900_to_6500_corrected.loc[3900:6499]

df_part3_corrected = df_6500_to_8800_corrected.loc[6500:8799]

# Dates and labels starting at 6500 from LIAR_train_saved_til_index_10239.jsonl
df_part4_corrected = df_10239_corrected.loc[8800:]

# Concatenating the corrected parts
combined_df_corrected = pd.concat([df_part1_corrected, df_part2_corrected, df_part3_corrected, df_part4_corrected])

# Checking the number of empty 'date' rows in the combined DataFrame
empty_dates_combined_corrected = combined_df_corrected['date'].isna().sum()
empty_dates_combined_corrected

combined_df_corrected.to_csv('updated_data.csv')