import pandas as pd

id_list = [
    450, 840, 1182, 280, 416, 471, 812, 975, 34, 1027, 609, 1711,
    1171, 1289, 1598, 174, 537, 1380, 1118, 1837, 335, 965, 1904,
    1845, 1241, 237, 348, 689, 277, 171, 116, 269, 378, 1698, 1851,
    1371, 1144, 1075, 990, 752, 438, 185, 1406, 1923, 205, 520, 1025,
    3, 1617, 1009, 1455
]

questions_df = pd.read_json('data/clarifying_context_questions_gpt-4-0613_temp0.5.jsonl', lines = True)
details_df = pd.read_json('data/missing_detail_experiment_gpt-4-0613_temp0.5.jsonl', lines = True)

questions_df, details_df = questions_df[questions_df['example_id'].isin(id_list)], details_df[details_df['example_id'].isin(id_list)]

questions_df.to_csv('questions_df.csv')
details_df.to_csv('details_df.csv')