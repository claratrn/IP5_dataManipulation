import pandas as pd

# Example DataFrames
df1 = pd.read_csv('../results/llama2/gsm8k/zero_shot_cot_gsm8k/20240603-112531205664_289c2160bbb5425186b314e0fb9e4551_answered.csv')
df2 = pd.read_csv('../results/llama2/gsm8k/zero_shot_cot_gsm8k/zero_shot_cot_gsm8k_fix_answered.csv')

# Filter and merge the 'True' rows, ensuring no duplicates
true_rows_df1 = df1[df1['correct'] == True]
true_rows_df2 = df2[df2['correct'] == True]
merged_true_rows = pd.concat([true_rows_df1, true_rows_df2]).drop_duplicates()

# Identify the 'False' rows
false_rows_df1 = df1[df1['correct'] == False]
false_rows_df2 = df2[df2['correct'] == False]
all_false_rows = pd.concat([false_rows_df1, false_rows_df2])


# Remove 'False' rows that have a corresponding 'True' row in the merged_true_rows
final_false_rows = all_false_rows[~all_false_rows['index'].isin(merged_true_rows['index'])]
# Drop only the first occurrence of duplicate false rows
final_false_rows = final_false_rows.drop_duplicates(subset='index', keep='last')
# Combine the merged_true_rows and final_false_rows
final_merged_df = pd.concat([merged_true_rows, final_false_rows])
final_merged_df = final_merged_df.drop_duplicates(subset='index')

# Display the resulting DataFrame
final_merged_df.to_csv('merged.csv', index=False)

#%%
