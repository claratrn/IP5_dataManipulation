import pandas as pd

# Example DataFrames
df1 = pd.read_csv('../results/llama2/gsm8k/few_shot_cot_number/20240603-114449626099_f68528eb5c7f4637a7e34112039be56b_answered.csv')
df2 = pd.read_csv('../results/llama2/gsm8k/few_shot_cot_number/few_shot_fix_answered.csv')



# Filter rows where 'correct' is True
df1_true = df1[df1['correct'] == True]
df2_true = df2[df2['correct'] == True]

# Concatenate the filtered dataframes
result = pd.concat([df1_true, df2_true])

# Remove duplicate rows
result = result.drop_duplicates()

# Reset index for better readability (optional)
result.reset_index(drop=True, inplace=True)

# Display the resulting DataFrame
result.to_csv('merged_true_rows.csv', index=False)

#%%
