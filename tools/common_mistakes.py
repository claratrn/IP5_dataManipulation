import pandas as pd

# Load the data from the CSV files
df_llama2 = pd.read_csv('../results/llama2/gsm8k/few_shot_cot_number/few_shot_fix_answered.csv')
df_openAi = pd.read_csv('../results/openAi/gsm8k/few_shot_cot_number/20240526-011547328226_bd607da7b1094ebf88a525b0859baef6_answered.csv')


# Filter for incorrect answers
df_llama2_incorrect = df_llama2[df_llama2['correct'] == False]
df_openAi_incorrect = df_openAi[df_openAi['correct'] == False]

# Merge the incorrect answers DataFrames on the question column
common_errors = pd.merge(df_llama2_incorrect, df_openAi_incorrect, on='question', suffixes=('_llama', '_openai'))

# Select relevant columns to display
common_errors_simplified = common_errors[['question', 'predicted label_llama', 'predicted label_openai', 'ground truth_llama', 'ground truth_openai']]

# Display common incorrectly answered questions
print("Common Incorrectly Answered Questions:")
print(common_errors_simplified)

common_errors_simplified.to_csv('common_wrong_answers.csv', index=False)

# Analyzing the frequency of common errors
error_counts = common_errors['question'].value_counts()
print("Frequency of Common Errors:")
print(error_counts)
error_counts.to_csv('frequency_of_common_errors.csv')

#%%
