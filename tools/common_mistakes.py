import pandas as pd

# Load the data from the CSV files
df_llama2 = pd.read_csv('../results/llama2/gsm8k/zero_shot_cot_gsm8k/20240603-112531205664_289c2160bbb5425186b314e0fb9e4551_answered.csv')
df_openAi = pd.read_csv('../results/openAi/gsm8k/zero_shot_cot_gsm8k/20240525-174150582952_72dcbbf5bc124790b148e5ddce48fc59_answered.csv')


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
