# Sample data
import pandas as pd
import matplotlib.pyplot as plt
#
# L_gsm8k_df = pd.read_csv('directory of all experiments with bigbench dataset in llama')
# L_date_df = pd.read_csv('directory of all experiments with date dataset in llama')
# L_commonsense_df = pd.read_csv('directory of all commonsense with bigbench dataset in llama')
# L_law_df = pd.read_csv('directory of all experiments with law dataset in llama')
#
# G_gsm8k_df = pd.read_csv('directory of all experiments with bigbench dataset in GPT')
# G_date_df = pd.read_csv('directory of all experiments with date dataset in GPT')
# G_commonsense_df = pd.read_csv('directory of all commonsense with bigbench dataset in GPT')
# G_law_df = pd.read_csv('directory of all experiments with law dataset in GPT')
#
# G_gsm8k_df['dataset'] = 'GSM8K'
# G_date_df['dataset'] = 'Date'
# G_commonsense_df['dataset'] = 'Common'
# G_law_df['dataset'] = 'Law'
#
# combined_df_GPT = pd.concat([G_gsm8k_df, G_date_df, G_commonsense_df, G_law_df])
# combined_df_Law = pd.concat([L_gsm8k_df, L_date_df, L_commonsense_df, L_law_df])
#

# Function to read CSV and extract relevant columns
def read_and_extract(filepath):
    df = pd.read_csv(filepath)
    return df[['confidence', 'predicted label', 'ground truth', 'correct']]

# Read the data from CSV files
df_openai = read_and_extract('path_to_openai_csv.csv')
df_llama2 = read_and_extract('path_to_llama2_csv.csv')

# Count correct and false predictions
correct_openai = df_openai['correct'].sum()
false_openai = len(df_openai) - correct_openai

correct_llama2 = df_llama2['correct'].sum()
false_llama2 = len(df_llama2) - correct_llama2

# Plotting the data
labels = ['OpenAI', 'Llama2']
correct_counts = [correct_openai, correct_llama2]
false_counts = [false_openai, false_llama2]

x = range(len(labels))

fig, ax = plt.subplots()

# Bar width
bar_width = 0.35

# Plotting bars for correct predictions
rects1 = ax.bar(x, correct_counts, bar_width, label='Correct Predictions')

# Plotting bars for false predictions
rects2 = ax.bar([p + bar_width for p in x], false_counts, bar_width, label='False Predictions')

# Adding labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('Count')
ax.set_title('Comparison of Predictions between OpenAI and Llama2')
ax.set_xticks([p + bar_width/2 for p in x])
ax.set_xticklabels(labels)
ax.legend()

# Display plot
plt.show()
plt.savefig('Llama_and_OpenAI.png')

#%%
