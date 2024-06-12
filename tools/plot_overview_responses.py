import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Directory containing the CSV files
directory_path = 'missingVal/OpenAi/CommonsenseQA'  # Adjust the path if needed

# Initialize a list to store data
data_list = []


def determine_method(file_name):
    if "few_shot" in file_name:
        return "few shot cot"
    elif "multi_step" in file_name:
        return "multi step"
    elif "vanilla" in file_name:
        return "vanilla"
    elif "zero" in file_name:
        return "zero shot cot"
    else:
        return "unknown"


# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('_summary.csv'):
        filepath = os.path.join(directory_path, filename)

        # Read the CSV file
        df = pd.read_csv(filepath)
        print(f"Columns in {filename}: {df.columns}")

        # Extract method from filename
        method = determine_method(filename)

        # Extract relevant data
        total_entries = df.loc[df['Metric'] == 'Total entries', 'Value'].values[0]
        answered_entries = df.loc[df['Metric'] == 'Answered entries', 'Value'].values[0]
        unanswered_entries = df.loc[df['Metric'] == 'Unanswered entries', 'Value'].values[0]
        missing_data_ratio = df.loc[df['Metric'] == 'Missing data ratio', 'Value'].values[0]

        total_entries = int(total_entries)
        answered_entries = int(answered_entries)
        unanswered_entries = int(unanswered_entries)
        missing_data_ratio = float(missing_data_ratio.strip('%'))

        # Append to the data list
        data_list.append({
            'Method': method,
            'Total Entries': total_entries,
            'Answered Entries': answered_entries,
            'Unanswered Entries': unanswered_entries,
            'Missing Data Ratio (%)': missing_data_ratio
        })

# Convert the data list to a DataFrame
df_summary = pd.DataFrame(data_list)

# Plot combined bar chart
fig, ax1 = plt.subplots(figsize=(12, 8))

bar_width = 0.3
index = np.arange(len(df_summary['Method']))

# Plot Answered Entries
bars1 = ax1.bar(index, df_summary['Answered Entries'], bar_width, label='Answered Entries', color='green')

# Plot Unanswered Entries
bars2 = ax1.bar(index + bar_width, df_summary['Unanswered Entries'], bar_width, label='Unanswered Entries', color='red')

# Adding labels
ax1.set_xlabel('Method')
ax1.set_ylabel('Entries')
ax1.set_title('Comparison of Answered and Unanswered Entries in GPT-3.5 - Commonsense QA')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(df_summary['Method'], rotation=45, ha='right')


# Adding data labels for bars
def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom')


# Adding text labels for Missing Data Ratio next to the Unanswered Entries bars
def add_ratio_labels(bars, ratios, ax):
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width(), height + 25, f'{ratio:.2f}%', ha='left', va='bottom',
                color='purple')


add_labels(bars1, ax1)
add_labels(bars2, ax1)
add_ratio_labels(bars2, df_summary['Missing Data Ratio (%)'], ax1)

# Adding custom legend entry for Missing Data Ratio text
missing_ratio = plt.Line2D([0], [0], color='purple', marker='o', linestyle='None', markersize=10,
                           label='Missing Data Ratio (%)')
ax1.legend(handles=[bars1, bars2, missing_ratio], loc='center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True,
           ncol=3)

# Adjust the left and right padding to ensure the ratio labels are not cut off
plt.tight_layout()
plt.savefig('invalid_ratio_openAI_common.svg')
plt.show()

