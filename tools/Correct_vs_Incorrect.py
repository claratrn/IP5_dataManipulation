import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Function to read and combine CSV files from a directory
def combine_csv_from_directory(directory, model_name):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            # Extract method and dataset name from filename
            method = os.path.splitext(filename)[0].split('_')[-1]
            # Assuming dataset name is part of the filename, e.g., 'datasetname_method.csv'
            dataset = os.path.splitext(filename)[0].split('_')[1]
            df['method'] = method
            df['dataset'] = dataset
            dataframes.append(df)

    combined_df = pd.concat(dataframes)
    combined_df['model'] = model_name
    return combined_df


# Function to plot correct vs. incorrect ratio
def plot_correct_incorrect_ratio(df, output_dir):
    # Calculate correct and incorrect counts
    df['correct'] = df['correct'].astype(bool)
    summary = df.groupby(['model', 'dataset', 'method'])['correct'].value_counts(normalize=True).unstack().fillna(0)
    summary.columns = ['Incorrect', 'Correct']
    summary = summary.reset_index()

    # Plot the data
    plt.figure(figsize=(14, 8))
    sns.barplot(x='method', y='Correct', hue='dataset', data=summary, errorbar=None, palette='muted')
    plt.title('Correct Answer Ratio by Method and Dataset')
    plt.ylabel('Correct Answer Ratio')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.legend(title='Dataset')
    plt.grid(True)

    # Save the plot
    plot_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'correct_incorrect_ratio.svg'))
    plt.close()


# Example usage
llama_directory = '../cleaned_data/Llama2/bigbench/'  # Directory containing LLaMA CSV files
output_dir = "correct_vs_incorrect_dir"

# Combine CSV files and create plot for LLaMA
combined_llama_df = combine_csv_from_directory(llama_directory, 'LLaMA')
plot_correct_incorrect_ratio(combined_llama_df, output_dir)

# If you have a directory for GPT-3.5, you can do the same:
# gpt_directory = 'path/to/gpt/csv/files'
# combined_gpt_df = combine_csv_from_directory(gpt_directory, 'GPT-3.5')
# plot_correct_incorrect_ratio(combined_gpt_df, output_dir)

# %%
