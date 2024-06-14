import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV files
# commonsense_df = pd.read_csv('../result_metrics/openAi/commonsense_qa/all_metrics_GPT_commonsense.csv')
# date_df = pd.read_csv('../result_metrics/openAi/dateUnd/all_metrics_date_GPT.csv')
# gsm8k_df = pd.read_csv('../result_metrics/openAi/gsm8k/all_metrics_GPT_GSM8K.csv')
# law_df = pd.read_csv('../result_metrics/openAi/mmlu/all_metrics_GPT_LAW.csv')

gsm8k_df = pd.read_csv('../result_metrics/llama2/gsm8k/all_metrics_gsm8k.csv')
date_df = pd.read_csv('../result_metrics/llama2/bigbench/all_metrics_date_Llama2.csv')
commonsense_df = pd.read_csv('../result_metrics/llama2/commonsense_qa/all_metrics_common_Llama2.csv')
law_df = pd.read_csv('../result_metrics/llama2/mmlu/all_metrics_llama2_mmlu.csv')
# Add a column to each DataFrame to indicate the dataset
gsm8k_df['dataset'] = 'GSM8K'
date_df['dataset'] = 'Date'
commonsense_df['dataset'] = 'Common'
law_df['dataset'] = 'Law'

# Combine all DataFrames into one
combined_df = pd.concat([gsm8k_df, date_df, commonsense_df, law_df])

# Function to plot boxplots
def plot_metric_boxplots(metrics_df, output_dir, metric_name):
    print("DataFrame passed to plot_metric_boxplots:")
    print(metrics_df.head())

    plt.figure(figsize=(14, 8))
    methods = metrics_df['method'].unique()
    data_to_plot = [metrics_df[metrics_df['method'] == method][metric_name] for method in methods]

    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightcoral']  # Customize colors as needed
    bplot = plt.boxplot(data_to_plot, labels=methods, notch=False, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

        plt.ylim(0, 1)

    plt.title(f'Comparison of {metric_name} - Llama 2')
    plt.ylabel(metric_name)
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.grid(True)
    box_dir = os.path.join(output_dir, "Box_Plots")
    os.makedirs(box_dir, exist_ok=True)
    plt.savefig(os.path.join(box_dir, f'boxplot_{metric_name}_comparison.svg'))
    plt.close()

output_dir = "Boxplots_dir"
metrics = ['ece', 'auroc', 'auprc', 'auprc_p', 'auprc_n']
for metric in metrics:
    plot_metric_boxplots(combined_df, output_dir, metric)
#%%
