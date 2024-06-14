import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

np.float = np.float64

from tools.compute_metrics import compute_conf_metrics, manual_ece

# Read CSV data into DataFrame
directory_path = "cleaned_data/openAi/gsm8k_p"
output_dir = "result_metrics/openAi/gsm8k_p"
visual_dir = os.path.join(output_dir, "visuals_as_svg")
os.makedirs(visual_dir, exist_ok=True)


#################### VISUALIZATION FUNCTIONS ####################
# y_true:correct , y_confs: confidence score

def plot_confidence_histogram(y_true, y_confs, method, model, dataset, file_name):
    y_confs = np.array(y_confs, dtype=float)
    print("Original confidence scores (first 5 entries):", y_confs[:5])
    y_confs = [conf * 100 for conf in y_confs]
    print("Modified confidence scores (x100, first 5 entries):", y_confs[:5])

    wrong_confs = [conf for conf, true in zip(y_confs, y_true) if not true]
    correct_confs = [conf for conf, true in zip(y_confs, y_true) if true]

    plt.figure(figsize=(6, 4))

    # to add a buffer around the histogram
    bins = np.linspace(-2.5, 102.5, 22)
    n_wrong, bins_wrong, patches_wrong = plt.hist(wrong_confs, bins=bins, alpha=0.7, color='red', label='Wrong')
    n_correct, bins_correct, patches_correct = plt.hist(correct_confs, bins=bins, alpha=0.7, color='green',
                                                        label='Correct', bottom=n_wrong)

    # # Annotating bars with the number of observations
    #  for count, x in zip(n_wrong + n_correct, bins_correct[:-1]):
    #      if count > 0:  # Only annotate non-zero bars to avoid clutter
    #          plt.text(x + (bins_correct[1] - bins_correct[0]) / 2, count, str(int(count)), ha='center', va='bottom')

    plt.title(f'Confidence Histogram - {method} {dataset} {model} ')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xticks(np.arange(0, 105, 5))
    histogram_dir = os.path.join(visual_dir, "histogram")
    os.makedirs(histogram_dir, exist_ok=True)

    plt.savefig(os.path.join(histogram_dir, f'{file_name}_histogram_{method}.svg'))
    plt.close()


def get_ece_from_all_metrics(all_metrics, method):
    print("Columns in all_metrics DataFrame:", all_metrics.columns)  # Debugging line
    print("Contents of all_metrics DataFrame:", all_metrics)  # Debugging line
    # Check if the DataFrame contains the desired method
    if method in all_metrics['method'].values:
        # Extract the ECE score for the specified method
        ece_score = all_metrics.loc[all_metrics['method'] == method, 'ece'].iloc[0]
        return ece_score
    else:
        return None


def plot_roc_curve(y_true, y_scores, method, model, dataset, file_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5.5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUROC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUROC - {method} {dataset} {model}')
    plt.legend(loc="lower right")
    plt.xticks(np.arange(0, 1.1, 0.1))
    roc_dir = os.path.join(visual_dir, "Auroc")
    os.makedirs(roc_dir, exist_ok=True)
    plt.savefig(os.path.join(roc_dir, f'{file_name}_ROC_{method}.svg'))
    plt.close()


def plot_precision_recall_curve(y_true, y_scores, method, model, dataset, file_name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(5.5, 4))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {method} {dataset} {model}')
    plt.legend(loc="lower left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    auprc_dir = os.path.join(visual_dir, "AUPRC")
    os.makedirs(auprc_dir, exist_ok=True)
    plt.savefig(os.path.join(auprc_dir, f'{file_name}_PRC_{method}.svg'))
    plt.close()


# df_commonsense = pd.read_csv('result_metrics/commonsense_experiments_metrics.csv')
# df_gsm8k = pd.read_csv('result_metrics/gsm8k_experiments_metrics.csv')
# metrics_combined = pd.concat([df_commonsense, df_gsm8k])
#
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']


def plot_metric_boxplots(metrics_df, output_dir, metric_name):
    print("DataFrame passed to plot_metric_boxplots:")
    print(metrics_df.head())

    plt.figure(figsize=(6.5, 4))
    methods = metrics_df['method'].unique()
    data_to_plot = [metrics_df[metrics_df['method'] == method][metric_name] for method in methods]

    bplot = plt.boxplot(data_to_plot, labels=methods, notch=False, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title(f'Comparison of {metric_name} by Elicitation Method')
    plt.ylabel(metric_name)
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.grid(True)
    box_dir = os.path.join(visual_dir, "Box_Plots")
    os.makedirs(box_dir, exist_ok=True)
    plt.savefig(os.path.join(box_dir, f'boxplot_{metric_name}_comparison.svg'))
    plt.close()


def plot_ece_diagram(y_true, y_confs, method, model, dataset, file_name):
    n_bins = 20
    plt.figure(figsize=(6.4, 4.8), dpi=300)

    # Create histogram bins for y_confidences
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(y_confs, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracy_per_bin = np.zeros(n_bins)
    avg_confidence_per_bin = np.zeros(n_bins)
    total_samples = np.sum(bin_counts)

    # Calculate the accuracy per bin only if there are elements in the bin
    for i in range(n_bins):
        if i == 0:
            in_bin = (y_confs >= bin_edges[i]) & (y_confs < bin_edges[i + 1])
        elif i == n_bins - 1:
            in_bin = (y_confs >= bin_edges[i]) & (y_confs <= bin_edges[i + 1])
        else:
            in_bin = (y_confs >= bin_edges[i]) & (y_confs < bin_edges[i + 1])

        if in_bin.any():  # Check if there are any elements in the bin
            accuracy_per_bin[i] = np.mean(y_true[in_bin])
            avg_confidence_per_bin[i] = np.mean(y_confs[in_bin])
        else:
            accuracy_per_bin[i] = np.nan  # Assign NaN for empty bins
            avg_confidence_per_bin[i] = np.nan

    sparse_threshold = 10
    # Plot the reliability diagram
    bar_width = 1 / (n_bins + 7)  # Increase the bar width slightly
    for i in range(n_bins):
        if not np.isnan(accuracy_per_bin[i]) and bin_counts[i] > 0:
            color = 'tab:blue' if bin_counts[i] >= sparse_threshold else 'tab:orange'
            plt.bar(float(bin_centers[i]), float(accuracy_per_bin[i]), width=bar_width, color=color, edgecolor='black',
                    alpha=0.7)
            plt.text(float(bin_centers[i]), float(accuracy_per_bin[i]) + 0.02, f'{bin_counts[i]}', ha='center',
                     fontsize=5)
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'r--')

    plt.title(f'Expected Calibration Error - {method} {dataset} {model}', fontsize=10)
    ece_score = manual_ece(y_true, y_confs, n_bins) * 100
    plt.text(0.05, 0.95, f'ECE: {ece_score:.2f}%', transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
    plt.text(0.05, 0.90, f'Total samples:{total_samples}', transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top')
    legend_elements = [
        plt.Line2D([], [], color='red', linestyle='--', label='Perfect Calibration', linewidth=1),
        plt.Rectangle((0, 0), 1, 1, color='tab:blue', label='Output', linewidth=1),
        plt.Rectangle((0, 0), 1, 1, color='tab:orange', label='Sparse Bins (< 10 samples)', linewidth=1)
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.16), fancybox=True, shadow=True,
               ncol=3, fontsize=6)
    tick_positions = np.linspace(0, 1, n_bins + 1)
    tick_labels = [f"{pos:.2f}" for pos in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=-45, ha='left', fontsize=6)

    plt.gca().set_aspect('equal', adjustable='box')

    # Add buffer around the plot
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 1.05)
    plt.subplots_adjust(bottom=0.3, top=0.9)
    plt.xlabel("Confidence", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)

    ece_dir = os.path.join(visual_dir, "ECE")
    os.makedirs(ece_dir, exist_ok=True)
    plt.tight_layout()
    # plt.savefig(os.path.join(ece_dir, f'{file_name}_ECE_{method}.svg'))
    plt.savefig(os.path.join(ece_dir, f'ECE_GSM8K_{method}.svg'))
    plt.close()


def plot_all_visualisations(y_true, y_confs, elicitation_method, model, dataset, file_name):
    y_true = np.array([1 if x else 0 for x in y_true])
    # plot_confidence_histogram(y_true, y_confs, elicitation_method, model, dataset, file_name)
    # plot_roc_curve(y_true, y_confs, elicitation_method, model, dataset, file_name)
    # plot_precision_recall_curve(y_true, y_confs, elicitation_method, model, dataset, file_name)
    plot_ece_diagram(y_true, y_confs, elicitation_method, model, dataset, file_name)
    print("All visualisations saved to: ", visual_dir)


def determine_method(file_name):
    if "few" in file_name:
        return "few shot cot"
    elif "multi" in file_name:
        return "multi step"
    elif "vanilla" in file_name:
        return "vanilla"
    elif "zero" in file_name:
        return "zero shot cot"
    else:
        return "unknown"


def determine_model(file_name):
    if "openAi" in file_name:
        return "Gpt 3.5"
    elif "llama2" in file_name:
        return "Llama 2"
    else:
        return "unknown"


def determine_dataset(file_name):
    if "commonsense" in file_name:
        return "commonsense QA"
    elif "gsm8k_test" in file_name:
        return "gsm8k_test"
    elif "gsm8k" in file_name:
        return "gsm8k"
    elif "mmlu" in file_name:
        return "Professional Law"
    elif "date" in file_name:
        return "Date Understanding"
    else:
        return "unknown"


all_metrics = pd.DataFrame()

# Extract confidence scores, true labels, and predicted labels from DataFrame
for file_name in os.listdir(directory_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory_path, file_name)
        dataFrame = pd.read_csv(file_path)
        if 'correct' not in dataFrame.columns:
            print(f"Column 'correct' not found in {file_name}. Skipping file.")
            continue

        correct = dataFrame['correct'].values
        confids = dataFrame['confidence'].values
        method = determine_method(file_name)
        model = determine_model(file_name)
        dataset = determine_dataset(file_name)

        metrics = compute_conf_metrics(correct, confids, method)
        metrics_df = pd.DataFrame([metrics])
        all_metrics = pd.concat([all_metrics, metrics_df], ignore_index=True)
        plot_all_visualisations(correct, confids, method, model, dataset, file_name)

        print(all_metrics.head())

output_path = os.path.join(output_dir, 'top_p_gsm8k.csv')
all_metrics.to_csv(output_path, index=False)

# %%
