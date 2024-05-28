import os

import pandas as pd
from matplotlib import pyplot as plt
from netcal.presentation import ReliabilityDiagram
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np

from tools.compute_metrics import compute_conf_metrics

# Read CSV data into DataFrame
directory_path = "cleaned_data/gsm8k"
output_dir = "result_metrics/gsm8k"
visual_dir = os.path.join(output_dir, "visuals")
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

    plt.figure(figsize=(12, 8))

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

    plt.savefig(os.path.join(histogram_dir, f'{file_name}_histogram_{method}.png'))
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

def determine_outlier_threshold(y_confs, n_bins=20):
    # Create histogram bins for y_confidences
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(y_confs, bins=n_bins, range=(0, 1))

    # Calculate mean and standard deviation of bin counts
    mean_count = np.mean(bin_counts)
    std_count = np.std(bin_counts)

    # Determine threshold using mean and standard deviation
    threshold = mean_count - 2 * std_count

    print(f"Mean bin count: {mean_count}")
    print(f"Standard deviation of bin counts: {std_count}")
    print(f"Outlier threshold: {threshold}")

    return threshold
def plot_ece_diagram(y_true, y_confs, method, model, dataset, file_name):
    outlier_threshold = determine_outlier_threshold(y_confs)
    n_bins = 20
    plt.figure(figsize=(10, 6), dpi=600)
    plt.gca().set_position([0.1, 0.1, 0.8, 0.8])

    # Create histogram bins for y_confidences
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, bin_edges = np.histogram(y_confs, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracy_per_bin = np.zeros(n_bins)

    print("Bin counts:", bin_counts, method)
    print("Bin edges:", bin_edges, method)


# Calculate the accuracy per bin only if there are elements in the bin
    for i in range(n_bins):
        in_bin = (y_confs >= bin_edges[i]) & (y_confs <= bin_edges[i+1]) if i == n_bins - 1 else (y_confs >= bin_edges[i]) & (y_confs < bin_edges[i+1])
        if in_bin.any():  # Check if there are any elements in the bin
            accuracy_per_bin[i] = np.mean(y_true[in_bin])
        else:
            accuracy_per_bin[i] = np.nan  # Assign NaN for empty bins
    print("Accuracy per bin:", accuracy_per_bin, method)

    # Plot the reliability diagram
    for i in range(n_bins):
        if not np.isnan(accuracy_per_bin[i]):
            color = 'tab:blue' if bin_counts[i] > outlier_threshold else 'tab:orange'
            plt.bar(bin_centers[i], accuracy_per_bin[i], width=1/n_bins, color='tab:blue', edgecolor='black', alpha=0.7)
        else:
            plt.bar(bin_centers[i], 0, width=1/n_bins, color='white', edgecolor='black', alpha=0.7)  # Invisible bar for empty bins


    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'r--')

    plt.title(f'Expected Calibration Error - {method} {dataset} {model}')
    ece_score = get_ece_from_all_metrics(all_metrics, method) * 100
    plt.text(0.05, 0.90, f'ECE: {ece_score:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    legend_elements = [
        plt.Line2D([], [], color='red', linestyle='--', label='Perfect Calibration'),
        plt.Rectangle((0, 0), 1, 1, color='tab:blue', label='Output'),
        plt.Rectangle((0, 0), 1, 1, color='tab:orange', label='Outlier'),

    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

    tick_positions = np.linspace(0, 1, n_bins + 1)
    tick_labels = [f"{pos:.2f}" for pos in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)

    # Add buffer around the plot
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 1.05)
    plt.subplots_adjust(bottom=0.25, hspace=0.5)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    ece_dir = os.path.join(visual_dir, "ece")
    os.makedirs(ece_dir, exist_ok=True)

    plt.savefig(os.path.join(ece_dir, f'{file_name}_ECE_{method}.png'))
    plt.close()


def plot_roc_curve(y_true, y_scores, method, model, dataset, file_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {method} {dataset} {model}')
    plt.legend(loc="lower right")
    plt.xticks(np.arange(0, 1.1, 0.1))
    roc_dir = os.path.join(visual_dir, "Auroc")
    os.makedirs(roc_dir, exist_ok=True)
    plt.savefig(os.path.join(roc_dir, f'{file_name}_ROC_{method}.png'))
    plt.close()


def plot_precision_recall_curve(y_true, y_scores, method, model, dataset, file_name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {method} {dataset} {model}')
    plt.legend(loc="lower left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    auprc_dir = os.path.join(visual_dir, "AUPRC")
    os.makedirs(auprc_dir, exist_ok=True)
    plt.savefig(os.path.join(auprc_dir, f'{file_name}_PRC_{method}.png'))
    plt.close()


# df_commonsense = pd.read_csv('result_metrics/commonsense_experiments_metrics.csv')
# df_gsm8k = pd.read_csv('result_metrics/gsm8k_experiments_metrics.csv')
# metrics_combined = pd.concat([df_commonsense, df_gsm8k])
#
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']


def plot_metric_boxplots(metrics_df, output_dir, metric_name):
    print("DataFrame passed to plot_metric_boxplots:")
    print(metrics_df.head())

    plt.figure(figsize=(12, 8))
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
    plt.savefig(os.path.join(box_dir, f'boxplot_{metric_name}_comparison.png'))
    plt.close()


def plot_all_visualisations(y_true, y_confs, elicitation_method, model, dataset, file_name):
    y_true = np.array([1 if x else 0 for x in y_true])
    plot_confidence_histogram(y_true, y_confs, elicitation_method, model, dataset, file_name)
    plot_roc_curve(y_true, y_confs, elicitation_method, model, dataset, file_name)
    plot_precision_recall_curve(y_true, y_confs, elicitation_method, model, dataset, file_name)
    plot_ece_diagram(y_true, y_confs, elicitation_method, model, dataset, file_name)
    print("All visualisations saved to: ", visual_dir)


def determine_method(file_name):
    if "few_shot" in file_name:
        return "few shot cot"
    elif "multi_step" in file_name:
        return "multi step"
    elif "vanilla" in file_name:
        return "vanilla"
    elif "zero_shot" in file_name:
        return "zero shot cot"
    else:
        return "unknown"


def determine_model(file_name):
    if "openAi" in file_name:
        return "Gpt 3.5"
    else:
        return "unknown"


def determine_dataset(file_name):
    if "commonsense" in file_name:
        return "commonsense QA"
    elif "gsm8k_test" in file_name:
        return "gsm8k_test"
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
        # plot_all_visualisations(correct, confids, method, model, dataset, file_name)

        print(all_metrics.head())

output_path = os.path.join(output_dir, 'allmetrics_gsm8k.csv')
all_metrics.to_csv(output_path, index=False)
print(f"All metrics saved to {output_path}")


#
# def plot_ece_diagram(y_true, y_confs, method, model, dataset, file_name):
#     from netcal.presentation import ReliabilityDiagram
#     n_bins = 20
#     diagram = ReliabilityDiagram(n_bins)
#
#     plt.figure()
#     diagram.plot(np.array(y_confs), np.array(y_true))
#
#     bin_counts, bin_edges = np.histogram(y_confs, bins=n_bins, range=(0, 1))
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     for i in range(n_bins):
#         if bin_counts[i] == 0:
#             plt.bar(bin_centers[i], 1, width=1/n_bins, color='white', edgecolor='black', alpha=0.0)
#
#
#     plt.title(f'Expected Calibration Error - {method} {dataset} {model}')
#     ece_score = get_ece_from_all_metrics(all_metrics, method) * 100
#     plt.text(0.05, 0.90, f'ECE: {ece_score:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
#
#     legend_elements = [
#         plt.Line2D([], [], color='red', linestyle='--', label='Perfect Calibration'),
#         plt.Rectangle((0, 0), 1, 1, color='tab:blue', label='Output'),
#         plt.Rectangle((0, 0), 1, 1, color='tab:red', label='Gap'),
#
#     ]
#     plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True,
#                ncol=3)
#
#     tick_positions = np.linspace(0, 1, n_bins + 1)
#
#     tick_labels = [f"{pos:.2f}" for pos in tick_positions]
#     plt.xticks(tick_positions, tick_labels, rotation=45)
#
# # Add buffer around the plot
#     plt.xlim(-0.05, 1.05)
#     plt.ylim(0, 1.05)
#     plt.subplots_adjust(bottom=0.25, hspace=0.5)
#     plt.xlabel("Confidence")
#     plt.ylabel("Accuracy")
#     plt.savefig((os.path.join(visual_dir, f'{file_name}_ECE_{method}.png')), dpi=600)
#
#%%
