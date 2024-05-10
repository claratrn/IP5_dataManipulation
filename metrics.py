import os

import pandas as pd
from matplotlib import pyplot as plt
from netcal.presentation import ReliabilityDiagram
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np

from tools.compute_metrics import compute_conf_metrics

# Read CSV data into DataFrame
directory_path = "cleaned_data/"
output_dir = "result_metrics"
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

    # Annotating bars with the number of observations
    #  for count, x in zip(n_wrong + n_correct, bins_correct[:-1]):
    #      if count > 0:  # Only annotate non-zero bars to avoid clutter
    #          plt.text(x + (bins_correct[1] - bins_correct[0]) / 2, count, str(int(count)), ha='center', va='bottom')

    plt.title(f'Confidence Histogram - {method} {dataset} {model} ')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xticks(np.arange(0, 105, 5))
    plt.savefig(os.path.join(visual_dir, f'{file_name}_histogram_{method}.png'))
    plt.close()


def plot_ece_diagram(y_true, y_confs, method, model, dataset, file_name):
    n_bins = 20
    diagram = ReliabilityDiagram(n_bins)
    plt.figure(figsize=(12, 8))
    diagram.plot(np.array(y_confs), np.array(y_true))
    plt.title(f'Expected Calibration Error - {method} {dataset} {model}')
    plt.savefig(os.path.join(visual_dir, f'{file_name}_ECE_{method}.png'))
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
    plt.savefig(os.path.join(visual_dir, f'{file_name}_ROC_{method}.png'))
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
    plt.savefig(os.path.join(visual_dir, f'{file_name}_PRC_{method}.png'))
    plt.close()


commonsense_metrics = pd.read_csv('result_metrics/commonsense_experiments_metrics.csv')


def plot_metric_boxplots(metrics_df, output_dir, metric_name):
    print("DataFrame passed to plot_metric_boxplots:")
    print(metrics_df.head())

    plt.figure(figsize=(12, 8))
    methods = metrics_df['method'].unique()
    data_to_plot = [metrics_df[metrics_df['method'] == method][metric_name] for method in methods]

    plt.boxplot(data_to_plot, labels=methods, notch=True, patch_artist=True)
    plt.title(f'Comparison of {metric_name} by Elicitation Method')
    plt.ylabel(metric_name)
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'boxplot_{metric_name}_comparison.png'))
    plt.close()


def plot_all_visualisations(y_true, y_confs, elicitation_method, model, dataset, file_name):
    plot_confidence_histogram(y_true, y_confs, elicitation_method, model, dataset,file_name)
    plot_ece_diagram(y_true, y_confs, elicitation_method, model, dataset, file_name)
    plot_roc_curve(y_true, y_confs, elicitation_method, model, dataset, file_name)
    plot_precision_recall_curve(y_true, y_confs, elicitation_method, model, dataset, file_name)


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
    elif "gsm8k" in file_name:
        return "gsm8k"
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

# plot_metric_boxplots(all_metrics, output_dir, 'ece')
# plot_metric_boxplots(all_metrics, output_dir, 'auroc')
# plot_metric_boxplots(all_metrics, output_dir, 'auprc')

output_path = os.path.join(output_dir, 'commonsense_experiments_metrics.csv')
all_metrics.to_csv(output_path, index=False)
print(f"All metrics saved to {output_path}")


#%%
