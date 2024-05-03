import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from netcal.metrics import ECE
import numpy as np

# Read CSV data into DataFrame
file_path = "cleaned_data/cleaned_commonsense_qa_multi_step_multiple_choice.csv"
dataFrame = pd.read_csv(file_path)

#y_true: true label, y_confs: ocnfidence score
def compute_conf_metrics(y_true, y_confs):
    result_metrics = {}

    # Compute accuracy
    accuracy = sum(y_true) / len(y_true)
    print("Accuracy:", accuracy)
    result_metrics['acc'] = accuracy

    # Check if confidence scores are in the range [0, 1]
    assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs

    # Convert  to numpy arrays
    y_confs, y_true = np.array(y_confs), np.array(y_true)

    # Compute AUROC
    roc_auc = roc_auc_score(y_true, y_confs)
    print("ROC AUC score:", roc_auc)
    result_metrics['auroc'] = roc_auc

    # Compute AUPRC
    auprc = average_precision_score(y_true, y_confs)
    print("AUC PRC score:", auprc)
    result_metrics['auprc'] = auprc

    # Compute ECE
    n_bins = 10
    ece = ECE(n_bins)
    ece_score = ece.measure(y_confs, y_true)
    print("ECE score:", ece_score)
    result_metrics['ece'] = ece_score

    return result_metrics


# Extract confidence scores, true labels, and predicted labels from DataFrame
true_labels = dataFrame['correct'].values
confids = dataFrame['confidence'].values

# Compute confidence metrics
metrics = compute_conf_metrics(true_labels, confids)

#%%
