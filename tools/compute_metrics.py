import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
from netcal.metrics import ECE


def compute_conf_metrics(y_true, y_confs, elicitation_method):
    result_metrics = {'method': elicitation_method}
    # confidence scores already normalized 0 to 1
    y_confs = np.array(y_confs)

    # Compute accuracy
    accuracy = sum(y_true) / len(y_true)
    print("Accuracy:", accuracy)
    result_metrics['acc'] = accuracy

    # Check if confidence scores are in the range [0, 1]
    assert all([0 <= x <= 1 for x in y_confs]), y_confs

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

    result_metrics.update({'acc': accuracy, 'auroc': roc_auc, 'auprc': auprc, 'ece': ece_score})

    return result_metrics
