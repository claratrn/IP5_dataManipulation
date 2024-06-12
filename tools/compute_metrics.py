import numpy as np
from netcal.metrics import ECE
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve


def filter_out_of_range(y_true, y_confs):
    valid_indices = [i for i, conf in enumerate(y_confs) if 0 <= conf <= 1]
    return y_true[valid_indices], y_confs[valid_indices]


def manual_ece(y_true, y_confs, bin_size):
    bin_boundaries = np.arange(0, 1.05, 0.05)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    n = len(y_true)
    ece = 0.0

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if bin_upper == 1.0:  # Include rightmost edge for the last bin
            in_bin = (y_confs >= bin_lower) & (y_confs <= bin_upper)
        else:
            in_bin = (y_confs >= bin_lower) & (y_confs < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_confs[in_bin])
            bin_ece = np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            ece += bin_ece

            # print(f"Bin {i + 1}:")
            # print(f"  Range: {bin_lower:.2f} - {bin_upper:.2f}")
            # print(f"  Proportion in bin: {prop_in_bin:.2f} ({prop_in_bin * n} samples)")
            # print(f"  Accuracy in bin: {accuracy_in_bin:.2f}")
            # print(f"  Average confidence in bin: {avg_confidence_in_bin:.2f}")

    print(f"Total ECE: {ece:.4f}")
    return ece
# y_true:correct , y_confs: confidence score
def compute_conf_metrics(y_true, y_confs, elicitation_method):
    result_metrics = {'method': elicitation_method}
    y_true = np.array([1 if correct else 0 for correct in y_true])
    # confidence scores already normalized 0 to 1
    y_confs = np.array(y_confs)
    y_true, y_confs = filter_out_of_range(y_true, y_confs)

    # Compute accuracy
    accuracy = sum(y_true) / len(y_true)
    print(len(y_true))
    print("Accuracy:", accuracy)
    result_metrics['acc'] = accuracy

    # Check if confidence scores are in the range [0, 1]
    print("Confidence values before assertion:", y_confs)
    print("Check if all values are between 0 and 1:", all([0 <= x <= 1 for x in y_confs]))

    out_of_range_values = [x for x in y_confs if not (0 <= x <= 1)]
    if out_of_range_values:
        print("Out-of-range confidence values:", out_of_range_values)

    assert all([0 <= x <= 1 for x in y_confs]), y_confs

    # Convert  to numpy arrays
    y_confs, y_true = np.array(y_confs), np.array(y_true)
    try:
        # Compute AUROC
        roc_auc = roc_auc_score(y_true, y_confs)
        print(f"ROC AUC score: {roc_auc}")
        result_metrics['auroc'] = roc_auc
    except ValueError as e:
        print(f"Error computing ROC AUC score: {e}")

    try:
        # Compute AUPRC
        auprc = average_precision_score(y_true, y_confs)
        print(f"AUC PRC score: {auprc}")
        result_metrics['auprc'] = auprc
    except ValueError as e:
        print(f"Error computing AUC PRC score: {e}")

    try:
        # AUPRC-Positive
        auprc_p = average_precision_score(y_true, y_confs)
        print(f"AUC PRC Positive score: {auprc_p}")
        result_metrics['auprc_p'] = auprc_p
    except ValueError as e:
        print(f"Error computing AUC PRC Positive score: {e}")

    try:
        # AUPRC-Negative
        auprc_n = average_precision_score(1 - y_true, 1 - y_confs)
        print(f"AUC PRC Negative score: {auprc_n}")
        result_metrics['auprc_n'] = auprc_n
    except ValueError as e:
        print(f"Error computing AUC PRC Negative score: {e}")

    try:
        # Compute ECE
        ece_score = manual_ece(y_true, y_confs, 20)
        print(f"ECE score: {ece_score}")
        result_metrics['ece'] = ece_score
    except ValueError as e:
        print(f"Error computing ECE score: {e}")

    # Compute ECE
    # n_bins = 20
    # ece = ECE(n_bins)
    # ece_score = ece.measure(y_confs, y_true)
    # print("ECE score:", ece_score)
    # result_metrics['ece'] = ece_score

    result_metrics.update({'acc': accuracy, 'auroc': roc_auc, 'auprc': auprc, 'auprd-p': auprc_p, 'auprc-n': auprc_n, 'ece': ece_score})

    return result_metrics

#%%
