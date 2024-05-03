import pandas as pd
from netcal.metrics import ECE
import numpy as np

file_path = "cleaned_data/cleaned_commonsense_qa_multi_step_multiple_choice.csv"
dataFrame = pd.read_csv(file_path)

# Scale confidence values from 0-100 to 0-1
dataFrame['confidence'] = dataFrame['confidence'] / 100


# y_true collection of data that is true
# y_conf confidence score associated with each prediction

def calculate_ece(true_labels, confidence_scores, n_bins=10):
    ece = ECE(n_bins)
    score = ece.measure(np.array(confidence_scores), np.array(true_labels))
    return score


true = dataFrame['correct'].values
confs = dataFrame['confidence'].values

ece_score = calculate_ece(true, confs)
print("ECE:", ece_score)

ECE: 0.33333333333333337
# %%
