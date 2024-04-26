import numpy as np
import pandas as pd


def calculate_ece(df, n_bins=10):
    # Create equally sized bins
    df['bin'] = pd.qcut(df['confidence'], q=n_bins, labels=False, duplicates='drop')

    # Calculate ECE
    ece = 0
    total_samples = len(df)

    for bin in range(n_bins):
        bin_df = df[df['bin'] == bin]
        if not bin_df.empty:
            bin_size = len(bin_df)
            prob_in_bin = bin_size / total_samples
            accuracy_in_bin = bin_df['correct'].mean()
            avg_confidence_in_bin = bin_df['confidence'].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece


dataFrame = pd.read_csv('cleaned_gsm8k_few_shot_cot_number.csv')
ece_value = calculate_ece(dataFrame)
print(f"Expected Calibration Error (ECE): {ece_value}")


#%%
