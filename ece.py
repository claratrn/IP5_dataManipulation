import numpy as np
import pandas as pd


def calculate_ece(df, n_bins=10):
    # Create bins for the confidence levels
    df['confidence_bin'] = pd.cut(df['confidence'], bins=np.linspace(0, 100, n_bins + 1), include_lowest=True,
                                  labels=False)

    # Group by the confidence bin
    grouped = df.groupby('confidence_bin').agg(
        avg_confidence=('confidence', 'mean'),
        accuracy=('correct', 'mean'),
        bin_count=('index', 'size')
    )

    # Calculate ECE as the weighted average of absolute differences between avg_confidence and accuracy
    ece = (grouped['bin_count'] * abs(grouped['avg_confidence'] / 100 - grouped['accuracy'])).sum() / df[
        'index'].count()

    return ece
