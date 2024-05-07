import pandas as pd


def save_metrics(metrics, filename):
    # Save metrics to a CSV file
    pd.DataFrame([metrics]).to_csv(filename, index=False)

