# Filename: data_processing.py
import json
import pandas as pd
import os


def generate_csv_filename(json_file_path):
    directory_path = os.path.dirname(json_file_path)
    base_directory = directory_path.replace('/', '_')
    if not isinstance(base_directory, str):
        base_directory = str(base_directory)

    new_filename = os.path.join('cleaned_data', 'cleaned_' + base_directory + '.csv')
    return new_filename


def save_data(df, csv_file_path):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")
