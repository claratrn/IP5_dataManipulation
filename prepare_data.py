import json
import logging

import pandas as pd

from ece import calculate_ece
from tools.generate_csv import save_data, generate_csv_filename

# Your JSON-like data

json_file_path = "gsm8k/few_shot_cot_number/20240425-231430863620_856fdf912d3f42a9909ce199385211dc.json"


def prepare_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        results = data['results']

    # Create DataFrame
    df = pd.DataFrame({
        "confidence": [int(item["response"]["confidence"]) for item in results],
        "predicted label": [item["response"]["final answer"] for item in results],
        "true label": [item["true_answer"] for item in results]
    })

    df['correct'] = df['predicted label'] == df['true label']

    csv_file_path = generate_csv_filename(json_file_path)
    save_data(df, csv_file_path)


prepare_data(json_file_path)
