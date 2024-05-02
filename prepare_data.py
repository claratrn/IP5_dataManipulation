import json
import pandas as pd
import os
from tools.generate_csv import generate_csv_filename, save_data


def prepare_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        results = data['results']

    # Create DataFrame
    df = pd.DataFrame({
        "confidence": [int(item["response"]["confidence"]) for item in results if "confidence" in item["response"]],
        "predicted label": [item["response"]["final answer"] for item in results if "final answer" in item["response"]],
        "true label": [str(item["true_answer"]).strip() for item in results if "true_answer" in item]
    })

    # Drop rows where any of the required fields are missing
    df = df.dropna(subset=['confidence', 'predicted label', 'true label'])

    # Calculate correctness
    df['correct'] = df['predicted label'] == df['true label']

    # Generate filename and save data to CSV
    csv_file_path = generate_csv_filename(file_path)
    save_data(df, csv_file_path)


json_file_path = "commonsense_qa/multi_step_multiple_choice/20240426-003728888189_7dc90fff41e444969f519a1abb1922e6.json"
print("Current Working Directory: ", os.getcwd())
prepare_data(json_file_path)

#%%
