import json
import logging

import pandas as pd

from ece import calculate_ece

# Your JSON-like data

file_path = "gsm8k/few_shot_cot_number/20240425-231430863620_856fdf912d3f42a9909ce199385211dc.json"

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
print(df)
csv_file_path = 'calibration.csv'

df.to_csv(csv_file_path, index=False)
logging.info(f'DataFrame saved successfully to {csv_file_path}')

#%%
