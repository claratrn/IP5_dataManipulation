import json
import pandas as pd
import os

from tools.generate_csv import generate_csv_filename, save_data


def prepare_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        results = data['results']

    records = []
    for item in results:
        # Ensure all needed keys exist and are not None
        if "response" in item and "true_answer" in item and item["response"].get("final answer") is not None:
            # Ensure 'confidence' is a valid integer or skip
            confidence_str = item["response"].get("confidence")
            if confidence_str is not None:
                try:
                    confidence = int(confidence_str)  # Convert to int and catch any conversion errors
                    final_answer = item["response"]["final answer"]  # We already checked it's not None
                    true_answer = str(item["true_answer"]).strip()

                    # Add to records if all checks are passed
                    records.append({
                        "confidence": confidence,
                        "predicted label": final_answer,
                        "true label": true_answer,
                        "correct": final_answer == true_answer
                    })
                except ValueError:
                    continue  # Skip this item if confidence can't be converted to int

    # Create DataFrame
    df = pd.DataFrame(records, columns=["confidence", "predicted label", "true label", "correct"])

    # Generate filename and save data to CSV
    csv_file_path = generate_csv_filename(file_path)
    save_data(df, csv_file_path)


json_file_path = "commonsense_qa/few_shot_multiple_choice_cot/20240503-003901054590_74916994c6714d68b83d27cf233b39d3.json"
print("Current Working Directory: ", os.getcwd())
prepare_data(json_file_path)

#%%
