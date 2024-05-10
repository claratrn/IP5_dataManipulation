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
        if "response" in item and "true_answer" in item and item["response"].get("final answer") is not None:
            # Ensure 'confidence' is a valid integer or skip
            confidence_str = item["response"].get("confidence")
            if confidence_str is not None:
                try:
                    confidence = int(confidence_str)
                    final_answer = item["response"]["final answer"]
                    true_answer = str(item["true_answer"]).strip()

                    records.append({
                        "confidence": confidence / 100,
                        "predicted label": final_answer,
                        "ground truth": true_answer,
                        "correct": final_answer == true_answer
                    })
                except ValueError:
                    continue

    # Create DataFrame
    df = pd.DataFrame(records, columns=["confidence", "predicted label", "ground truth", "correct"])

    # Generate filename and save data to CSV
    csv_file_path = generate_csv_filename(file_path)
    save_data(df, csv_file_path)


json_file_path = ("../results/openAi/commonsense_qa/zero_shot_cot/20240502-230935659971_15429b3095584d7090a87a04bf3e9ba0.json")
print("Current Working Directory: ", os.getcwd())
prepare_data(json_file_path)


#%%
