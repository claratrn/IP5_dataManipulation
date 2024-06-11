import json
import pandas as pd
import os

def prepare_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        results = data['results']

    answered_records = []
    unanswered_records = []

    for item in results:
        response = item.get("response", {})
        final_answer = response.get("final answer")
        confidence_str = response.get("confidence")

        # Convert provided and true answers to strings and strip any extra whitespace
        provided_answer = str(item.get("answer", "")).strip()
        true_answer = str(item.get("true_answer", "")).strip()

        if final_answer is None or confidence_str is None:
            # Record details of unanswered questions
            unanswered_records.append({
                "index": item["index"],
                "question": item["question"],
                "answer": item["answer"],
                "provided answer": provided_answer,
                "expected true answer": true_answer,
                "response details": json.dumps(response)  # Store incomplete response as JSON string
            })
        else:
            try:
                # Attempt to parse confidence and prepare answered record
                confidence = float(confidence_str) / 100  # Convert confidence to a proper decimal
                answered_records.append({
                    "index": item["index"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "predicted label": final_answer,
                    "ground truth": true_answer,
                    "confidence": confidence,
                    "correct": final_answer == true_answer
                })
            except ValueError:
                print(f"ValueError processing item index {item['index']}: Check confidence value")

    # Create DataFrames
    df_answered = pd.DataFrame(answered_records)
    df_unanswered = pd.DataFrame(unanswered_records)

    # Generate paths for saving CSV files
    answered_path = file_path.replace(".json", "_answered.csv")
    unanswered_path = file_path.replace(".json", "_unanswered.csv")

    # Save DataFrames to CSV
    df_answered.to_csv(answered_path, index=False)
    df_unanswered.to_csv(unanswered_path, index=False)

    print(f"Data saved: {answered_path}, {unanswered_path}")
    return df_answered, df_unanswered

# Example usage
json_file_path = ("../results/openAi/mmlu/few_shot_cot_law/complete.json")
prepare_data(json_file_path)

#%%
