import pandas as pd


def quantify_missing_data(df_answered, df_unanswered, output_file):
    total_entries = len(df_answered) + len(df_unanswered)
    missing_data_ratio = len(df_unanswered) / total_entries
    print(f"Total entries: {total_entries}")
    print(f"Answered entries: {len(df_answered)}")
    print(f"Unanswered entries: {len(df_unanswered)}")
    print(f"Missing data ratio: {missing_data_ratio:.2%}")

    data = {
        "Metric": ["Total entries", "Answered entries", "Unanswered entries", "Missing data ratio"],
        "Value": [total_entries, len(df_answered), len(df_unanswered), f"{missing_data_ratio:.2%}"]
    }
    # Create a DataFrame
    df_summary = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df_summary.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    return df_summary


def assess_distribution(df_unanswered):
    unanswered_summary = df_unanswered.groupby('question').size().reset_index(name='count')
    print(unanswered_summary)


def calculate_metrics(df):
    total = len(df)
    correct = df['correct'].sum()
    accuracy = correct / total
    return accuracy


def compare_metrics(df_answered, df_unanswered):
    accuracy_answered = calculate_metrics(df_answered)
    print(f"Accuracy with answered data: {accuracy_answered:.2%}")

    df_all = pd.concat([df_answered, df_unanswered.assign(correct=False)])
    accuracy_all = calculate_metrics(df_all)
    print(f"Accuracy with all data: {accuracy_all:.2%}")


def sensitivity_analysis(df_answered, missing_data_ratios):
    original_accuracy = calculate_metrics(df_answered)
    print(f"Original accuracy: {original_accuracy:.2%}")

    for ratio in missing_data_ratios:
        num_missing = int(len(df_answered) * ratio)
        df_missing_simulated = df_answered.sample(n=num_missing, random_state=1).assign(correct=False)
        df_combined = pd.concat([df_answered, df_answered.append(df_missing_simulated)])
        accuracy_simulated = calculate_metrics(df_combined)
        print(f"Simulated accuracy with {ratio:.2%} missing data: {accuracy_simulated:.2%}")


df_answered = pd.read_csv(
    "../results/openAi/commonsense_qa/vanilla/vanilla_common_fixed_answered.csv")
df_unanswered = pd.read_csv(
    "../results/openAi/commonsense_qa/vanilla/vanilla_common_fixed_unanswered.csv")

summary_file = "missingVal/OpenAi/CommonsenseQA/common_missing_value_vanilla_summary.csv"
df_summary = quantify_missing_data(df_answered, df_unanswered, summary_file)


#%%
