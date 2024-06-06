import os


def generate_csv_filename(json_file_path):
    absolute_path = os.path.abspath(json_file_path)
    parts = absolute_path.split(os.sep)
    if 'openAi' in parts:
        start_index = parts.index('openAi')
    elif 'mistral' in parts:
        start_index = parts.index('mistral')
    elif 'fastchat' in parts:
        start_index = parts.index('fastchat')
    elif 'llama2' in parts:
        start_index = parts.index('llama2')
    else:
        start_index = 0
    path_parts = parts[start_index:-1]
    base_filename = 'cleaned_' + '_'.join(path_parts) + '.csv'

    new_filename = os.path.join('..', 'cleaned_data', base_filename)
    return new_filename


def save_data(df, csv_file_path):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")
