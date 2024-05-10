import os


def generate_csv_filename(json_file_path):
    absolute_path = os.path.abspath(json_file_path)
    parts = absolute_path.split(os.sep)
    start_index = parts.index('openAi') if 'openAi' in parts else 0
    path_parts = parts[start_index:-1]
    base_filename = 'cleaned_' + '_'.join(path_parts) + '.csv'

    new_filename = os.path.join('..', 'cleaned_data', base_filename)
    return new_filename


def save_data(df, csv_file_path):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")
