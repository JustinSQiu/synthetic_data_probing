import pandas as pd
import os
import random
import glob

def process_csv_to_tsv(input_csv, output_tsv, num_samples=100):
    df = pd.read_csv(input_csv)
    if len(df) < num_samples * 2:
        raise ValueError("Not enough rows to sample 100 pairs.")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_sampled = df.iloc[:num_samples*2]
    rows = []
    for i in range(0, num_samples*2, 2):
        anchor1 = df_sampled.iloc[i]['original']
        anchor2 = df_sampled.iloc[i]['simple']
        alternative1 = df_sampled.iloc[i+1]['original']
        alternative2 = df_sampled.iloc[i+1]['simple']
        rows.append([anchor1, anchor2, alternative1, alternative2])
    
    tsv_df = pd.DataFrame(rows, columns=['anchor1', 'anchor2', 'alternative1', 'alternative2'])
    tsv_df.to_csv(output_tsv, sep='\t', index=False)

def process_language_folders(root_folder, output_folder):
    for language_folder in os.listdir(root_folder):
        language_path = os.path.join(root_folder, language_folder)
        if os.path.isdir(language_path):
            csv_files = glob.glob(os.path.join(language_path, '*_train.csv'))
            for csv_file in csv_files:
                tsv_file = os.path.join(output_folder, f'{language_folder}_sampled.tsv')
                try:
                    print(f'Processing {csv_file}...')
                    process_csv_to_tsv(csv_file, tsv_file)
                    print(f'Successfully saved to {tsv_file}')
                except ValueError as e:
                    print(f"Skipping {csv_file}: {e}")

root_folder = 'MultiSim/data'
output_folder = 'multilingual_output'

os.makedirs(output_folder, exist_ok=True)
process_language_folders(root_folder, output_folder)
