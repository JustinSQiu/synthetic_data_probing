import pandas as pd
import os
import glob
import random

def convert_complex_multisim():
    root_folder = 'MultiSim/data'
    output_folder = 'multilingual_output/complex'
    os.makedirs(output_folder, exist_ok=True)

    for language_folder in os.listdir(root_folder):
        language_path = os.path.join(root_folder, language_folder)
        if os.path.isdir(language_path):
            csv_files = glob.glob(os.path.join(language_path, '*_train.csv'))
            for csv_file in csv_files:
                tsv_file = os.path.join(output_folder, f'{language_folder.lower()}_complex.tsv')
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) < 100 * 2:
                        raise ValueError("Not enough rows to sample 100 pairs.")
                    
                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    df_sampled = df.iloc[:100*2]
                    rows = []
                    for i in range(0, 100*2, 2):
                        anchor1 = df_sampled.iloc[i]['original']
                        anchor2 = df_sampled.iloc[i]['simple']
                        alternative1 = df_sampled.iloc[i+1]['original']
                        alternative2 = df_sampled.iloc[i+1]['simple']
                        rows.append([anchor1, anchor2, alternative1, alternative2])
                    
                    tsv_df = pd.DataFrame(rows, columns=['anchor1', 'anchor2', 'alternative1', 'alternative2'])
                    tsv_df['correct_alternative'] = 1
                    tsv_df['style_type'] = 'simplicity'
                    tsv_df['language'] = language_folder
                    tsv_df.to_csv(tsv_file, sep='\t', index=False)
                except ValueError as e:
                    print(f"Skipping {csv_file}: {e}")

def convert_formal_xformal():
    def read_file_lines(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    def create_tsv_for_each_language(base_dir, num_samples=100):
        languages = os.listdir(base_dir)

        for lang in languages:
            formal_file = os.path.join(base_dir, lang, 'formal0')
            informal_file = os.path.join(base_dir, lang, 'informal')
            
            if not os.path.exists(formal_file) or not os.path.exists(informal_file):
                print(f"Warning: Missing files for {lang}!")
                continue

            formal_lines = read_file_lines(formal_file)
            informal_lines = read_file_lines(informal_file)

            data = {
                "anchor1": [],
                "anchor2": [],
                "alternative1": [],
                "alternative2": []
            }

            sampled_indices = random.sample(range(len(formal_lines)), min(num_samples * 2, len(formal_lines)))
            for i in range(0, len(sampled_indices), 2):
                idx_anchor = sampled_indices[i]
                idx_alt = sampled_indices[i + 1]
                anchor1 = formal_lines[idx_anchor]
                anchor2 = informal_lines[idx_anchor]
                alternative1 = formal_lines[idx_alt]
                alternative2 = informal_lines[idx_alt]
                data["anchor1"].append(anchor1)
                data["anchor2"].append(anchor2)
                data["alternative1"].append(alternative1)
                data["alternative2"].append(alternative2)

            df = pd.DataFrame(data)
            df['correct_alternative'] = 1
            df['style_type'] = 'formal'
            df['language'] = lang.lower()

            output_file = f'multilingual_output/formal/{lang.lower()}_formal.tsv'
            df.to_csv(output_file, sep='\t', index=False)
            print(f"TSV file saved for {lang} at {output_file}")

    base_dir = 'XFormal'
    create_tsv_for_each_language(base_dir, num_samples=100)


# convert_complex_multisim()
convert_formal_xformal()