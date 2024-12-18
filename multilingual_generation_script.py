import pandas as pd
import os
import glob
import random
from datasets import load_dataset

def convert_complex_multisim():
    root_folder = 'MultiSim/data'
    output_folder = 'multilingual_stel/complex'
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

            output_file = f'multilingual_stel/formal/{lang.lower()}_formal.tsv'
            df.to_csv(output_file, sep='\t', index=False)
            print(f"TSV file saved for {lang} at {output_file}")

    base_dir = 'XFormal'
    create_tsv_for_each_language(base_dir, num_samples=100)

def convert_toxic_paradetox():
    ds = load_dataset("textdetox/multilingual_paradetox")

    def sample_and_create_tsv(dataset, language, num_samples=100):
        """Sample the dataset and create a TSV file for each language."""
        data = {
            "anchor1": [],
            "anchor2": [],
            "alternative1": [],
            "alternative2": []
        }

        # Get the number of rows in the dataset for the specific language
        num_rows = dataset.num_rows
        
        # Sample indices (2 rows for each output line)
        sampled_indices = random.sample(range(num_rows), min(num_samples * 2, num_rows))
        
        for i in range(0, len(sampled_indices), 2):
            idx_anchor = sampled_indices[i]
            idx_alternative = sampled_indices[i + 1]
            
            # Get anchor sentences
            anchor1 = dataset[idx_anchor]['toxic_sentence']
            anchor2 = dataset[idx_anchor]['neutral_sentence']
            
            # Get alternative sentences
            alternative1 = dataset[idx_alternative]['toxic_sentence']
            alternative2 = dataset[idx_alternative]['neutral_sentence']
            
            # Append to the data dictionary
            data["anchor1"].append(anchor1)
            data["anchor2"].append(anchor2)
            data["alternative1"].append(alternative1)
            data["alternative2"].append(alternative2)

        # Create a pandas DataFrame
        df = pd.DataFrame(data)
        df['correct_alternative'] = 1
        df['style_type'] = 'toxic'
        df['language'] = lang.lower()

        # Write the DataFrame to a TSV file for the specific language
        output_file = f'multilingual_stel/toxic/{language}_toxic.tsv'  # File name based on language
        df.to_csv(output_file, sep='\t', index=False)
        print(f"TSV file saved for {language} at {output_file}")

    # Loop through each language in the dataset and create TSV files
    for lang in ds:
        sample_and_create_tsv(ds[lang], lang, num_samples=100)

def convert_positive_indian():
    output_dir = 'multilingual_stel/positive'
    def sample_and_create_tsv(input_csv, output_tsv, lang, num_samples=100):
        """Convert a CSV into a TSV with sampled anchor and alternative pairs."""
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv, delimiter=',')

        # Ensure enough rows to sample
        if len(df) < num_samples * 2:
            print(f"Warning: Not enough data in {input_csv} for 100 samples.")
            num_samples = len(df) // 2

        # Sample 2 * num_samples unique row indices
        sampled_indices = random.sample(range(len(df)), num_samples * 2)

        # Prepare data storage for sampled rows
        data = {
            "anchor1": [],
            "anchor2": [],
            "alternative1": [],
            "alternative2": []
        }

        # Sample anchor and alternative pairs
        print(df)
        for i in range(0, len(sampled_indices), 2):
            idx_anchor = sampled_indices[i]
            idx_alt = sampled_indices[i + 1]

            # Extract anchor and alternative pairs
            anchor1 = df.iloc[idx_anchor]["NEGATIVE"]
            anchor2 = df.iloc[idx_anchor]["POSITIVE"]
            alternative1 = df.iloc[idx_alt]["NEGATIVE"]
            alternative2 = df.iloc[idx_alt]["POSITIVE"]

            # Append to the data dictionary
            data["anchor1"].append(anchor1)
            data["anchor2"].append(anchor2)
            data["alternative1"].append(alternative1)
            data["alternative2"].append(alternative2)
            data['correct_alternative'] = 1
            data['style_type'] = 'simplicity'
            data['language'] = lang

        # Create a DataFrame from the sampled data
        output_df = pd.DataFrame(data)

        # Write the DataFrame to a TSV file
        output_df.to_csv(output_tsv, sep='\t', index=False)
        print(f"TSV file saved at {output_tsv}")

    def process_multilingual_datasets(base_dir):
        for lang in os.listdir(base_dir):
            lang_dir = os.path.join(base_dir, lang)
            try:
                csv_files = [f for f in os.listdir(lang_dir) if f.endswith(".csv")]
                input_csv = os.path.join(lang_dir, csv_files[0])
                output_tsv = os.path.join(output_dir, f"{lang}_positive.tsv")

                sample_and_create_tsv(input_csv, output_tsv, lang)
            except:
                pass
    # Define the base directory containing language folders
    base_dir = "multilingual-tst-datasets"

    # Process all language folders
    process_multilingual_datasets(base_dir)


# convert_complex_multisim()
# convert_formal_xformal()
convert_toxic_paradetox()
# convert_positive_indian()
