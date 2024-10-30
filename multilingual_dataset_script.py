import pandas as pd
import os
import glob
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset

def convert_complex_multisim():
    root_folder = 'multilingual_raw/MultiSim/data'
    output_folder = 'multilingual_raw_output/complex'

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
                    for i in range(100):
                        anchor1 = df_sampled.iloc[i]['original']
                        anchor2 = df_sampled.iloc[i]['simple']
                        rows.append([anchor1, anchor2])
                    
                    tsv_df = pd.DataFrame(rows, columns=['positive', 'negative'])
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
                "positive": [],
                "negative": [],
            }

            for i in range(100):
                anchor1 = formal_lines[i]
                anchor2 = informal_lines[i]
                data["positive"].append(anchor1)
                data["negative"].append(anchor2)

            df = pd.DataFrame(data, columns=['positive', 'negative'])
            df['style_type'] = 'formal'
            df['language'] = lang.lower()

            output_file = f'multilingual_raw_output/formal/{lang.lower()}_formal.tsv'
            df.to_csv(output_file, sep='\t', index=False)
            print(f"TSV file saved for {lang} at {output_file}")

    base_dir = 'multilingual_raw/XFormal'
    create_tsv_for_each_language(base_dir, num_samples=100)

def convert_toxic_paradetox():
    ds = load_dataset("textdetox/multilingual_paradetox")

    def sample_and_create_tsv(dataset, language, num_samples=100):
        """Sample the dataset and create a TSV file for each language."""
        data = {
            "positive": [],
            "negative": [],
        }        
        for i in range(100):
            anchor1 = dataset[i]['toxic_sentence']
            anchor2 = dataset[i]['neutral_sentence']
            data["positive"].append(anchor1)
            data["negative"].append(anchor2)

        # Create a pandas DataFrame
        df = pd.DataFrame(data, columns=['positive', 'negative'])
        df['style_type'] = 'toxic'
        df['language'] = lang.lower()

        # Write the DataFrame to a TSV file for the specific language
        output_file = f'multilingual_raw_output/toxic/{language}_toxic.tsv'  # File name based on language
        df.to_csv(output_file, sep='\t', index=False)
        print(f"TSV file saved for {language} at {output_file}")

    # Loop through each language in the dataset and create TSV files
    for lang in ds:
        sample_and_create_tsv(ds[lang], lang, num_samples=100)

def convert_positive_indian():
    output_dir = 'multilingual_raw_output/positive'
    def sample_and_create_tsv(input_csv, output_tsv, lang, num_samples=100):
        """Convert a CSV into a TSV with sampled anchor and alternative pairs."""
        df = pd.read_csv(input_csv, delimiter=',')
        data = {
            "positive": [],
            "negative": [],
        }
        for i in range(100):
            anchor1 = df.iloc[i]["NEGATIVE"]
            anchor2 = df.iloc[i]["POSITIVE"]
            data["positive"].append(anchor1)
            data["negative"].append(anchor2)

        output_df = pd.DataFrame(data, columns=['positive', 'negative'])
        output_df['style_type'] = 'positivity'
        output_df['language'] = lang
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
    base_dir = "multilingual_raw/multilingual-tst-datasets"

    # Process all language folders
    process_multilingual_datasets(base_dir)


convert_complex_multisim()
convert_formal_xformal()
convert_toxic_paradetox()
convert_positive_indian()

def merge_csvs():
    # Step 1: Load and concatenate all CSVs
    all_files = glob.glob("multilingual_raw_output/*/*.tsv")
    df = pd.concat((pd.read_csv(file, sep='\t') for file in all_files), ignore_index=True)

    # Step 2: Create train/test split
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Step 3: Convert to Hugging Face Dataset
    # train_dataset = Dataset.from_pandas(train_df)
    # test_dataset = Dataset.from_pandas(test_df)

    # Step 4: Combine and push to Hugging Face
    dataset = DatasetDict({
        "data": Dataset.from_pandas(df)
        # "train": train_dataset,
        # "test": test_dataset
    })
    dataset.push_to_hub("multilingual_stel_test")

merge_csvs()