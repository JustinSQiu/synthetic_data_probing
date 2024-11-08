import pandas as pd
import os
import glob
from datasets import load_dataset
from datasets import DatasetDict, Dataset
import itertools

language_map = {
    'German': 'de',
    'german': 'de',
    'English': 'en',
    'english': 'en',
    'refined-english': 'en',
    'Refined-English': 'en',
    'French': 'fr',
    'french': 'fr',
    'Italian': 'it',
    'italian': 'it',
    'Japanese': 'ja',
    'japanese': 'ja',
    'Brazilian Portuguese': 'pt-br',
    'brazilian portuguese': 'pt-br',
    'brazilian_portuguese': 'pt-br',
    'Brazilian_Portuguese': 'pt-br',
    'Russian': 'ru',
    'russian': 'ru',
    'Slovene': 'sl',
    'slovene': 'sl',
    'Spanish': 'es',
    'spanish': 'es',
    'Hindi': 'hi',
    'hindi': 'hi',
    'Ukrainian': 'uk',
    'ukrainian': 'uk',
    'Chinese': 'zh',
    'chinese': 'zh',
    'Bengali': 'bn',
    'bengali': 'bn',
    'Magahi': 'mag',
    'magahi': 'mag',
    'Malayalam': 'ml',
    'malayalam': 'ml',
    'Marathi': 'mr',
    'marathi': 'mr',
    'Odia': 'or',
    'odia': 'or',
    'Punjabi': 'pa',
    'punjabi': 'pa',
    'Telugu': 'te',
    'telugu': 'te',
    'Urdu': 'ur',
    'urdu': 'ur',
    'Amharic': 'am',
    'amharic': 'am',
    'Arabic': 'ar',
    'arabic': 'ar',
    'am': 'am',
    'ru': 'ru',
    'hi': 'hi',
    'en': 'en',
    'uk': 'uk',
    'ar': 'ar',
    'es': 'es',
    'zh': 'zh',
    'de': 'de'
}

def get_paired_tsv(df):
    pairwise_combinations = list(itertools.combinations(df.index, 2))
    pairwise_df = pd.DataFrame({
        'anchor_1': [df.loc[i, 'positive'] for i, j in pairwise_combinations],
        'alternative_1': [df.loc[i, 'negative'] for i, j in pairwise_combinations],
        'anchor_2': [df.loc[j, 'positive'] for i, j in pairwise_combinations],
        'alternative_2': [df.loc[j, 'negative'] for i, j in pairwise_combinations],
    })
    return pairwise_df


def convert_complex_multisim():
    languages = ['English', 'Italian', 'French', 'Japanese', 'Brazilian Portuguese', 'German', 'Russian', 'Slovene']
    output_folder = 'multilingual_raw_output/complex'
    pairwise_output_folder = 'pairwise_multilingual_raw_output/complex'

    for lang in languages:
        ds = load_dataset('MichaelR207/MultiSim', lang, trust_remote_code=True)
        tsv_file = os.path.join(output_folder, f'{language_map[lang]}.tsv')
        pairwise_tsv_file = os.path.join(pairwise_output_folder, f'{language_map[lang]}.tsv')
        rows = []
        for i in range(100):
            anchor1 = ds['train'][i]['original']
            anchor2 = ds['train'][i]['simple']['simplifications'][0]
            rows.append([anchor1, anchor2])
        df = pd.DataFrame(rows, columns=['positive', 'negative'])
        df['style_type'] = 'simplicity'
        df['language'] = language_map[lang]
        df.to_csv(tsv_file, sep='\t', index=False)

        pairwise_df = get_paired_tsv(df)
        pairwise_df['style_type'] = 'simplicity'
        pairwise_df['language'] = language_map[lang]
        pairwise_df.to_csv(pairwise_tsv_file, sep='\t', index=False)
        print(f'TSV file saved for {lang} at {tsv_file}')

def convert_formal_xformal():
    root_folder = 'multilingual_raw/XFormal'
    output_folder = 'multilingual_raw_output/formal'
    pairwise_output_folder = 'pairwise_multilingual_raw_output/formal'

    for lang in os.listdir(root_folder):
        formal_file = os.path.join(root_folder, lang, 'formal0')
        informal_file = os.path.join(root_folder, lang, 'informal')
        tsv_file = os.path.join(output_folder, f'{language_map[lang.lower()]}.tsv')
        pairwise_tsv_file = os.path.join(pairwise_output_folder, f'{language_map[lang.lower()]}.tsv')
        if not os.path.exists(formal_file) or not os.path.exists(informal_file):
            print(f'Warning: Missing files for {lang}!')
            continue
        with open(formal_file, 'r', encoding='utf-8') as f:
            formal_lines = [line.strip() for line in f.readlines()]
        with open(informal_file, 'r', encoding='utf-8') as f:
            informal_lines = [line.strip() for line in f.readlines()]

        rows = []
        for i in range(100):
            anchor1 = formal_lines[i]
            anchor2 = informal_lines[i]
            rows.append([anchor1, anchor2])

        df = pd.DataFrame(rows, columns=['positive', 'negative'])
        df['style_type'] = 'formality'
        df['language'] = language_map[lang.lower()]
        df.to_csv(tsv_file, sep='\t', index=False)

        pairwise_df = get_paired_tsv(df)
        pairwise_df['style_type'] = 'formality'
        pairwise_df['language'] = language_map[lang]
        pairwise_df.to_csv(pairwise_tsv_file, sep='\t', index=False)
        print(f'TSV file saved for {lang} at {tsv_file}')


def convert_toxic_paradetox():
    dataset = load_dataset('textdetox/multilingual_paradetox', trust_remote_code=True)
    output_folder = 'multilingual_raw_output/toxic'
    pairwise_output_folder = 'pairwise_multilingual_raw_output/toxic'
    for lang in dataset:
        tsv_file = os.path.join(output_folder, f'{language_map[lang.lower()]}.tsv')
        pairwise_tsv_file = os.path.join(pairwise_output_folder, f'{language_map[lang.lower()]}.tsv')
        data = []
        for i in range(100):
            anchor1 = dataset[lang][i]['toxic_sentence']
            anchor2 = dataset[lang][i]['neutral_sentence']
            data.append([anchor1, anchor2])

        df = pd.DataFrame(data, columns=['positive', 'negative'])
        df['style_type'] = 'toxicity'
        df['language'] = language_map[lang.lower()]
        df.to_csv(tsv_file, sep='\t', index=False)

        pairwise_df = get_paired_tsv(df)
        pairwise_df['style_type'] = 'toxicity'
        pairwise_df['language'] = language_map[lang.lower()]
        pairwise_df.to_csv(pairwise_tsv_file, sep='\t', index=False)
        print(f'TSV file saved for {lang} at {output_folder}')

def convert_positive_indian():
    base_dir = 'multilingual_raw/multilingual-tst-datasets'
    output_folder = 'multilingual_raw_output/positive'
    pairwise_output_folder = 'pairwise_multilingual_raw_output/positive'

    for lang in os.listdir(base_dir):
        lang_dir = os.path.join(base_dir, lang)
        csv_files = [f for f in os.listdir(lang_dir) if f.endswith('.csv')]
        if not csv_files:
            continue
        input_csv = os.path.join(lang_dir, csv_files[0])
        tsv_file = os.path.join(output_folder, f'{language_map[lang.lower()]}.tsv')
        pairwise_tsv_file = os.path.join(pairwise_output_folder, f'{language_map[lang.lower()]}.tsv')
        csv_data = pd.read_csv(input_csv, delimiter=',')
        data = []
        for i in range(100):
            anchor1 = csv_data.iloc[i]['NEGATIVE']
            anchor2 = csv_data.iloc[i]['POSITIVE']
            data.append([anchor1, anchor2])

        df = pd.DataFrame(data, columns=['positive', 'negative'])
        df['style_type'] = 'positivity'
        df['language'] = language_map[lang]
        df.to_csv(tsv_file, sep='\t', index=False)

        pairwise_df = get_paired_tsv(df)
        pairwise_df['style_type'] = 'positivity'
        pairwise_df['language'] = language_map[lang.lower()]
        pairwise_df.to_csv(pairwise_tsv_file, sep='\t', index=False)
        print(f'TSV file saved for {lang} at {tsv_file}')

def merge_csvs():
    all_files = glob.glob('multilingual_raw_output/*/*.tsv')
    df = pd.concat((pd.read_csv(file, sep='\t') for file in all_files), ignore_index=True)
    dataset = DatasetDict({
        'train': Dataset.from_pandas(df)
    })
    return dataset

# convert_complex_multisim()
# convert_formal_xformal()
# convert_toxic_paradetox()
# convert_positive_indian()
dataset = merge_csvs()
dataset.push_to_hub('StyleDistance/multilingual_stel')
dataset.push_to_hub('multilingual_stel')