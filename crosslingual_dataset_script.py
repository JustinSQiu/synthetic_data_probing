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
    'portuguese': 'pt',
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


def convert_formal_crosslingual():
    root_folder = 'crosslingual_raw/XFormal_crosslingual'
    output_folder = 'crosslingual_raw_output/formal'
    # pairwise_output_folder = 'pairwise_crosslingual_raw_output/formal'

    for lang in os.listdir(root_folder):
        formal_file = os.path.join(root_folder, lang, 'formal')
        informal_file = os.path.join(root_folder, lang, 'informal')
        tsv_file = os.path.join(output_folder, f'{language_map[lang.lower()]}.tsv')
        # pairwise_tsv_file = os.path.join(pairwise_output_folder, f'{language_map[lang.lower()]}.tsv')
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

        # pairwise_df = get_paired_tsv(df)
        # pairwise_df['style_type'] = 'formality'
        # pairwise_df['language'] = language_map[lang]
        # pairwise_df.to_csv(pairwise_tsv_file, sep='\t', index=False)
        print(f'TSV file saved for {lang} at {tsv_file}')

convert_formal_crosslingual()


def merge_csvs():
    all_files = glob.glob('crosslingual_raw_output/*/*.tsv')
    df = pd.concat((pd.read_csv(file, sep='\t') for file in all_files), ignore_index=True)
    dataset = DatasetDict({
        'train': Dataset.from_pandas(df)
    })
    return dataset

dataset = merge_csvs()
# dataset.push_to_hub('StyleDistance/multilingual_stel')
dataset.push_to_hub('crosslingual_stel')