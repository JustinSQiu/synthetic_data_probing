import glob
import itertools
import os

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

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
    'de': 'de',
}


def get_paired_tsv(df):
    pairwise_combinations = list(itertools.combinations(df.index, 2))
    pairwise_df = pd.DataFrame(
        {
            'anchor_1': [df.loc[i, 'positive'] for i, j in pairwise_combinations],
            'alternative_1': [df.loc[i, 'negative'] for i, j in pairwise_combinations],
            'anchor_2': [df.loc[j, 'positive'] for i, j in pairwise_combinations],
            'alternative_2': [df.loc[j, 'negative'] for i, j in pairwise_combinations],
        }
    )
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


def convert_formal_crosslingual_stel():
    pairwise_output_folder = 'pairwise_multilingual_raw_output/formal_crosslingual_stel'
    dataset = load_dataset('justinsunqiu/crosslingual_stel', trust_remote_code=True)
    NUM_ROWS_PER_CATEGORY = 100
    paired_sentences = []
    for i in range(dataset.num_rows['train'] // NUM_ROWS_PER_CATEGORY):
        pos_texts = [pos_text for pos_text in dataset['train']['positive'][i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]]
        neg_texts = [neg_text for neg_text in dataset['train']['negative'][i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]]
        paired = [(pos, neg) for pos, neg in zip(pos_texts, neg_texts)]
        paired_sentences.append(tuple(paired))
    categories = []
    languages = []
    for i in range(dataset.num_rows['train'] // NUM_ROWS_PER_CATEGORY):
        languages.append(dataset['train']['language'][i * NUM_ROWS_PER_CATEGORY])
        for j in range(i + 1, dataset.num_rows['train'] // NUM_ROWS_PER_CATEGORY):
            categories.append(
                dataset['train']['style_type'][i * NUM_ROWS_PER_CATEGORY]
                + '_'
                + dataset['train']['language'][i * NUM_ROWS_PER_CATEGORY]
                + '-'
                + dataset['train']['language'][j * NUM_ROWS_PER_CATEGORY]
            )
    cnt = 0
    for i in range(len(paired_sentences)):
        paired_1 = paired_sentences[i]
        for j in range(i + 1, len(paired_sentences)):
            rows = []
            pairwise_tsv_file = os.path.join(pairwise_output_folder, f'{categories[cnt]}_stel.tsv')
            paired_2 = paired_sentences[j]
            for k in range(len(paired_1)):
                anchor_pos, anchor_neg = paired_1[k]  # L1C1S1, L1C1S2
                for l in range(len(paired_2)):
                    if k == l:
                        continue  # Skip when content is equal
                    alt_pos, alt_neg = paired_2[l]  # L2C2S1 L2C2S2
                    rows.append([anchor_pos, anchor_neg, alt_pos, alt_neg])
            df = pd.DataFrame(rows, columns=['anchor_1', 'alternative_1', 'anchor_2', 'alternative_2'])
            df['style_type'] = 'formality'
            df['language_1'] = languages[i]
            df['language_2'] = languages[j]

            df.to_csv(pairwise_tsv_file, sep='\t', index=False)
            cnt += 1


def convert_formal_crosslingual_stel_or_content():
    # same as above except replace to be stel-or-content setup and make not pairwise for categories and data
    pairwise_output_folder = 'pairwise_multilingual_raw_output/formal_crosslingual_stel_or_content'
    dataset = load_dataset('justinsunqiu/crosslingual_stel', trust_remote_code=True)
    NUM_ROWS_PER_CATEGORY = 100
    paired_sentences = []
    for i in range(dataset.num_rows['train'] // NUM_ROWS_PER_CATEGORY):
        pos_texts = [pos_text for pos_text in dataset['train']['positive'][i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]]
        neg_texts = [neg_text for neg_text in dataset['train']['negative'][i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]]
        paired = [(pos, neg) for pos, neg in zip(pos_texts, neg_texts)]
        paired_sentences.append(tuple(paired))
    categories = []
    languages = []
    for i in range(dataset.num_rows['train'] // NUM_ROWS_PER_CATEGORY):
        languages.append(dataset['train']['language'][i * NUM_ROWS_PER_CATEGORY])
        for j in range(dataset.num_rows['train'] // NUM_ROWS_PER_CATEGORY):
            if i == j:
                continue
            categories.append(
                dataset['train']['style_type'][i * NUM_ROWS_PER_CATEGORY]
                + '_'
                + dataset['train']['language'][i * NUM_ROWS_PER_CATEGORY]
                + '-'
                + dataset['train']['language'][j * NUM_ROWS_PER_CATEGORY]
            )
    cnt = 0
    for i in range(len(paired_sentences)):
        paired_1 = paired_sentences[i]
        for j in range(len(paired_sentences)):
            if i == j:
                continue
            rows = []
            pairwise_tsv_file = os.path.join(pairwise_output_folder, f'{categories[cnt]}_stel_or_content.tsv')
            paired_2 = paired_sentences[j]
            for k in range(len(paired_1)):
                anchor_pos, _ = paired_1[k]  # L1C1S1, L1C1S2
                _, same_content_diff_language_style = paired_2[k]  # L2C1S2
                for l in range(len(paired_2)):
                    if k == l:
                        continue  # Skip when content is equal
                    alt_pos, _ = paired_2[l]  # L2C2S1 L2C2S2
                    alt_neg = same_content_diff_language_style
                    rows.append([anchor_pos, '', alt_pos, alt_neg])
            df = pd.DataFrame(rows, columns=['anchor_1', 'alternative_1', 'anchor_2', 'alternative_2'])
            df['style_type'] = 'formality'
            df['language_1'] = languages[i]
            df['language_2'] = languages[j]
            print(df)

            df.to_csv(pairwise_tsv_file, sep='\t', index=False)
            cnt += 1


# convert_formal_crosslingual()
# convert_formal_crosslingual_stel()
convert_formal_crosslingual_stel_or_content()


def merge_csvs():
    all_files = glob.glob('crosslingual_raw_output/*/*.tsv')
    df = pd.concat((pd.read_csv(file, sep='\t') for file in all_files), ignore_index=True)
    dataset = DatasetDict({'train': Dataset.from_pandas(df)})
    return dataset


# dataset = merge_csvs()
# dataset.push_to_hub('StyleDistance/crosslingual_stel')
# dataset.push_to_hub('crosslingual_stel')
