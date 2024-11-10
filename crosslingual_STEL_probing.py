import math
import pickle

import numpy as np
import pandas as pd
from datadreamer import DataDreamer
from datadreamer.embedders import Embedder, SentenceTransformersEmbedder
from datadreamer.llms import OpenAI
from datadreamer.steps import (CosineSimilarity, DataFromPrompt, Embed,
                               HFHubDataSource, concat)
from transformers import AutoModel, AutoTokenizer

from luar_utils import load_luar_as_sentence_transformer
from stylegenome_lisa_sfam.lisa_inference_utils import (load_lisa,
                                                        predict_lisa_embedder)

NUM_ROWS_PER_CATEGORY = 100

with DataDreamer('./output'):
    stel_dataset = HFHubDataSource('Lexical Features', path='justinsunqiu/crosslingual_stel', split='train')


def save_embeddings(paired_embeddings, filename):
    with open('output_embeddings/' + filename, 'wb') as file:
        pickle.dump(paired_embeddings, file)


def load_embeddings(filename):
    with open('output_embeddings/' + filename, 'rb') as file:
        paired_embeddings = pickle.load(file)
    return paired_embeddings


def compute_embeddings(dataset_pos, dataset_neg, model_name, model):
    with DataDreamer('./output'):
        pos_embedded_data = Embed(
            name=f'{model_name.replace("/", " ")} Crosslingual Embeddings for Positive Examples',
            inputs={'texts': dataset_pos},
            args={
                'embedder': model,
                'truncate': True,
            },
            outputs={'texts': 'sentences', 'embeddings': 'embeddings'},
        )
        neg_embedded_data = Embed(
            name=f'{model_name.replace("/", " ")} Crosslingual Embeddings for Negative Examples',
            inputs={'texts': dataset_neg},
            args={
                'embedder': model,
                'truncate': True,
            },
            outputs={'texts': 'sentences', 'embeddings': 'embeddings'},
        )
    return pos_embedded_data, neg_embedded_data


def convert_embeddings(pos_embedded_data, neg_embedded_data):
    paired_embeddings = []
    for i in range(len(pos_embedded_data.output) // NUM_ROWS_PER_CATEGORY):
        pos_embeddings = np.array(pos_embedded_data.output['embeddings'][i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY])
        neg_embeddings = np.array(neg_embedded_data.output['embeddings'][i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY])
        paired = [(pos, neg) for pos, neg in zip(pos_embeddings, neg_embeddings)]
        paired_embeddings.append(paired)
    return paired_embeddings


def get_embeddings_LISA(dataset_pos, dataset_neg, use_cached=True):
    if use_cached:
        return load_embeddings('LISA_crosslingual')
    paired_embeddings = []
    model, tokenizer, embedder = load_lisa('stylegenome_lisa_sfam/lisa_checkpoint')
    for i in range(len(dataset_pos) // NUM_ROWS_PER_CATEGORY):
        pos_embeddings = [
            predict_lisa_embedder(model, tokenizer, embedder, pos_text)
            for pos_text in dataset_pos[i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]
        ]
        neg_embeddings = [
            predict_lisa_embedder(model, tokenizer, embedder, neg_text)
            for neg_text in dataset_neg[i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]
        ]
        paired = [(pos, neg) for pos, neg in zip(pos_embeddings, neg_embeddings)]
        paired_embeddings.append(tuple(paired))
    save_embeddings(paired_embeddings, 'LISA_crosslingual')
    return paired_embeddings


def get_embeddings_LUAR(dataset_pos, dataset_neg, use_cached=True):
    if use_cached:
        return load_embeddings('LUAR_crosslingual')
    paired_embeddings = []
    model = load_luar_as_sentence_transformer('rrivera1849/LUAR-MUD')
    for i in range(len(dataset_pos) // NUM_ROWS_PER_CATEGORY):
        pos_embeddings = [model.encode(pos_text) for pos_text in dataset_pos[i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]]
        neg_embeddings = [model.encode(neg_text) for neg_text in dataset_neg[i * NUM_ROWS_PER_CATEGORY : (i + 1) * NUM_ROWS_PER_CATEGORY]]
        paired = [(pos, neg) for pos, neg in zip(pos_embeddings, neg_embeddings)]
        paired_embeddings.append(tuple(paired))
    save_embeddings(paired_embeddings, 'LUAR_crosslingual')
    return paired_embeddings


def compute_accuracy_STEL(paired_embeddings_1: list, paired_embeddings_2: list):
    accuracy = 0
    correct = 0
    rand = 0
    incorrect = 0
    for i in range(len(paired_embeddings_1)):
        anchor_pos, anchor_neg = paired_embeddings_1[i]  # L1C1S1, L1C1S2
        norm_anchor_pos, norm_anchor_neg = anchor_pos / np.linalg.norm(anchor_pos), anchor_neg / np.linalg.norm(anchor_neg)
        for j in range(len(paired_embeddings_2)):
            if i == j:
                continue  # Skip when content is equal
            alt_pos, alt_neg = paired_embeddings_2[j]  # L2C2S1 L2C2S2
            norm_alt_pos, norm_alt_neg = alt_pos / np.linalg.norm(alt_pos), alt_neg / np.linalg.norm(alt_neg)
            sim1 = np.dot(norm_anchor_pos, norm_alt_pos)
            sim2 = np.dot(norm_anchor_neg, norm_alt_neg)
            sim3 = np.dot(norm_anchor_pos, norm_alt_neg)
            sim4 = np.dot(norm_anchor_neg, norm_alt_pos)
            if math.pow(1 - sim1, 2) + math.pow(1 - sim2, 2) == math.pow(1 - sim3, 2) + math.pow(1 - sim4, 2):
                accuracy += 0.5
                rand += 1
            elif math.pow(1 - sim1, 2) + math.pow(1 - sim2, 2) < math.pow(1 - sim3, 2) + math.pow(1 - sim4, 2):
                accuracy += 1
                correct += 1
            else:
                accuracy += 0
                incorrect += 1
    return accuracy / (len(paired_embeddings_1) * (len(paired_embeddings_2) - 1))  # don't divide by 2 because we are doing every combination


def compute_accuracy_STEL_or_content(paired_embeddings_1: list, paired_embeddings_2: list):
    accuracy = 0
    correct = 0
    rand = 0
    incorrect = 0
    for i in range(len(paired_embeddings_1)):
        anchor_pos, _ = paired_embeddings_1[i]  # L1C1S1, L1C1S2
        norm_anchor_pos, _ = anchor_pos / np.linalg.norm(anchor_pos), _
        _, same_content_diff_language_style = paired_embeddings_2[i]  # L2C1S2
        norm_same_content_diff_language_style = same_content_diff_language_style / np.linalg.norm(same_content_diff_language_style)
        for j in range(len(paired_embeddings_2)):
            if i == j:
                continue  # Skip when content is equal
            alt_pos, _ = paired_embeddings_2[j]  # L2C2S1, L2C2S2
            norm_alt_pos, _ = alt_pos / np.linalg.norm(alt_pos), _
            norm_alt_neg = norm_same_content_diff_language_style
            sim1 = np.dot(norm_anchor_pos, norm_alt_pos)  # L1C1S1, L2C2S1
            sim2 = np.dot(norm_anchor_pos, norm_alt_neg)  # L1C1S1, L2C1S2
            if sim1 == sim2:
                accuracy += 0.5
                rand += 1
            elif sim1 > sim2:
                accuracy += 1
                correct += 1
            else:
                accuracy += 0
                incorrect += 1
    return accuracy / (len(paired_embeddings_1) * (len(paired_embeddings_2) - 1))  # don't divide by 2 because we are doing every combination


def STEL_benchmark(dataset_pos, dataset_neg, model_name, model, tpe='STEL'):
    if model_name == 'lisa':
        paired_embeddings = get_embeddings_LISA(dataset_pos, dataset_neg)
    elif model_name == 'luar':
        paired_embeddings = get_embeddings_LUAR(dataset_pos, dataset_neg)
    else:
        pos_embedded_data, neg_embedded_data = compute_embeddings(dataset_pos, dataset_neg, model_name, model)
        paired_embeddings = convert_embeddings(pos_embedded_data, neg_embedded_data)
    accuracies = []
    for i in range(len(paired_embeddings)):
        paired_1 = paired_embeddings[i]
        if tpe == 'STEL':
            for j in range(i + 1, len(paired_embeddings)):
                if i == j:
                    continue
                paired_2 = paired_embeddings[j]
                accuracies.append(compute_accuracy_STEL(paired_1, paired_2))
        elif tpe == 'STEL-or-content':
            for j in range(len(paired_embeddings)):
                if i == j:
                    continue
                paired_2 = paired_embeddings[j]
                accuracies.append(compute_accuracy_STEL_or_content(paired_1, paired_2))
    avg_accuracy = np.mean(accuracies)
    return accuracies, avg_accuracy


def STEL_categories(tpe='STEL'):
    categories = []
    for i in range(len(stel_dataset.output) // NUM_ROWS_PER_CATEGORY):
        if tpe == 'STEL':
            for j in range(i + 1, len(stel_dataset.output) // NUM_ROWS_PER_CATEGORY):
                categories.append(
                    stel_dataset.output['style_type'][i * NUM_ROWS_PER_CATEGORY]
                    + ' '
                    + stel_dataset.output['language'][i * NUM_ROWS_PER_CATEGORY]
                    + '-'
                    + stel_dataset.output['language'][j * NUM_ROWS_PER_CATEGORY]
                )
        elif tpe == 'STEL-or-content':
            for j in range(len(stel_dataset.output) // NUM_ROWS_PER_CATEGORY):
                if i == j:
                    continue
                categories.append(
                    stel_dataset.output['style_type'][i * NUM_ROWS_PER_CATEGORY]
                    + ' '
                    + stel_dataset.output['language'][i * NUM_ROWS_PER_CATEGORY]
                    + '-'
                    + stel_dataset.output['language'][j * NUM_ROWS_PER_CATEGORY]
                )
    return categories


def STEL_table(model_name, model=None, tpe='STEL'):
    if not model:
        model = SentenceTransformersEmbedder(model_name=model)

    accuracies, avg_accuracy = STEL_benchmark(
        stel_dataset.output['positive'],
        stel_dataset.output['negative'],
        model_name,
        model,
        tpe,
    )
    accuracies.append(avg_accuracy)
    categories = STEL_categories(tpe)
    categories.append('average')
    data = {'Metric': categories, f'{model_name} Embeddings': accuracies}
    print(data)
    df = pd.DataFrame(data)
    return df


def merge_dfs(dfs):
    for df in dfs:
        df.set_index('Metric', inplace=True)
    merged_df = pd.concat(dfs, axis=1)
    return merged_df


tpe = 'STEL'
models = [
    ('Wegmann', SentenceTransformersEmbedder(model_name='AnnaWegmann/Style-Embedding')),
    (
        'StyleDistance',
        SentenceTransformersEmbedder(model_name='SynthSTEL/styledistance'),
    ),
    (
        'StyleDistance Synthetic Only',
        SentenceTransformersEmbedder(model_name='SynthSTEL/styledistance_synthetic_only'),
    ),
    ('bert-base-cased', SentenceTransformersEmbedder('google-bert/bert-base-cased')),
    ('roberta-base', SentenceTransformersEmbedder('FacebookAI/roberta-base')),
    ('xlm-roberta-base', SentenceTransformersEmbedder('FacebookAI/xlm-roberta-base')),
    ('lisa', None),
    ('luar', None),
]
tables = [STEL_table(name, model, tpe=tpe) for name, model in models]
merged_dfs = merge_dfs(tables)
print(merged_dfs.to_markdown())
output_file = f'output_xlsx/{tpe}_crosslingual.xlsx'
merged_dfs.to_excel(output_file, index=False)
