from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import DataFromPrompt, Embed, CosineSimilarity, concat, HFHubDataSource
from datadreamer.embedders import SentenceTransformersEmbedder
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pandas as pd
import os
import tabulate

NUM_ROWS_PER_CATEGORY = 10

with DataDreamer("./output"):
    stel_dataset = HFHubDataSource(
        "Lexical Features",
        path="StyleDistance/synthstel",
        split="test"
    )


def compute_embeddings(
        dataset_pos, dataset_neg, model: str
):
    with DataDreamer("./output"):
        pos_embedded_data = Embed(
            name = f"{model.replace('/', ' ')} Embeddings for Positive Examples",
            inputs = {
                "texts": dataset_pos
            },
            args = {
                "embedder": SentenceTransformersEmbedder(
                    model_name=model
                ),
                "truncate": True
            },
            outputs = {
                "texts": "sentences",
                "embeddings": "embeddings"
            },
        )
        neg_embedded_data = Embed(
            name = f"{model.replace('/', ' ')} Embeddings for Negative Examples",
            inputs = {
                "texts": dataset_neg
            },
            args = {
                "embedder": SentenceTransformersEmbedder(
                    model_name=model
                ),
                "truncate": True
            },
            outputs = {
                "texts": "sentences",
                "embeddings": "embeddings"
            },
        )
    return pos_embedded_data, neg_embedded_data

def convert_embeddings(pos_embedded_data, neg_embedded_data):
    paired_embeddings = []
    for i in range(len(pos_embedded_data.output) // NUM_ROWS_PER_CATEGORY):
        pos_embeddings = np.array(pos_embedded_data.output["embeddings"][i * NUM_ROWS_PER_CATEGORY : (i+1) * NUM_ROWS_PER_CATEGORY])
        neg_embeddings = np.array(neg_embedded_data.output["embeddings"][i * NUM_ROWS_PER_CATEGORY : (i+1) * NUM_ROWS_PER_CATEGORY])
        paired = [(pos, neg) for pos, neg in zip(pos_embeddings, neg_embeddings)]
        paired_embeddings.append(paired)
    return paired_embeddings

def compute_accuracy_STEL(paired_embeddings: list):
    accuracy = 0
    correct = 0
    rand = 0
    incorrect = 0
    for i in range(len(paired_embeddings)):
        anchor_pos, anchor_neg = paired_embeddings[i]
        norm_anchor_pos, norm_anchor_neg = anchor_pos / np.linalg.norm(anchor_pos), anchor_neg / np.linalg.norm(anchor_neg)
        for j in range(i+1, len(paired_embeddings)):
            alt_pos, alt_neg = paired_embeddings[j]
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
    return accuracy / (len(paired_embeddings) * (len(paired_embeddings) - 1) / 2)

def compute_accuracy_STEL_or_content(paired_embeddings: list):
    accuracy = 0
    correct = 0
    rand = 0
    incorrect = 0
    for i in range(len(paired_embeddings)):
        anchor_pos, anchor_neg = paired_embeddings[i]
        norm_anchor_pos, norm_anchor_neg = anchor_pos / np.linalg.norm(anchor_pos), anchor_neg / np.linalg.norm(anchor_neg)
        for j in range(i+1, len(paired_embeddings)):
            alt_pos, alt_neg = paired_embeddings[j]
            norm_alt_pos, norm_alt_neg = alt_pos / np.linalg.norm(alt_pos), alt_neg / np.linalg.norm(alt_neg)
            norm_alt_neg = norm_anchor_neg
            sim1 = np.dot(norm_anchor_pos, norm_alt_pos)
            sim2 = np.dot(norm_anchor_pos, norm_alt_neg)
            if sim1 == sim2:
                accuracy += 0.5
                rand += 1
            elif sim1 > sim2:
                accuracy += 1
                correct += 1
            else:
                accuracy += 0
                incorrect += 1
    return accuracy / (len(paired_embeddings) * (len(paired_embeddings) - 1) / 2)

def STEL_benchmark(dataset_pos, dataset_neg, model, type='STEL'):
    pos_embedded_data, neg_embedded_data = compute_embeddings(dataset_pos, dataset_neg, model)
    paired_embeddings = convert_embeddings(pos_embedded_data, neg_embedded_data)
    accuracies = []
    for paired in paired_embeddings:
        if type == 'STEL':
            accuracies.append(compute_accuracy_STEL(paired))
        elif type == 'STEL-or-content':
            accuracies.append(compute_accuracy_STEL_or_content(paired))
    avg_accuracy = np.mean(accuracies)
    return accuracies, avg_accuracy
    

def STEL_categories():
    categories = []
    for i in range(len(stel_dataset.output) // NUM_ROWS_PER_CATEGORY):
        categories.append(stel_dataset.output['feature'][i * NUM_ROWS_PER_CATEGORY])
    return categories

def STEL_table(model, type='STEL'):
    accuracies, avg_accuracy = STEL_benchmark(stel_dataset.output['positive'], stel_dataset.output['negative'], model, type)
    accuracies.append(avg_accuracy)
    categories = STEL_categories()
    categories.append('average')
    data = {
        'Metric': categories,
        f'{model} Embeddings': accuracies
    }
    df = pd.DataFrame(data)
    return df

def merge_dfs(dfs):
    for df in dfs:
        df.set_index('Metric', inplace=True)
    merged_df = pd.concat(dfs, axis=1)
    return merged_df


tpe = 'STEL'

models = ['AnnaWegmann/Style-Embedding', 'google-bert/bert-base-cased', 'FacebookAI/roberta-base', 'SynthSTEL/styledistance', 'SynthSTEL/styledistance_synthetic_only', 'StyleDistance/styledistance_synthetic_only_ablation_hard', 'StyleDistance/styledistance_synthetic_only_ablation_easy']
tables = [STEL_table(model, type=tpe) for model in models]
merged_dfs = merge_dfs(tables)
print(merged_dfs.to_markdown())