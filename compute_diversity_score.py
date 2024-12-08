from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
dataset_name = "StyleDistance/msynthstel"

def compute_diversity_score(sentences):
    # Encode sentences to get their embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Initialize variables
    n = len(sentences)
    total_diversity = 0
    count = 0

    # Compute pairwise cosine similarities
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                diversity = 1 - similarity
                total_diversity += diversity
                count += 1

    # Calculate diversity score using the provided formula
    diversity_score = total_diversity / count if count > 0 else 0

    return diversity_score

langs = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'ja', 'ko', 'ru', 'zh-hans']

for lang in langs:
    print(lang)
    data = load_dataset(dataset_name, lang)
    # print(data['test']['positive'])
    score = compute_diversity_score(data['test']['positive'])
    print(score)