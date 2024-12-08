from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
dataset_name = "StyleDistance/msynthstel"
similarity_scores = {}
langs = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'ja', 'ko', 'ru', 'zh-hans']

for lang in langs:
    data = load_dataset(dataset_name, lang)
    train_split = data['test']
    n = len(train_split)
    
    positives = train_split['positive']
    negatives = train_split['negative']
    
    positive_embeddings = model.encode(positives, convert_to_tensor=True)
    negative_embeddings = model.encode(negatives, convert_to_tensor=True)
    similarities = []

    for i in range(n):
        similarities.append(util.pytorch_cos_sim(positive_embeddings[i], negative_embeddings[i]).item())

    avg_similarity = sum(similarities) / len(similarities)    
    similarity_scores[lang] = avg_similarity

results_df = pd.DataFrame(list(similarity_scores.items()), columns=["Language", "Avg_Similarity"])
results_df.to_csv("language_similarity_scores.csv", index=False)