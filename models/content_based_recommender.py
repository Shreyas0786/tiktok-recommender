import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.embeddings import generate_content_embeddings
from utils.preprocess import load_and_preprocess

def recommend_similar_content(user_id, top_k=5):
    interactions = load_and_preprocess()
    content_df = generate_content_embeddings()

    # Get items the user liked/shared (score >= 2)
    liked_content_ids = interactions[
        (interactions["user_id"] == user_id) & 
        (interactions["interaction_score"] >= 2)
    ]["content_id"].tolist()

    if not liked_content_ids:
        return []

    liked_embeddings = content_df[content_df["content_id"].isin(liked_content_ids)]["embedding"].tolist()
    liked_embeddings = np.array(liked_embeddings).mean(axis=0).reshape(1, -1)

    all_embeddings = np.vstack(content_df["embedding"].tolist())
    sims = cosine_similarity(liked_embeddings, all_embeddings).flatten()

    content_df["similarity"] = sims
    top_n = content_df[~content_df["content_id"].isin(liked_content_ids)].sort_values(by="similarity", ascending=False).head(top_k)

    return top_n[["content_id", "description", "similarity"]]
