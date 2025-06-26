import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Simulate data
random.seed(42)
num_users = 100
num_contents = 50

data = []
for _ in range(2000):  # 2000 interactions
    user_id = random.randint(1, num_users)
    content_id = random.randint(1, num_contents)
    interaction_type = random.choices(["view", "like", "share"], weights=[0.6, 0.3, 0.1])[0]
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    data.append([user_id, content_id, interaction_type, timestamp])

df = pd.DataFrame(data, columns=["user_id", "content_id", "interaction_type", "timestamp"])
os.makedirs("data", exist_ok=True)
df.to_csv("data/interactions.csv", index=False)
print("Simulated dataset saved at data/interactions.csv")

from models.collaborative_filtering import train_svd_model

model, predictions, rmse_score = train_svd_model()
print(f"Model trained! RMSE: {rmse_score:.2f}")

from models.collaborative_filtering import get_top_n_recommendations
from utils.preprocess import load_and_preprocess

df = load_and_preprocess()
user_id = 10  # try a user ID from your dataset
top_recs = get_top_n_recommendations(model, df, user_id)

print(f"\nTop recommendations for user {user_id}:")
for content_id, score in top_recs:
    print(f"Content {content_id} — predicted score: {score:.2f}")
    
    
from models.content_based_recommender import recommend_similar_content

print("\n📚 Content-Based Recommendations using Embeddings:")
top_sim = recommend_similar_content(user_id=10)
print(top_sim)