from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import pandas as pd
from utils.preprocess import load_and_preprocess

def train_svd_model():
    df = load_and_preprocess()

    reader = Reader(rating_scale=(1, 10))  # Our interaction scores go from 1 to ~9
    data = Dataset.load_from_df(df[["user_id", "content_id", "interaction_score"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    rmse_score = rmse(predictions)

    return model, predictions, rmse_score

def get_top_n_recommendations(model, df, user_id, n=5, metadata_path="data/content_metadata.csv"):
    all_contents = df["content_id"].unique()
    seen_contents = df[df["user_id"] == user_id]["content_id"].unique()
    unseen_contents = [c for c in all_contents if c not in seen_contents]

    predictions = []
    for content_id in unseen_contents:
        pred = model.predict(user_id, content_id)
        predictions.append((content_id, pred.est))

    # Sort by predicted rating
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata_dict = dict(zip(metadata["content_id"], metadata["description"]))

    # Combine with descriptions
    enriched_recs = [
        {
            "content_id": content_id,
            "description": metadata_dict.get(content_id, "No description available."),
            "score": score
        }
        for content_id, score in top_n
    ]

    return enriched_recs