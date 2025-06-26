import pandas as pd

def load_and_preprocess(path="data/interactions.csv"):
    df = pd.read_csv(path)

    # Assign numerical weights to interaction types
    weights = {"view": 1, "like": 2, "share": 3}
    df["interaction_score"] = df["interaction_type"].map(weights)

    # Aggregate multiple interactions per user-content pair
    df_agg = df.groupby(["user_id", "content_id"])["interaction_score"].sum().reset_index()

    return df_agg