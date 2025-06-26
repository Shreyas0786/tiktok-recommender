import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_content_embeddings(metadata_path="data/content_metadata.csv", model_name="all-MiniLM-L6-v2"):
    df = pd.read_csv(metadata_path)
    descriptions = df["description"].tolist()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(descriptions, convert_to_tensor=True)

    df["embedding"] = embeddings.tolist()
    return df