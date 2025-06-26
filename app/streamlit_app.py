import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from models.collaborative_filtering import train_svd_model, get_top_n_recommendations
from models.content_based_recommender import recommend_similar_content
from utils.preprocess import load_and_preprocess

st.set_page_config(page_title="TikTok-Style Recommender", layout="centered")

st.sidebar.title("📊 Prediction Score Legend")
st.sidebar.markdown("""
**Prediction Score Meaning (Collaborative Filtering):**

- `1.0` — User likely **viewed** the content  
- `2.0` — User likely **liked** the content  
- `3.0` — User likely **shared** the content  

The predicted score ranges between 1 and 3 and reflects how strongly the model expects the user to engage with the content.
""")

st.sidebar.title("🧠 Similarity Score (Content-Based)")

st.sidebar.markdown("""
- Ranges from `0` to `1`  
- Measures semantic similarity between content descriptions  
- Closer to `1` means very similar content to what the user liked before
""")

st.title("🎯 Personalized Content Recommendation Engine")
st.markdown("Choose a user and a model type to view personalized recommendations.")

# --- User Selection ---
user_id = st.selectbox("Select a user ID", options=list(range(1, 101)), index=9)

# --- Model Type ---
model_type = st.radio("Select recommendation type", ["Collaborative Filtering", "Content-Based (Embeddings)"])

# --- Load Data ---
df = load_and_preprocess()

# --- Recommendation Logic ---
if model_type == "Collaborative Filtering":
    model, _, _ = train_svd_model()
    recs = get_top_n_recommendations(model, df, user_id)

    st.subheader("🤖 Recommendations using Collaborative Filtering")

    for rec in recs:
        st.markdown(f"""
        **Content {rec['content_id']}**  
        _{rec['description']}_  
        📈 Predicted Score: `{rec['score']:.2f}`
        """)

else:
    st.subheader("🧠 Recommendations using Content Embeddings")
    top_n = recommend_similar_content(user_id)

    if top_n.empty:
        st.warning("User has no liked content yet. Try another user.")
    else:
        for _, row in top_n.iterrows():
            st.markdown(f"**Content {int(row['content_id'])}** — {row['description']}  \nSimilarity: `{row['similarity']:.2f}`")