# 🎯 TikTok-Style Personalized Content Recommendation Engine

A portfolio-ready machine learning project that mimics TikTok’s personalized content feed. It combines collaborative filtering, content-based filtering with sentence embeddings, and an interactive Streamlit web app to deliver personalized recommendations.

---

## 🚀 Features

- ✅ Simulated user-content interaction data
- ✅ Collaborative Filtering using SVD (Surprise)
- ✅ Content-Based Filtering using sentence embeddings (MiniLM)
- ✅ Streamlit App with user-friendly UI
- ✅ Score legend + explanation in sidebar
- ✅ Recommendation descriptions for real context

---

## 🧠 How it Works

### 1. Collaborative Filtering (CF)

- Built using **SVD matrix factorization** from the `Surprise` library.
- Learns user-content interaction scores from simulated data:
  
  | Interaction | Score |
  |-------------|--------|
  | View        | 1.0    |
  | Like        | 2.0    |
  | Share       | 3.0    |

- The model predicts scores from 1 to 3:
  - `1.0`: User might view the content
  - `2.0`: User might like the content
  - `3.0`: User might share the content

> **Example:**  
> "User 10 is predicted to like Content 40 with score `2.14`."

---

### 2. Content-Based Filtering (CBF)

- Uses `sentence-transformers` (MiniLM) to convert content descriptions into dense embeddings.
- Calculates cosine similarity between liked content and all other content.
- Ranks content by similarity for personalized recommendations.

> **Similarity score** ranges from `0` (unrelated) to `1` (very similar).

---

## 🖥️ Streamlit App Features

- Select any **User ID (1–100)**
- Choose recommendation type:  
  - 🎲 Collaborative Filtering  
  - 🧠 Content-Based Filtering (Embeddings)
- See **top 5 recommendations** with:
  - Content ID
  - Human-readable description
  - Score or similarity
- 📊 Sidebar with clear explanation of score meanings

---

## 📈 Sample Output

### Collaborative Filtering

Content 15
"Behind-the-scenes of music video shoot"
📈 Predicted Score: 1.93

### Content-Based Filtering

Content 40
"Street magic tricks that shock people"
🧠 Similarity Score: 0.87

---

## 🛠️ Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt

streamlit run app/streamlit_app.py

