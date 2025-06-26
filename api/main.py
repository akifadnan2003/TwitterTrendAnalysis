
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import os

from .schemas import CommentInput, SuggestionOutput
from src.data_processing.preprocessor import clean_tweet_text

app = FastAPI(
    title="Advanced Demo Tweet Analyzer API",
    description="Predicts likes based on tweet text AND user reach.",
    version="6.0.0"
)

# --- Add CORS Middleware ---
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor_pipeline = None
# ... (other global variables are the same)

@app.on_event("startup")
def load_models():
    global predictor_pipeline, semantic_model, kw_model, corpus_df, corpus_embeddings

    # --- UPDATED FILE PATHS ---
    model_path = "models/engagement_predictor/final_advanced_demo_model.pkl"
    data_path = "data/processed/processed_advanced_demo_tweets.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("ERROR: Model/data not found. Run training script first.")
        return

    try:
        print("Loading ADVANCED DEMO models and data...")
        predictor_pipeline = joblib.load(model_path)
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        kw_model = KeyBERT()
        corpus_df = pd.read_csv(data_path, low_memory=False).dropna(subset=['text', 'cleaned_text'])
        corpus_embeddings = semantic_model.encode(corpus_df['cleaned_text'].tolist(), convert_to_tensor=True)
        print("Advanced demo models and data loaded successfully.")
    except Exception as e:
        print(f"An error occurred during model loading: {e}")

# ... (@app.get("/") is the same)

@app.post("/analyze", response_model=SuggestionOutput)
async def analyze_comment(comment: CommentInput):
    if not predictor_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    user_text_cleaned = clean_tweet_text(comment.text)
    
    # --- Prepare input as a DataFrame for the multi-input model ---
    input_data = pd.DataFrame({
        'cleaned_text': [user_text_cleaned],
        'Reach': [comment.reach]
    })

    # --- Predict using the advanced model ---
    predicted_likes = predictor_pipeline.predict(input_data)[0]

    # --- The rest of the logic is the same (uses only text) ---
    query_embedding = semantic_model.encode(user_text_cleaned, convert_to_tensor=True)
    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]
    similar_posts_text = [corpus_df.iloc[hit['corpus_id']]['text'] for hit in search_hits]

    suggested_keywords = []
    if similar_posts_text:
        keywords = kw_model.extract_keywords(
            similar_posts_text[0], keyphrase_ngram_range=(1, 2),
            stop_words='english', use_mmr=True, diversity=0.7, top_n=5
        )
        if keywords:
            suggested_keywords = [kw[0] for kw in keywords]
    
    return SuggestionOutput(
        predicted_likes=round(predicted_likes, 0),
        similar_comments=similar_posts_text,
        suggested_keywords=suggested_keywords
    )
