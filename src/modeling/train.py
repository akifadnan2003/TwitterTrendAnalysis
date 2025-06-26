import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import os

from src.data_processing.preprocessor import clean_tweet_text

# --- Define file paths ---
DATA_PATH = "data/raw/Twitterdatainsheets.csv" # The dataset that contains the original 'Reach'
PROCESSED_DATA_PATH = "data/processed/processed_advanced_demo_tweets.csv"
MODEL_DIR = "models/engagement_predictor"
MODEL_PATH = os.path.join(MODEL_DIR, "final_advanced_demo_model.pkl")

def train_model():
    """
    Loads the first 90,000 rows of the cutted.csv dataset, generates
    synthetic 'Likes' for a dynamic demo, and trains an advanced model
    that uses both text and the original 'Reach' as input features.
    """
    print(f"Starting ADVANCED DEMO model training with the first 90,000 rows of {DATA_PATH}...")

    # --- FIX: Load only the first 90,000 rows ---
    try:
        df = pd.read_csv(DATA_PATH, nrows=90000)
        print(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"Failed to read the CSV file. Error: {e}")
        return

    df.columns = df.columns.str.strip()
    print("Cleaned column names.")

    # --- Define our feature and target columns ---
    TEXT_COLUMN = 'text'
    NUMERICAL_COLUMN = 'Reach'
    TARGET_COLUMN = 'Likes'

    if TEXT_COLUMN not in df.columns or NUMERICAL_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
        raise ValueError("Required columns 'text', 'Reach', or 'Likes' not found.")

    # --- Generate Synthetic Likes for a Better Demonstration ---
    print("Generating synthetic 'Likes' data...")
    np.random.seed(42)
    base_likes = np.random.randint(0, 50, size=len(df))
    reach_bonus = (df[NUMERICAL_COLUMN] * np.random.uniform(0.001, 0.05, size=len(df))).fillna(0)
    df[TARGET_COLUMN] = (base_likes + reach_bonus).astype(int)
    print("Successfully replaced 'Likes' column with generated data.")

    # Clean the text data
    df['cleaned_text'] = df[TEXT_COLUMN].apply(clean_tweet_text)

    # --- Define features (X) and target (y) ---
    X = df[['cleaned_text', NUMERICAL_COLUMN]]
    y = df[TARGET_COLUMN]

    # Handle any missing values in the features
    X[NUMERICAL_COLUMN] = X[NUMERICAL_COLUMN].fillna(0)
    print("Handled missing values in 'Reach' column.")

    # --- Create the advanced preprocessing pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=2000, stop_words='english'), 'cleaned_text'),
            ('numeric', StandardScaler(), [NUMERICAL_COLUMN])
        ])

    # --- Create the full model pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fitting the advanced multi-input pipeline...")
    model_pipeline.fit(X_train, y_train)

    score = model_pipeline.score(X_test, y_test)
    print(f"ADVANCED DEMO Model R^2 score on test set: {score:.4f}")

    # Save the new model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Advanced demo model saved successfully to {MODEL_PATH}")

    # Save the processed data for the API
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data for API saved to {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    train_model()
