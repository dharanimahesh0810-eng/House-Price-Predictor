"""
train_model.py
----------------
Reproduces the K-Means song clustering pipeline from the notebook:
    select audio features -> StandardScaler -> KMeans (k=8)
    -> recommend songs from the same cluster as a query song

Run:
    python train_model.py

Outputs (in model/):
    scaler.pkl        -> fitted StandardScaler
    kmeans_model.pkl   -> fitted KMeans (n_clusters=8)
    music_data.pkl      -> song_name/genre/features/Cluster dataframe used for lookups
"""

import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_PATH = "genres_v2.csv"
MODEL_DIR = "model"
N_CLUSTERS = 8

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Raw dataset shape: {df.shape}")

    music = df[["song_name", "genre"] + FEATURES].dropna()
    # de-duplicate identical song names, keeping the first occurrence
    music = music.drop_duplicates(subset="song_name").reset_index(drop=True)
    print(f"Cleaned dataset shape: {music.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(music[FEATURES])

    print(f"Fitting KMeans with k={N_CLUSTERS} ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    music["Cluster"] = kmeans.fit_predict(X_scaled)

    print("\nSongs per cluster:")
    print(music["Cluster"].value_counts().sort_index())

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(music, os.path.join(MODEL_DIR, "music_data.pkl"))

    print("\nSaved:")
    print(" - model/scaler.pkl")
    print(" - model/kmeans_model.pkl")
    print(" - model/music_data.pkl")


if __name__ == "__main__":
    main()
