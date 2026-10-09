"""
app.py
------
Streamlit web interface for the Spotify Song Recommendation System.

Loads the fitted StandardScaler, KMeans model, and clustered song dataset
built by train_model.py. The user types a song they like, the app fuzzy-
matches it to the dataset, and recommends other songs from the same
audio-feature cluster.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import difflib
import joblib
import os

st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎧",
    layout="centered",
)

MODEL_DIR = "model"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
DATA_PATH = os.path.join(MODEL_DIR, "music_data.pkl")

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(p) for p in [SCALER_PATH, KMEANS_PATH, DATA_PATH]):
        return None, None, None
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    music = joblib.load(DATA_PATH)
    return scaler, kmeans, music


scaler, kmeans, music = load_artifacts()

st.title("🎧 Spotify Song Recommender")
st.write(
    "Type a song you like and get recommendations for similar songs, "
    "based on K-Means clustering of Spotify audio features "
    "(danceability, energy, acousticness, valence, tempo, etc.)."
)

if scaler is None:
    st.error(
        "Model files not found. Run `python train_model.py` first "
        "(make sure `genres_v2.csv` is in this folder), then reload this app."
    )
    st.stop()

st.divider()

all_song_names = music["song_name"].dropna().unique().tolist()

song_name = st.text_input("Enter a song name:", placeholder="e.g. Black Helicopters")
num_recommendations = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)

if st.button("Get Recommendations", use_container_width=True):
    if not song_name.strip():
        st.warning("Please enter a song name.")
    else:
        close_matches = difflib.get_close_matches(song_name, all_song_names, n=1, cutoff=0.5)

        if not close_matches:
            st.error(f"No close match found for '{song_name}'. Try a different spelling.")
        else:
            matched_song = close_matches[0]
            song_row = music[music["song_name"] == matched_song].iloc[0]
            cluster = song_row["Cluster"]

            recommendations = music[
                (music["Cluster"] == cluster) & (music["song_name"] != matched_song)
            ].sample(n=min(num_recommendations, len(music) - 1), random_state=None)

            st.divider()
            st.subheader(f"Because you liked: {matched_song}")
            if matched_song.lower() != song_name.strip().lower():
                st.caption(f"(Matched '{song_name}' to closest title in dataset: '{matched_song}')")
            st.caption(f"Genre: {song_row['genre']}  |  Cluster: {int(cluster)}")

            st.markdown("**Recommended songs:**")
            display_df = recommendations[["song_name", "genre"]].reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df, use_container_width=True)

            with st.expander("See audio feature profile of the matched song"):
                st.dataframe(song_row[FEATURES].to_frame("value"), use_container_width=True)

st.divider()
st.caption(
    f"Dataset: {len(music):,} unique songs across {music['genre'].nunique()} genres, "
    "grouped into 8 clusters by audio features (not genre labels or listening history)."
)
