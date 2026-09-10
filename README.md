# Create Similar Playlist · 2026

A hybrid music recommender rebuilt for the 2026 API landscape.

The original version depended heavily on Spotify Audio Features, Recommendations, and Related Artists. Those APIs are no longer a dependable foundation for a public Development Mode app, so this version removes them completely.

## 2026 Architecture

```text
Seed tracks entered by the user
        |
Last.fm collaborative similarity
        |
Multi-seed reciprocal-rank fusion
        |
Last.fm tags -> TF-IDF content vectors
        |
Hybrid relevance score
        |
MMR-style diversity reranking
        |
Optional Spotify search + playlist export
```

## What Makes v2 Better

- **No deprecated Spotify recommendation endpoints.** Recommendation quality no longer depends on Spotify Audio Features, Recommendations, or Related Artists.
- **Hybrid recommendation.** Combines collaborative similarity, tag-based content similarity, and evidence across multiple seed tracks.
- **Reciprocal-rank fusion.** Candidates supported by multiple seeds receive stronger ranking evidence.
- **TF-IDF tag representation.** Music tags are converted into content vectors for a lightweight, interpretable similarity signal.
- **MMR-style diversification.** Final ranking penalizes repetitive results and enforces an artist cap.
- **Explainable scoring.** The UI exposes collaborative similarity, tag similarity, and seed coverage for every recommendation.
- **2026 Spotify integration.** Spotify is optional and only used for track lookup and playlist export through currently supported endpoints.
- **Cloud-friendly.** No local FAISS index, no persistent token cache, and no dependency on machine-local files.

## Stack

- Python
- Streamlit
- Last.fm API
- scikit-learn
- TF-IDF + cosine similarity
- Reciprocal-rank fusion
- MMR-style reranking
- RapidFuzz
- Spotify Web API / Spotipy 2.26

## Run Locally

```bash
git clone https://github.com/ncdev06/create-similar-playlist.git
cd create-similar-playlist
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Create a `.env` file:

```env
LASTFM_API_KEY=your_lastfm_api_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8501/
```

Only `LASTFM_API_KEY` is required for recommendations. Spotify credentials are optional unless you want Spotify matching and playlist export.

## Deploy on Streamlit Community Cloud

1. Connect this GitHub repository to Streamlit Community Cloud.
2. Set the main file to `streamlit_app.py`.
3. Add secrets in the Streamlit app settings:

```toml
LASTFM_API_KEY = "..."
SPOTIFY_CLIENT_ID = "..."
SPOTIFY_CLIENT_SECRET = "..."
SPOTIFY_REDIRECT_URI = "https://YOUR-APP.streamlit.app/"
```

4. Add the exact same redirect URL to the Spotify Developer Dashboard.
5. If your Spotify app is in Development Mode, add your own Spotify account to the app allowlist before testing export.

## Important Spotify Note

Spotify Development Mode changed substantially in 2026. This project intentionally treats Spotify as an optional output layer rather than the recommendation engine. That keeps the recommender functional even when Spotify limits access to recommendation-specific endpoints.

## Security

Do not commit API keys, Spotify client secrets, OAuth tokens, `.env` files, or Streamlit secrets. The repository ignores local auth/cache artifacts.

## Project Structure

```text
streamlit_app.py       # cloud-ready UI + optional Spotify OAuth/export
src/
├── lastfm_client.py   # candidate discovery + track tags
├── ranking.py         # hybrid scoring + diversity reranking
└── __init__.py
```

## Recommendation Formula

The base score combines:

- **60% collaborative similarity** from Last.fm similar-track evidence
- **25% tag similarity** from TF-IDF/cosine similarity
- **15% seed coverage** across the user's input tracks

The final list then applies an MMR-style redundancy penalty and a configurable maximum number of tracks per artist.

This makes the recommender more explainable and more robust than the original single-vector nearest-neighbor version while remaining lightweight enough to run as a public portfolio demo.
