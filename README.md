# Create Similar Playlist

A live hybrid music recommender that turns a handful of seed tracks into a ranked, diversity-aware playlist.

**Live demo:** https://create-similar-playlist.streamlit.app/

## How It Works

```text
Seed tracks
    |
Last.fm collaborative similarity
    |
Multi-seed reciprocal-rank fusion
    |
Last.fm music tags
    |
TF-IDF tag vectors + cosine similarity
    |
Hybrid relevance score
    |
MMR-style diversity reranking
    |
Optional Spotify matching + playlist export
```

The recommender combines several lightweight signals rather than relying on a single API recommendation endpoint. It collects similar-track evidence for each seed, rewards candidates supported across multiple seeds, compares music-tag profiles, and then reranks the final list to reduce repetition.

## Features

- Generates recommendations from 1–8 seed tracks without requiring Spotify login.
- Uses Last.fm collaborative similarity for candidate discovery.
- Applies reciprocal-rank fusion across multiple seed tracks.
- Builds TF-IDF-style music-tag vectors and compares them with cosine similarity.
- Combines collaborative similarity, tag similarity, and seed coverage into an interpretable hybrid score.
- Applies an MMR-style redundancy penalty plus a configurable per-artist cap for more varied playlists.
- Shows recommendation-level match, similarity, tag, and coverage signals in the UI.
- Optionally matches recommendations to Spotify and saves the result as a playlist.
- Runs entirely in Streamlit Community Cloud with no local index or persistent OAuth cache.

## Recommendation Formula

The base score is:

- **60% collaborative similarity** from Last.fm similar-track evidence
- **25% tag similarity** from TF-IDF/cosine similarity
- **15% seed coverage** across the user's input tracks

The final ranking subtracts a configurable diversity penalty based on tag-vector similarity to tracks already selected and enforces a maximum number of songs per artist.

## Tech Stack

- Python
- Streamlit
- Last.fm API
- Spotify Web API / Spotipy
- TF-IDF-style text weighting
- Cosine similarity
- Reciprocal-rank fusion
- MMR-style reranking

The ranking math is implemented directly in Python to keep the deployment lightweight and avoid unnecessary ML dependencies.

## Project Structure

```text
streamlit_app.py       # Streamlit UI + optional Spotify OAuth/export
src/
├── lastfm_client.py   # candidate discovery + track tags
├── ranking.py         # TF-IDF, hybrid scoring, and diversity reranking
└── __init__.py
```

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

Only `LASTFM_API_KEY` is required for recommendations. Spotify credentials are optional unless you want track matching and playlist export.

## Deploy

1. Connect this repository to Streamlit Community Cloud.
2. Set the main file to `streamlit_app.py`.
3. Add the required secrets in the Streamlit app settings.
4. If enabling Spotify export, add the exact Streamlit redirect URL to the Spotify Developer Dashboard and authorize the test account for the app.

## Security

Never commit API keys, Spotify client secrets, OAuth tokens, `.env` files, or Streamlit secrets. Keep deployment credentials in Streamlit Secrets or local environment variables only.
