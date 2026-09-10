# Create Similar Playlist

A Spotify recommendation app that builds playlists from **audio-feature similarity** using the Spotify Web API, NumPy/Pandas, and approximate nearest-neighbor search.

## Why this project is interesting
Instead of relying on genre labels alone, the system represents tracks with normalized audio features, builds a searchable vector index, and recommends songs that are close to a target playlist in feature space. A multi-pass reranker then trades off similarity with playlist diversity.

## Key Features
- Builds a candidate corpus from public Spotify playlists and related tracks.
- Computes and normalizes track feature vectors for similarity search.
- Uses **FAISS HNSW** when available, with a **scikit-learn cosine nearest-neighbor fallback**.
- Represents an input playlist using the mean of its track vectors.
- Retrieves a wide candidate pool, removes seed tracks, and reranks results with artist-diversity and explicit-content constraints.
- Lets users create the generated playlist directly in their Spotify account through a **Streamlit** interface.
- Includes a cloud-ready OAuth flow in `streamlit_app.py` so the deployed app can be opened from a phone, tablet, or computer.

## Tech Stack
- **Python**
- **NumPy / Pandas**
- **FAISS**
- **scikit-learn**
- **Spotipy / Spotify Web API**
- **Streamlit**

## Architecture
```text
Spotify playlist
      |
Candidate-corpus expansion
      |
Feature extraction + normalization
      |
FAISS HNSW / cosine-NN index
      |
Playlist-vector query
      |
Candidate retrieval
      |
Diversity-aware reranking
      |
Streamlit results + Spotify playlist creation
```

## Repository Structure
```text
streamlit_app.py       # cloud-ready entrypoint + Spotify OAuth
src/
├── config.py           # local configuration and environment variables
├── corpus_builder.py   # expands the candidate track corpus
├── features.py         # feature extraction and matrix construction
├── indexer.py          # FAISS HNSW / sklearn nearest-neighbor index
├── recommender.py      # playlist vector, retrieval, and reranking logic
├── spotify_client.py   # local Spotify OAuth client
└── ui_app.py           # original local Streamlit interface
```

## Deploy on Streamlit Community Cloud
This project is already prepared for Streamlit Community Cloud. This is the recommended host for the current Streamlit architecture.

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create a new app from `ncdev06/create-similar-playlist`.
3. Choose branch `main` and entrypoint `streamlit_app.py`.
4. Pick a permanent app URL before configuring Spotify OAuth.
5. In the Spotify Developer Dashboard, add that exact deployed URL as a Redirect URI.
6. In the Streamlit app's **Settings → Secrets**, add:

```toml
SPOTIFY_CLIENT_ID = "your_client_id"
SPOTIFY_CLIENT_SECRET = "your_client_secret"
SPOTIFY_REDIRECT_URI = "https://YOUR-APP.streamlit.app/"
```

7. Save the secrets, reboot the app, and click **Connect Spotify**.

The deployed URL is accessible from any modern browser, including mobile devices. Spotify Development Mode may still restrict which Spotify accounts are allowed to authenticate.

## Spotify Development Mode limitations
Spotify has restricted several Web API capabilities used by this project, including Audio Features, Recommendations, and Related Artists for many Development Mode apps. If your Spotify developer app does not have access to those endpoints, the similarity-index build step will not work even though deployment itself succeeds.

Development Mode is also intended for personal/testing use and limits the number of authorized users. For a portfolio demo, the safest setup is to keep your own Spotify account and a few testers allowlisted in the Spotify Developer Dashboard.

## Run Locally

### 1. Clone and install
```bash
git clone https://github.com/ncdev06/create-similar-playlist.git
cd create-similar-playlist
pip install -r requirements.txt
```

### 2. Configure Spotify credentials
Create a `.env` file:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8501
```

### 3. Start the original local app
```bash
streamlit run src/ui_app.py
```

Or test the cloud-ready OAuth entrypoint locally:
```bash
streamlit run streamlit_app.py
```

## Notes
FAISS is included for deployment. If it cannot load on a platform, the project falls back to scikit-learn nearest-neighbor search using cosine distance.
