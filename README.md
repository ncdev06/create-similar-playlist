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
src/
├── config.py           # configuration and environment variables
├── corpus_builder.py   # expands the candidate track corpus
├── features.py         # feature extraction and matrix construction
├── indexer.py          # FAISS HNSW / sklearn nearest-neighbor index
├── recommender.py      # playlist vector, retrieval, and reranking logic
├── spotify_client.py   # Spotify OAuth client
└── ui_app.py           # Streamlit interface
```

## Run Locally

### 1. Clone and install
```bash
git clone https://github.com/ncdev06/create-similar-playlist.git
cd create-similar-playlist
pip install -r requirements.txt
```

### 2. Configure Spotify credentials
Create a `.env` file with the Spotify credentials expected by `src/config.py`.

### 3. Start the app
```bash
streamlit run src/ui_app.py
```

## Notes
FAISS is optional at runtime. If it is unavailable, the project falls back to scikit-learn nearest-neighbor search using cosine distance.
