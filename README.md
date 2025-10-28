# Create Similar Playlist  
**Python · NumPy · Pandas · Streamlit · FAISS**

## Overview  
Create Similar Playlist is a machine learning application that generates music playlists based on acoustic feature similarity.  
Using the Spotify Web API and FAISS for vector indexing, the system models song relationships through audio features and produces playlists ranked by similarity and diversity.

---

## Key Features  
- **Vector Similarity Search:** Embedded 900+ songs using features such as valence, tempo, and energy; indexed with FAISS for fast nearest-neighbor lookup.  
- **Real-Time Spotify Integration:** Retrieves and processes track data via the Spotify API for dynamic playlist generation.  
- **Streamlit Interface:** Interactive UI allowing users to input seed tracks and generate similar playlists.  
- **Optimized Performance:** FAISS index returns recommendations in under two seconds.  
- **Re-ranking Logic:** Applies post-processing to balance similarity and variety within playlists.

---

## Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| Backend | Python, NumPy, Pandas |
| Similarity Search | FAISS |
| Frontend | Streamlit |
| Data | Spotify Web API, JSON |

---

## Repository Structure  
create-similar-playlist/
│
├── src/
│ ├── spotify_api.py # Spotify authentication and data retrieval
│ ├── feature_engineering.py # Feature extraction and normalization
│ ├── similarity_index.py # FAISS index creation and query logic
│ └── app.py # Streamlit interface
│
├── .gitignore
├── requirements.txt
└── README.md
---

## How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/ncdev06/create-similar-playlist.git
   cd create-similar-playlist
2. Install dependencies:
    pip install -r requirements.txt
3. Set up Spotify API credentials:
    Create an app at Spotify Developer Dashboard
    Add CLIENT_ID and CLIENT_SECRET to a .env file or environment variables
4. Launch the application:
    streamlit run src/app.py

Acknowledgments
    Data accessed through the Spotify Web API
    Similarity search implemented with FAISS (Facebook AI Similarity Search)
