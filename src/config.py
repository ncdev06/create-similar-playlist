import os
from dotenv import load_dotenv
load_dotenv()

# Accept either prefix to match your .env history
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET") or os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = (
    os.getenv("SPOTIFY_REDIRECT_URI")
    or os.getenv("SPOTIPY_REDIRECT_URI")
    or "http://127.0.0.1:8080/callback"
)
MARKET = os.getenv("SPOTIFY_MARKET", "US")
FILTER_EXPLICIT = os.getenv("FILTER_EXPLICIT", "0") == "1"

SCOPES = (
    "playlist-read-private playlist-modify-public "
    "playlist-modify-private user-read-private"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
