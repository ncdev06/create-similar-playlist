import spotipy
from spotipy.oauth2 import SpotifyOAuth
from src.config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPES


_auth_cache = ".cache" # local token cache




def get_spotify():
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=_auth_cache,
        open_browser=True,
    )
)
    return sp