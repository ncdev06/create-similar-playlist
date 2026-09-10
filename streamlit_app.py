from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time

import numpy as np
import spotipy
import streamlit as st
from spotipy.oauth2 import SpotifyOAuth

from src.corpus_builder import expand_corpus_from_playlist
from src.features import build_feature_matrix, to_matrix
from src.indexer import save_index
from src.recommender import recommend_for_playlist


st.set_page_config(page_title="Create Similar Playlist", page_icon="🎧", layout="wide")


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read from Streamlit Cloud secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


CLIENT_ID = get_secret("SPOTIFY_CLIENT_ID") or get_secret("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = get_secret("SPOTIFY_CLIENT_SECRET") or get_secret("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = get_secret("SPOTIFY_REDIRECT_URI") or get_secret("SPOTIPY_REDIRECT_URI")
SCOPES = "playlist-read-private playlist-modify-public playlist-modify-private user-read-private"


def _make_state(secret: str) -> str:
    payload = f"{int(time.time())}:{secrets.token_urlsafe(16)}"
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    raw = f"{payload}:{signature}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _valid_state(value: str | None, secret: str, max_age_seconds: int = 900) -> bool:
    if not value:
        return False
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode()
        timestamp, nonce, signature = decoded.split(":", 2)
        payload = f"{timestamp}:{nonce}"
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        return 0 <= time.time() - int(timestamp) <= max_age_seconds
    except Exception:
        return False


def _oauth(state: str | None = None) -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        state=state,
        show_dialog=True,
        open_browser=False,
        cache_handler=None,
    )


def spotify_client() -> spotipy.Spotify:
    if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
        st.error(
            "Spotify credentials are not configured. Add SPOTIFY_CLIENT_ID, "
            "SPOTIFY_CLIENT_SECRET, and SPOTIFY_REDIRECT_URI to your deployment secrets."
        )
        st.stop()

    code = st.query_params.get("code")
    returned_state = st.query_params.get("state")
    error = st.query_params.get("error")

    if error:
        st.error(f"Spotify authorization failed: {error}")
        st.query_params.clear()
        st.stop()

    if code and "spotify_token" not in st.session_state:
        if not _valid_state(returned_state, CLIENT_SECRET):
            st.error("Spotify login could not be verified. Please try connecting again.")
            st.query_params.clear()
            st.stop()

        try:
            token_info = _oauth().get_access_token(code, as_dict=True, check_cache=False)
            st.session_state["spotify_token"] = token_info
            st.query_params.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not finish Spotify login: {exc}")
            st.stop()

    token_info = st.session_state.get("spotify_token")
    if not token_info:
        state = _make_state(CLIENT_SECRET)
        auth_url = _oauth(state=state).get_authorize_url()
        st.title("🎧 Create Similar Playlist")
        st.write(
            "Generate a new playlist from the musical profile of one of your Spotify playlists."
        )
        st.link_button("Connect Spotify", auth_url, type="primary")
        st.caption("Spotify authorization is required to read playlist items and create a playlist in your account.")
        st.stop()

    oauth = _oauth()
    if oauth.is_token_expired(token_info):
        try:
            token_info = oauth.refresh_access_token(token_info["refresh_token"])
            st.session_state["spotify_token"] = token_info
        except Exception:
            st.session_state.pop("spotify_token", None)
            st.rerun()

    return spotipy.Spotify(auth=token_info["access_token"], requests_timeout=20, retries=2)


sp = spotify_client()

st.title("🎧 Create Similar Playlist")
st.write(
    "Build a candidate index from one of your playlists, then generate a new playlist using "
    "feature similarity plus a diversity-aware reranker."
)

header_left, header_right = st.columns([5, 1])
with header_right:
    if st.button("Disconnect"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

with st.expander("1 · Build / refresh candidate index", expanded=True):
    playlist_seed = st.text_input(
        "Spotify playlist URL",
        placeholder="https://open.spotify.com/playlist/...",
        help="With Spotify Development Mode, use a playlist owned by or shared with your authorized Spotify account.",
    )

    if st.button("Build index", type="primary"):
        if not playlist_seed.strip():
            st.error("Paste a Spotify playlist URL first.")
        else:
            try:
                with st.spinner("Reading playlist and expanding the candidate set..."):
                    all_ids, stats = expand_corpus_from_playlist(sp, playlist_seed.strip(), limit_per_seed=80)

                if not all_ids:
                    st.error(
                        "No playlist tracks were returned. Check that the playlist belongs to (or is collaborative with) "
                        "the Spotify account you authorized and that your Spotify developer app has access."
                    )
                else:
                    st.caption(
                        f"Seed tracks: {stats.get('seed_tracks', 0)} · "
                        f"Candidate tracks: {len(all_ids)}"
                    )

                    with st.spinner("Computing features and building the similarity index..."):
                        df = build_feature_matrix(sp, all_ids)
                        X = to_matrix(df)
                        if X.size == 0:
                            raise RuntimeError("Spotify returned no usable track features.")
                        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                        save_index(X.astype("float32"), df["id"].tolist())

                    st.session_state["index_ready"] = True
                    st.session_state["index_seed"] = playlist_seed.strip()
                    st.success(f"Index ready with {len(df)} tracks.")
            except Exception as exc:
                st.error(f"Could not build the index: {exc}")
                st.info(
                    "Spotify has restricted several recommendation/audio-feature endpoints for many Development Mode apps. "
                    "If this works locally with your existing Spotify app but not with a new app, use the same Spotify app credentials."
                )

st.markdown("---")

with st.expander("2 · Generate a similar playlist", expanded=True):
    target_playlist = st.text_input(
        "Target playlist URL",
        key="target_playlist",
        placeholder="https://open.spotify.com/playlist/...",
    )
    k = st.slider("Number of recommendations", min_value=10, max_value=100, value=30, step=5)
    max_per_artist = st.slider("Maximum tracks per artist", min_value=1, max_value=5, value=3)
    allow_explicit = st.checkbox("Allow explicit tracks", value=True)

    if st.button("Generate recommendations"):
        if not target_playlist.strip():
            st.error("Paste a target playlist URL first.")
        elif not st.session_state.get("index_ready"):
            st.warning("Build the candidate index first.")
        else:
            try:
                with st.spinner("Finding similar tracks..."):
                    rec_ids = recommend_for_playlist(
                        sp,
                        target_playlist.strip(),
                        k=k,
                        max_per_artist=max_per_artist,
                        allow_explicit=allow_explicit,
                    )
                st.session_state["rec_ids"] = rec_ids
            except Exception as exc:
                st.error(f"Could not generate recommendations: {exc}")

    rec_ids = st.session_state.get("rec_ids", [])
    if rec_ids:
        st.success(f"Found {len(rec_ids)} recommendations.")
        try:
            tracks = sp.tracks(rec_ids).get("tracks", [])
        except Exception:
            tracks = []

        if tracks:
            cols = st.columns(4)
            for i, track in enumerate(tracks):
                if not track:
                    continue
                with cols[i % 4]:
                    images = (track.get("album") or {}).get("images") or []
                    if images:
                        st.image(images[min(1, len(images) - 1)]["url"], use_container_width=True)
                    artists = track.get("artists") or [{}]
                    st.caption(f"{track.get('name', 'Unknown')} — {artists[0].get('name', 'Unknown')}")

        if st.button("Create playlist in Spotify", type="primary"):
            try:
                me = sp.current_user()
                name = "Similar Playlist · Create Similar Playlist"
                new_playlist = sp.user_playlist_create(me["id"], name=name, public=True)
                sp.playlist_add_items(new_playlist["id"], rec_ids)
                playlist_url = (new_playlist.get("external_urls") or {}).get("spotify")
                st.success("Playlist created in Spotify.")
                if playlist_url:
                    st.link_button("Open playlist in Spotify", playlist_url)
            except Exception as exc:
                st.error(f"Could not create the Spotify playlist: {exc}")

st.caption(
    "Portfolio project · Python · Streamlit · Spotify Web API · FAISS / scikit-learn nearest-neighbor search"
)
