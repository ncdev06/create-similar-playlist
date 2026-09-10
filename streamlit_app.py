from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import spotipy
import streamlit as st
from spotipy.oauth2 import SpotifyOAuth

from src.lastfm_client import LastFMClient, LastFMError, TrackRef
from src.ranking import build_scores, rank_candidates

st.set_page_config(page_title="Create Similar Playlist", page_icon="🎧", layout="wide")


def secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


LASTFM_API_KEY = secret("LASTFM_API_KEY")
CLIENT_ID = secret("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = secret("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = secret("SPOTIFY_REDIRECT_URI")
SCOPES = "playlist-modify-public playlist-modify-private"


def make_state(key):
    payload = f"{int(time.time())}:{secrets.token_urlsafe(16)}"
    sig = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode().rstrip("=")


def valid_state(value, key):
    try:
        raw = base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode()).decode()
        stamp, nonce, sig = raw.split(":", 2)
        payload = f"{stamp}:{nonce}"
        expected = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected) and 0 <= time.time() - int(stamp) <= 900
    except Exception:
        return False


def oauth(state=None):
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


def spotify_client():
    if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
        return None

    code = st.query_params.get("code")
    returned_state = st.query_params.get("state")
    if code and "spotify_token" not in st.session_state:
        if not valid_state(str(returned_state or ""), CLIENT_SECRET):
            st.error("Spotify login could not be verified. Reconnect and try again.")
            st.query_params.clear()
            return None
        try:
            token = oauth().get_access_token(str(code), as_dict=True, check_cache=False)
            st.session_state["spotify_token"] = token
            st.query_params.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Spotify login failed: {exc}")
            return None

    token = st.session_state.get("spotify_token")
    if not token:
        return None
    auth = oauth()
    if auth.is_token_expired(token):
        token = auth.refresh_access_token(token["refresh_token"])
        st.session_state["spotify_token"] = token
    return spotipy.Spotify(auth=token["access_token"], requests_timeout=15, retries=2)


def parse_seeds(text):
    out, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pair = None
        for sep in (" — ", " – ", " | ", " - "):
            if sep in line:
                pair = line.split(sep, 1)
                break
        if not pair:
            continue
        artist, title = pair[0].strip(), pair[1].strip()
        track = TrackRef(artist, title)
        if artist and title and track.key not in seen:
            seen.add(track.key)
            out.append(track)
    return out[:8]


def recommend(seeds, count, max_artist, diversity):
    client = LastFMClient(LASTFM_API_KEY)
    with ThreadPoolExecutor(max_workers=min(6, len(seeds))) as pool:
        rows = list(pool.map(lambda s: client.similar_tracks(s, limit=45), seeds))
    similar = {seed.key: row for seed, row in zip(seeds, rows)}
    tracks, collab, coverage = build_scores(seeds, similar)
    if not tracks:
        raise RuntimeError("No candidates found. Try different seed tracks.")

    pool_tracks = sorted(
        tracks.values(),
        key=lambda t: (collab.get(t.key, 0), coverage.get(t.key, 0)),
        reverse=True,
    )[: max(40, count * 2)]
    keep = {t.key for t in pool_tracks}
    tracks = {k: v for k, v in tracks.items() if k in keep}
    collab = {k: v for k, v in collab.items() if k in keep}
    coverage = {k: v for k, v in coverage.items() if k in keep}

    with ThreadPoolExecutor(max_workers=6) as pool:
        tag_rows = list(pool.map(lambda t: client.top_tags(t, limit=12), seeds + pool_tracks))
    tags = {t.key: row for t, row in zip(seeds + pool_tracks, tag_rows)}
    return rank_candidates(seeds, tracks, collab, coverage, tags, count, max_artist, diversity)


def _text_similarity(left, right):
    left = " ".join(str(left).casefold().split())
    right = " ".join(str(right).casefold().split())
    if not left or not right:
        return 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    left_words, right_words = set(left.split()), set(right.split())
    union = left_words | right_words
    token_overlap = len(left_words & right_words) / len(union) if union else 0.0
    return 100.0 * (0.75 * sequence + 0.25 * token_overlap)


def spotify_match(sp, track):
    q = f'track:"{track.title}" artist:"{track.artist}"'
    items = sp.search(q=q, type="track", limit=5).get("tracks", {}).get("items", [])
    best, best_score = None, 0.0
    for item in items:
        artists = item.get("artists") or [{}]
        artist = artists[0].get("name", "")
        title = item.get("name", "")
        score = 0.68 * _text_similarity(track.title, title) + 0.32 * _text_similarity(track.artist, artist)
        if score > best_score:
            best_score, best = score, item
    return best if best_score >= 72 else None


st.title("🎧 Create Similar Playlist")
st.write("Start with a few songs you love and discover a fresh playlist built around their shared sound and style.")
st.caption("Collaborative similarity · music-tag matching · multi-seed ranking · diversity reranking · optional Spotify export")

with st.sidebar:
    st.subheader("How it works")
    st.write("Add 1–8 seed tracks. The recommender compares listening patterns and music tags across your picks, then ranks songs that best match the overall vibe.")
    st.write("Adjust the playlist size and diversity controls to make the results tighter, broader, or more varied.")
    st.write("Connect Spotify only if you want to save the final recommendations directly to your account.")

if not LASTFM_API_KEY:
    st.error("Recommendations are temporarily unavailable because the music data service is not configured.")
    st.stop()

left, right = st.columns([3, 2])
with left:
    seed_text = st.text_area(
        "Seed tracks — one per line as Artist — Track",
        height=180,
        placeholder="Tame Impala — The Less I Know the Better\nDaft Punk — Instant Crush\nMGMT — Electric Feel",
    )
with right:
    count = st.slider("Playlist size", 10, 30, 20, 5)
    max_artist = st.slider("Max tracks per artist", 1, 4, 2)
    diversity = st.slider("Diversity strength", 0.00, 0.35, 0.18, 0.01)

if st.button("Generate recommendations", type="primary"):
    seeds = parse_seeds(seed_text)
    if not seeds:
        st.error("Add at least one track using the format `Artist — Track`.")
    else:
        try:
            with st.spinner("Finding tracks that fit your playlist..."):
                st.session_state["recs"] = recommend(seeds, count, max_artist, diversity)
                st.session_state.pop("spotify_matches", None)
        except (LastFMError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

recs = st.session_state.get("recs", [])
if recs:
    st.markdown("---")
    st.subheader("Your recommendations")
    matches = st.session_state.get("spotify_matches", {})
    for i, item in enumerate(recs, 1):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{i}. {item.track.title}**  \n{item.track.artist}")
                if item.tags:
                    st.caption(" · ".join(item.tags[:6]))
                if item.track.url:
                    st.link_button("View on Last.fm", item.track.url)
                match = matches.get(item.track.key)
                if match:
                    url = (match.get("external_urls") or {}).get("spotify")
                    if url:
                        st.link_button("Open in Spotify", url)
            with c2:
                st.metric("Match", f"{round(item.score * 100)}%")
                st.caption(f"similarity {item.collaborative_score:.2f}\ntags {item.tag_score:.2f}\ncoverage {item.seed_coverage:.2f}")

    st.markdown("### Save to Spotify")
    sp = spotify_client()
    if sp is None and CLIENT_ID and CLIENT_SECRET and REDIRECT_URI:
        login = oauth(make_state(CLIENT_SECRET)).get_authorize_url()
        st.link_button("Connect Spotify", login)
        st.caption("Connect your Spotify account to save these recommendations as a playlist.")
    elif sp is not None:
        if st.button("Find these tracks on Spotify"):
            mapped = {}
            progress = st.progress(0)
            for idx, item in enumerate(recs, 1):
                try:
                    match = spotify_match(sp, item.track)
                except Exception:
                    match = None
                if match:
                    mapped[item.track.key] = match
                progress.progress(idx / len(recs))
            st.session_state["spotify_matches"] = mapped
            st.success(f"Found {len(mapped)} of {len(recs)} recommendations on Spotify.")
            st.rerun()

        matches = st.session_state.get("spotify_matches", {})
        if matches:
            name = st.text_input("Playlist name", "Create Similar Playlist")
            public = st.checkbox("Make playlist public", True)
            if st.button("Create playlist in Spotify", type="primary"):
                uris = [matches[x.track.key]["uri"] for x in recs if x.track.key in matches]
                try:
                    playlist = sp.current_user_playlist_create(name=name, public=public, description="Generated from your seed tracks with hybrid similarity and diversity-aware ranking.")
                    sp.playlist_add_items(playlist["id"], uris)
                    st.success(f"Your playlist is ready with {len(uris)} tracks.")
                    url = (playlist.get("external_urls") or {}).get("spotify")
                    if url:
                        st.link_button("Open playlist in Spotify", url)
                except Exception as exc:
                    st.error(f"Spotify export failed: {exc}")

st.markdown("---")
st.caption("Python · Streamlit · Last.fm API · hybrid ranking · cosine similarity · reciprocal-rank fusion · MMR · Spotify Web API")
