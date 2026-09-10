from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import requests
import spotipy
import streamlit as st
import streamlit.components.v1 as components
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

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
SCOPES = (
    "playlist-read-private playlist-read-collaborative "
    "playlist-modify-public playlist-modify-private"
)


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
        try:
            token = auth.refresh_access_token(token["refresh_token"])
            st.session_state["spotify_token"] = token
        except Exception:
            st.session_state.pop("spotify_token", None)
            return None

    return spotipy.Spotify(auth=token["access_token"], requests_timeout=15, retries=2)


@st.cache_resource(show_spinner=False)
def spotify_catalog_client(client_id, client_secret):
    if not (client_id and client_secret):
        return None
    manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        cache_handler=None,
    )
    return spotipy.Spotify(auth_manager=manager, requests_timeout=15, retries=2)


def spotify_access_token():
    token = st.session_state.get("spotify_token") or {}
    return token.get("access_token")


def spotify_api(method, path, *, token, params=None, payload=None):
    response = requests.request(
        method,
        f"https://api.spotify.com/v1/{path.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        params=params,
        json=payload,
        timeout=20,
    )
    if response.status_code >= 400:
        detail = ""
        try:
            error = response.json().get("error") or {}
            detail = error.get("message", "") if isinstance(error, dict) else str(error)
        except Exception:
            detail = response.text[:160]
        raise RuntimeError(
            f"Spotify returned {response.status_code}"
            + (f": {detail}" if detail else ".")
        )
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def extract_playlist_id(value):
    value = (value or "").strip()
    if not value:
        raise ValueError("Paste a Spotify playlist URL first.")

    if value.startswith("spotify:playlist:"):
        playlist_id = value.rsplit(":", 1)[-1]
    else:
        match = re.search(r"open\.spotify\.com/playlist/([A-Za-z0-9]+)", value)
        playlist_id = match.group(1) if match else value

    if not re.fullmatch(r"[A-Za-z0-9]{10,40}", playlist_id):
        raise ValueError("That does not look like a valid Spotify playlist URL.")
    return playlist_id


def load_spotify_playlist(url, max_items=200):
    token = spotify_access_token()
    if not token:
        raise RuntimeError("Connect Spotify before importing a playlist.")

    playlist_id = extract_playlist_id(url)
    try:
        metadata = spotify_api("GET", f"playlists/{playlist_id}", token=token)
    except RuntimeError:
        metadata = {}

    tracks = []
    seen = set()
    offset = 0
    page_size = 50

    while offset < max_items:
        try:
            page = spotify_api(
                "GET",
                f"playlists/{playlist_id}/items",
                token=token,
                params={"limit": page_size, "offset": offset},
            )
        except RuntimeError as exc:
            if "403" in str(exc):
                raise RuntimeError(
                    "Spotify currently lets this app read playlist contents only for "
                    "playlists you own or collaborate on. A public playlist you own works, "
                    "but arbitrary public playlists from other accounts do not."
                ) from exc
            raise

        rows = page.get("items") or []
        if not rows:
            break

        for row in rows:
            item = row.get("item") or row.get("track") or {}
            if item.get("type") != "track":
                continue
            artists = item.get("artists") or []
            artist = (artists[0] or {}).get("name", "").strip() if artists else ""
            title = item.get("name", "").strip()
            if not artist or not title:
                continue
            track = TrackRef(artist, title)
            if track.key in seen:
                continue
            seen.add(track.key)
            tracks.append(track)

        offset += len(rows)
        if len(rows) < page_size:
            break

    if not tracks:
        raise RuntimeError("No readable tracks were found in that playlist.")

    return metadata.get("name") or "Spotify playlist", tracks


def evenly_spaced_defaults(options, limit=8):
    if len(options) <= limit:
        return options
    if limit <= 1:
        return options[:1]
    indexes = [round(i * (len(options) - 1) / (limit - 1)) for i in range(limit)]
    return [options[i] for i in indexes]


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
        score = (
            0.68 * _text_similarity(track.title, title)
            + 0.32 * _text_similarity(track.artist, artist)
        )
        if score > best_score:
            best_score, best = score, item
    return best if best_score >= 72 else None


def render_spotify_preview(track):
    preview_url = track.get("preview_url")
    if preview_url:
        st.audio(preview_url, format="audio/mpeg")
        st.caption("30-second Spotify preview")
        return

    spotify_id = track.get("id")
    if spotify_id and re.fullmatch(r"[A-Za-z0-9]{10,40}", spotify_id):
        components.html(
            f'''<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{spotify_id}" width="100%" height="152" frameborder="0" allowfullscreen allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>''',
            height=165,
        )


def create_spotify_playlist(name, public, uris):
    token = spotify_access_token()
    if not token:
        raise RuntimeError("Reconnect Spotify and try again.")

    playlist = spotify_api(
        "POST",
        "me/playlists",
        token=token,
        payload={
            "name": name,
            "public": public,
            "description": "Generated from seed tracks with hybrid similarity and diversity-aware ranking.",
        },
    )
    playlist_id = playlist.get("id")
    if not playlist_id:
        raise RuntimeError("Spotify did not return a playlist ID.")

    for start in range(0, len(uris), 100):
        spotify_api(
            "POST",
            f"playlists/{playlist_id}/items",
            token=token,
            payload={"uris": uris[start : start + 100]},
        )
    return playlist


st.title("🎧 Create Similar Playlist")
st.write(
    "Start with a few songs you love and discover a fresh playlist built around "
    "their shared sound and style."
)
st.caption(
    "Collaborative similarity · music-tag matching · multi-seed ranking · "
    "diversity reranking · optional Spotify export"
)

if not LASTFM_API_KEY:
    st.error("Recommendations are temporarily unavailable because the music data service is not configured.")
    st.stop()

sp_user = spotify_client()
catalog_sp = spotify_catalog_client(CLIENT_ID, CLIENT_SECRET)

with st.sidebar:
    st.subheader("How it works")
    st.write(
        "Start with 1–8 seed tracks or import a Spotify playlist. The recommender "
        "compares listening patterns and music tags, then ranks songs that best match the overall vibe."
    )
    st.write(
        "Adjust the playlist size and diversity controls to make the results tighter, broader, or more varied."
    )
    st.write("Spotify is only needed for playlist import, previews, and saving the finished recommendations.")
    if sp_user is not None and st.button("Disconnect Spotify"):
        st.session_state.pop("spotify_token", None)
        st.session_state.pop("spotify_matches", None)
        st.session_state.pop("preview_matches", None)
        st.rerun()

input_method = st.radio(
    "Start from",
    ["Type songs", "Import Spotify playlist"],
    horizontal=True,
)

left, right = st.columns([3, 2])
selected_seeds = []

with left:
    if input_method == "Type songs":
        seed_text = st.text_area(
            "Seed tracks — one per line as Artist — Track",
            height=180,
            placeholder=(
                "Tame Impala — The Less I Know the Better\n"
                "Daft Punk — Instant Crush\n"
                "MGMT — Electric Feel"
            ),
        )
        selected_seeds = parse_seeds(seed_text)
    else:
        st.write("Paste a Spotify playlist link and choose the tracks that best represent it.")
        if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
            st.info("Spotify import becomes available after Spotify credentials are added to the app.")
        elif sp_user is None:
            login = oauth(make_state(CLIENT_SECRET)).get_authorize_url()
            st.link_button("Connect Spotify to import", login, type="primary")
            st.caption(
                "Spotify currently exposes playlist contents only for playlists you own or collaborate on, "
                "even when another playlist is public."
            )
        else:
            playlist_url = st.text_input(
                "Spotify playlist URL",
                placeholder="https://open.spotify.com/playlist/...",
            )
            if st.button("Load playlist"):
                try:
                    with st.spinner("Reading your playlist..."):
                        playlist_name, imported_tracks = load_spotify_playlist(playlist_url)
                    st.session_state["imported_playlist_name"] = playlist_name
                    st.session_state["imported_playlist_tracks"] = imported_tracks
                except Exception as exc:
                    st.error(str(exc))

            imported_tracks = st.session_state.get("imported_playlist_tracks", [])
            if imported_tracks:
                labels = [f"{track.artist} — {track.title}" for track in imported_tracks]
                lookup = {f"{track.artist} — {track.title}": track for track in imported_tracks}
                st.success(
                    f"Loaded {st.session_state.get('imported_playlist_name', 'playlist')} "
                    f"with {len(imported_tracks)} tracks."
                )
                chosen = st.multiselect(
                    "Choose up to 8 seed tracks",
                    labels,
                    default=evenly_spaced_defaults(labels, 8),
                    max_selections=8,
                    help="Eight tracks keeps recommendations fast while sampling different parts of the playlist.",
                )
                selected_seeds = [lookup[label] for label in chosen]

with right:
    count = st.slider("Playlist size", 10, 30, 20, 5)
    max_artist = st.slider("Max tracks per artist", 1, 4, 2)
    diversity = st.slider("Diversity strength", 0.00, 0.35, 0.18, 0.01)

if st.button("Generate recommendations", type="primary"):
    if not selected_seeds:
        if input_method == "Type songs":
            st.error("Add at least one track using the format `Artist — Track`.")
        else:
            st.error("Load a Spotify playlist and select at least one seed track.")
    else:
        try:
            with st.spinner("Finding tracks that fit your playlist..."):
                st.session_state["recs"] = recommend(selected_seeds, count, max_artist, diversity)
                st.session_state.pop("spotify_matches", None)
                st.session_state.pop("preview_matches", None)
        except (LastFMError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

recs = st.session_state.get("recs", [])
if recs:
    st.markdown("---")
    st.subheader("Your recommendations")
    matches = st.session_state.get("spotify_matches", {})
    preview_matches = st.session_state.setdefault("preview_matches", {})

    for i, item in enumerate(recs, 1):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{i}. {item.track.title}**  \n{item.track.artist}")
                if item.tags:
                    st.caption(" · ".join(item.tags[:6]))

                actions = st.columns([1, 1, 4])
                with actions[0]:
                    if item.track.url:
                        st.link_button("Last.fm", item.track.url)
                with actions[1]:
                    if catalog_sp is not None and st.button(
                        "▶ Preview",
                        key=f"preview_{i}_{item.track.key}",
                    ):
                        try:
                            match = spotify_match(catalog_sp, item.track)
                            if match:
                                preview_matches[item.track.key] = match
                                st.session_state["preview_matches"] = preview_matches
                            else:
                                st.warning("No Spotify preview found for this track.")
                        except Exception:
                            st.warning("Preview is temporarily unavailable.")

                preview_track = preview_matches.get(item.track.key)
                if preview_track:
                    render_spotify_preview(preview_track)

                match = matches.get(item.track.key)
                if match:
                    url = (match.get("external_urls") or {}).get("spotify")
                    if url:
                        st.link_button("Open in Spotify", url)

            with c2:
                st.metric("Match", f"{round(item.score * 100)}%")
                st.caption(
                    f"similarity {item.collaborative_score:.2f}\n"
                    f"tags {item.tag_score:.2f}\n"
                    f"coverage {item.seed_coverage:.2f}"
                )

    st.markdown("### Save to Spotify")
    if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
        st.caption("Spotify saving will appear once Spotify is connected to this app.")
    elif sp_user is None:
        login = oauth(make_state(CLIENT_SECRET)).get_authorize_url()
        st.link_button("Connect Spotify", login, type="primary")
        st.caption("Connect your Spotify account to save these recommendations as a playlist.")
    else:
        if st.button("Find these tracks on Spotify"):
            mapped = {}
            progress = st.progress(0)
            for idx, item in enumerate(recs, 1):
                try:
                    match = spotify_match(sp_user, item.track)
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
                uris = [
                    matches[x.track.key]["uri"]
                    for x in recs
                    if x.track.key in matches
                ]
                try:
                    playlist = create_spotify_playlist(name, public, uris)
                    st.success(f"Your playlist is ready with {len(uris)} tracks.")
                    url = (playlist.get("external_urls") or {}).get("spotify")
                    if url:
                        st.link_button("Open playlist in Spotify", url)
                except Exception as exc:
                    st.error(f"Spotify export failed: {exc}")

st.markdown("---")
st.caption(
    "Python · Streamlit · Last.fm API · hybrid ranking · cosine similarity · "
    "reciprocal-rank fusion · MMR · Spotify Web API"
)
