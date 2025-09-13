# --- add project root to sys.path so "src" is importable ---
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # folder that contains /src
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------

import streamlit as st
import numpy as np
from typing import List
from src.spotify_client import get_spotify
from src.corpus_builder import expand_corpus_from_playlist
from src.features import build_feature_matrix, to_matrix
from src.indexer import save_index
from src.recommender import recommend_for_playlist



st.set_page_config(page_title="Create Similar Playlist", layout="wide")
st.title("🎧 Create Similar Playlist (MVP)")

sp = get_spotify()

# -------------------
# Step 1
# -------------------
# -------------------
# Step 1
# -------------------
with st.expander("Step 1 — Build or refresh the candidate index", expanded=True):
    playlist_seed = st.text_input(
        "Enter ANY public playlist URL to bootstrap the candidate corpus (can be your target too)"
    )
    if st.button("Build/Refresh Index"):
        if not playlist_seed.strip():
            st.error("Please paste a public playlist URL.")
        else:
            with st.spinner("Expanding corpus (this may take a minute)..."):
                all_ids, stats = expand_corpus_from_playlist(sp, playlist_seed, limit_per_seed=80)

            # Read stats defensively to support both old/new corpus_builder schemas
            seed_tracks     = stats.get("seed_tracks", stats.get("seed", 0))
            seed_artists    = stats.get("seed_artists", 0)
            per_track_recs  = stats.get("per_track", 0)
            multi_seed_recs = stats.get("multi_seed", 0)
            artist_tops     = stats.get("artist_tops", 0)
            related_artists = stats.get("related_artists", 0)
            related_tops    = stats.get("related_tops", 0)
            recs_optional   = stats.get("recs_optional", 0)

            # Show whichever numbers are available
            st.info(
                " | ".join([
                    f"Seed tracks: {seed_tracks}",
                    *( [f"Seed artists: {seed_artists}"] if seed_artists else [] ),
                    *( [f"per-track recs: {per_track_recs}"] if per_track_recs or "per_track" in stats else [] ),
                    *( [f"multi-seed: {multi_seed_recs}"] if multi_seed_recs or "multi_seed" in stats else [] ),
                    *( [f"artist tops: {artist_tops}"] if artist_tops else [] ),
                    *( [f"related artists: {related_artists}"] if related_artists else [] ),
                    *( [f"related tops: {related_tops}"] if related_tops else [] ),
                    *( [f"recs (optional): {recs_optional}"] if recs_optional else [] ),
                ])
            )

            if stats.get("error"):
                st.warning(f"Expansion note: {stats['error']}")

            if len(all_ids) <= (seed_tracks or 0):
                st.warning(
                    "Expansion returned ~no new candidates. "
                    "Try a different seed playlist. (We’ll still build the index so Step 2 can run.)"
                )

            with st.spinner("Computing features and building index..."):
                df = build_feature_matrix(sp, all_ids)
                X = to_matrix(df)
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                save_index(X.astype("float32"), df["id"].tolist())
            st.success(f"Indexed {len(all_ids)} tracks.")


st.markdown("---")

# -------------------
# Step 2
# -------------------
with st.expander("Step 2 — Generate a similar playlist", expanded=True):
    target_playlist = st.text_input("Target public playlist URL", "")
    k = st.slider("How many recommendations?", min_value=10, max_value=100, value=30, step=5)
    max_per_artist = st.slider("Max per artist (diversity)", min_value=1, max_value=5, value=3)
    allow_explicit = st.checkbox("Allow explicit tracks", value=True)

    if st.button("Generate"):
        with st.spinner("Finding similar tracks..."):
            rec_ids = recommend_for_playlist(
                sp, target_playlist, k=k
            )
        st.session_state["rec_ids"] = rec_ids  # save across reruns

    if "rec_ids" in st.session_state and st.session_state["rec_ids"]:
        rec_ids = st.session_state["rec_ids"]
        st.success(f"Got {len(rec_ids)} tracks.")
        tracks = sp.tracks(rec_ids)["tracks"]
        cols = st.columns(5)
        for i, t in enumerate(tracks):
            with cols[i % 5]:
                st.image(t["album"]["images"][1]["url"], use_column_width=True)
                st.caption(f"{t['name']} — {t['artists'][0]['name']}")

        # Create playlist button (now works!)
        if st.button("Create playlist in my Spotify"):
            user = sp.current_user()
            pl_name = "Similar to: " + target_playlist.split("/")[-1][:20]
            new_pl = sp.user_playlist_create(user["id"], name=pl_name, public=True)
            sp.playlist_add_items(new_pl["id"], rec_ids)
            st.success("✅ Created! Open it in Spotify app.")

