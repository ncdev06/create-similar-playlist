# src/recommender.py
from __future__ import annotations
from typing import List, Dict
import numpy as np

from src.features import build_feature_matrix, to_matrix
from src.indexer import load_index, search

try:
    from src.config import FILTER_EXPLICIT as CFG_FILTER_EXPLICIT
except Exception:
    CFG_FILTER_EXPLICIT = False


# ---------------------------
# Helpers
# ---------------------------
def _unique_preserve_order(ids: List[str]) -> List[str]:
    seen, out = set(), []
    for x in ids:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out


def _prefetch_meta(sp, ids: List[str]) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        try:
            for t in sp.tracks(ids[i:i+50]).get("tracks", []):
                if t: meta[t["id"]] = t
        except Exception:
            # ignore batch failures; we'll still fill later
            pass
    return meta


# ---------------------------
# Playlist utilities
# ---------------------------
def playlist_track_ids(sp, playlist_url_or_id: str) -> List[str]:
    pid = playlist_url_or_id.split("playlist/")[-1].split("?")[0]
    items = []
    results = sp.playlist_items(pid, additional_types=["track"], limit=100)
    items.extend(results.get("items", []))
    while results.get("next"):
        results = sp.next(results)
        items.extend(results.get("items", []))
    ids = [it.get("track", {}).get("id") for it in items if it.get("track")]
    return [x for x in ids if x]


def playlist_vector(sp, track_ids: List[str]) -> np.ndarray:
    df = build_feature_matrix(sp, track_ids)
    X = to_matrix(df)
    if X.size == 0:
        return np.zeros((1, len(X.shape) and X.shape[1] or 0), dtype="float32")
    q = X.mean(axis=0, keepdims=True)
    # normalize and de-NaN
    denom = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q = q / denom
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return q



# ---------------------------
# Reranker with hard fallback
# ---------------------------
def rerank(
    sp,
    candidate_ids: List[str],
    playlist_ids: List[str],
    k: int = 30,
    max_per_artist: int = 3,
    allow_explicit: bool | None = None,
) -> List[str]:
    """
    Multi-pass reranker that ALWAYS fills up to k if the index has enough
    non-seed items. If metadata is sparse or constraints are too tight,
    it falls back to raw ANN order (minus seeds).
    """
    if allow_explicit is None:
        allow_explicit = not CFG_FILTER_EXPLICIT

    seed_set = set(playlist_ids)
    baseline = [c for c in candidate_ids if c and c not in seed_set]
    if not baseline:
        return []

    # Try smart constraints first
    kept: List[str] = []
    kept_set = set()
    artist_count: Dict[str, int] = {}

    meta = _prefetch_meta(sp, baseline)

    def artist_id_of(t: dict) -> str | None:
        arts = t.get("artists") or []
        return arts[0].get("id") if arts else None

    def try_add(tid: str, cap: int | None, honor_explicit: bool) -> bool:
        if tid in kept_set or tid in seed_set:
            return False
        t = meta.get(tid)

        # If no metadata, accept during relaxed/fill phases
        if t is None:
            if cap is None or honor_explicit is False:
                kept.append(tid); kept_set.add(tid); return True
            return False

        if honor_explicit and t.get("explicit", False):
            return False
        aid = artist_id_of(t)
        if aid is None:
            if cap is None or honor_explicit is False:
                kept.append(tid); kept_set.add(tid); return True
            return False
        if cap is not None and artist_count.get(aid, 0) >= cap:
            return False

        artist_count[aid] = artist_count.get(aid, 0) + 1
        kept.append(tid); kept_set.add(tid)
        return True

    # Pass 1: strict (honor explicit, cap per artist)
    for tid in baseline:
        if len(kept) >= k: break
        try_add(tid, cap=max_per_artist, honor_explicit=not allow_explicit)

    # Pass 2: relax cap to 5, still honor explicit
    if len(kept) < k:
        for tid in baseline:
            if len(kept) >= k: break
            try_add(tid, cap=max(5, max_per_artist), honor_explicit=not allow_explicit)

    # Pass 3: ignore explicit, cap 5
    if len(kept) < k:
        for tid in baseline:
            if len(kept) >= k: break
            try_add(tid, cap=max(5, max_per_artist), honor_explicit=False)

    # Pass 4: ultimate fill — no cap, ignore explicit, accept even if no meta
    if len(kept) < k:
        for tid in baseline:
            if len(kept) >= k: break
            try_add(tid, cap=None, honor_explicit=False)

    # Final guard: if somehow still short, just fill from baseline order
    if len(kept) < k:
        for tid in baseline:
            if len(kept) >= k: break
            if tid not in kept_set:
                kept.append(tid); kept_set.add(tid)

    return kept[:k]


# ---------------------------
# Main entry
# ---------------------------
def recommend_for_playlist(
    sp,
    playlist_url_or_id: str,
    k: int = 30,
    max_per_artist: int = 3,
    allow_explicit: bool | None = None,
) -> List[str]:
    ids, vecs, index_obj = load_index()
    if len(ids) == 0:
        return []

    seed_ids = playlist_track_ids(sp, playlist_url_or_id)
    if not seed_ids:
        return []

    # Query vector
    q = playlist_vector(sp, seed_ids)

    # Normalize index vectors for cosine
    V = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)

    # Wide retrieve so the reranker has room
    topN = min(10000, len(ids))
    I, _ = search(index_obj, V, q, top_k=topN)
    idx_list = I[0].tolist()

    # Primary candidate pool from ANN
    cand_ids = _unique_preserve_order([ids[i] for i in idx_list])

    # If that pool is too small after seed removal, FALL BACK to whole index
    # (still respecting seed removal)
    if len([c for c in cand_ids if c not in set(seed_ids)]) < k * 5:
        cand_ids = _unique_preserve_order(
            cand_ids + [x for x in ids.tolist() if x not in set(seed_ids)]
        )

    return rerank(
        sp,
        cand_ids,
        seed_ids,
        k=k,
        max_per_artist=max_per_artist,
        allow_explicit=allow_explicit,
    )
