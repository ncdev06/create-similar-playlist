# src/corpus_builder.py
from __future__ import annotations
from typing import List, Tuple, Set, Dict
import re

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
ID_RE = re.compile(r"^[0-9A-Za-z]{22}$")

def _is_id(x: str | None) -> bool:
    """Return True if x looks like a valid Spotify base62 ID (22 chars)."""
    return bool(x and ID_RE.match(x))

def unique(seq: List[str]) -> List[str]:
    """De-dupe while preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -----------------------------------------------------------------------------
# Low-level fetchers (playlist -> tracks, tracks -> seed artists)
# -----------------------------------------------------------------------------
def _playlist_track_ids(sp, playlist_id_or_url: str) -> List[str]:
    """
    Extract track IDs from a public playlist URL or ID.
    Skips entries without a proper track object/ID.
    """
    pid = playlist_id_or_url.split("playlist/")[-1].split("?")[0]
    items: List[dict] = []

    try:
        results = sp.playlist_items(pid, additional_types=["track"], limit=100)
    except Exception:
        return []

    items.extend(results.get("items", []))
    while results.get("next"):
        results = sp.next(results)
        items.extend(results.get("items", []))

    ids = [it.get("track", {}).get("id") for it in items if it.get("track")]
    ids = [x for x in ids if _is_id(x)]
    return ids

def _seed_artist_ids(sp, track_ids: List[str]) -> List[str]:
    """
    From a list of track IDs, get the primary artist ID for each track.
    Only returns valid 22-char IDs and preserves order (deduped).
    """
    aids: List[str] = []
    for i in range(0, len(track_ids), 50):
        chunk_ids = track_ids[i:i+50]
        try:
            tracks = sp.tracks(chunk_ids).get("tracks", [])
        except Exception:
            tracks = []
        for t in tracks or []:
            arts = (t or {}).get("artists") or []
            if not arts:
                continue
            aid = arts[0].get("id")
            if _is_id(aid):
                aids.append(aid)

    # de-dupe while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for a in aids:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out

# -----------------------------------------------------------------------------
# Safe wrappers around Spotify endpoints (avoid throwing, validate IDs first)
# -----------------------------------------------------------------------------
def _safe_top_tracks(sp, artist_id: str, country: str = "US") -> List[str]:
    """
    Return up to 10 top track IDs for an artist in a country.
    Returns [] on any error or non-ID values.
    """
    if not _is_id(artist_id):
        return []
    try:
        tops = sp.artist_top_tracks(artist_id, country=country).get("tracks", [])
        return [t.get("id") for t in tops if _is_id(t.get("id"))]
    except Exception:
        return []

def _safe_related_artists(sp, artist_id: str) -> List[str]:
    """
    Return related artist IDs. Returns [] on 404/any error/bad IDs.
    """
    if not _is_id(artist_id):
        return []
    try:
        rel = sp.artist_related_artists(artist_id).get("artists", [])
        return [a.get("id") for a in rel if _is_id(a.get("id"))]
    except Exception:
        # 404s and other errors are common in the wild; just skip quietly
        return []

def _try_recommendations(sp, seed_tracks: List[str], limit: int = 80) -> List[str]:
    """
    Optional helper: per-track recommendations. Some accounts/regions 404 this
    endpoint; we ignore failures and return [] for those seeds.
    """
    outs: List[str] = []
    for tid in seed_tracks:
        if not _is_id(tid):
            continue
        try:
            recs = sp.recommendations(seed_tracks=[tid], limit=min(100, limit)).get("tracks", [])
            outs.extend([t.get("id") for t in recs if _is_id(t.get("id"))])
        except Exception:
            # ignore; best-effort only
            pass
    return outs

# -----------------------------------------------------------------------------
# Public entry point used by Step 1
# -----------------------------------------------------------------------------
def expand_corpus_from_playlist(
    sp,
    playlist_url_or_id: str,
    limit_per_seed: int = 80,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Expand candidate track IDs starting from a public playlist.

    Stages:
      A) Seed artists' top tracks (per artist)
      B) Related artists (cap ~10 per seed artist) -> each artist's top tracks
      C) OPTIONAL: per-track recommendations (best-effort; skipped on failure)

    Returns:
      all_ids: List[str]  — de-duped candidate track IDs (includes seed tracks)
      stats:   Dict[str,int] — counts per stage for UI/debugging
    """
    # 0) Seeds
    seed_tracks = _playlist_track_ids(sp, playlist_url_or_id)
    seed_artists = _seed_artist_ids(sp, seed_tracks)

    stats: Dict[str, int] = {
        "seed_tracks": len(seed_tracks),
        "seed_artists": len(seed_artists),
        "artist_tops": 0,
        "related_artists": 0,
        "related_tops": 0,
        "recs_optional": 0,
    }

    cand: List[str] = []

    # A) Seed artists' top tracks
    for aid in seed_artists:
        cand.extend(_safe_top_tracks(sp, aid, country="US"))
    stats["artist_tops"] = len(unique(cand))

    # B) Related artists (widen but cap per seed)
    rel_ids: List[str] = []
    for aid in seed_artists:
        rel_ids.extend(_safe_related_artists(sp, aid)[:10])  # cap to avoid explosion
    rel_ids = unique(rel_ids)
    stats["related_artists"] = len(rel_ids)

    # ...then their top tracks
    rel_tops: List[str] = []
    for rid in rel_ids:
        rel_tops.extend(_safe_top_tracks(sp, rid, country="US"))
    cand.extend(rel_tops)
    stats["related_tops"] = len(unique(cand)) - stats["artist_tops"]

    # C) OPTIONAL: per-track recommendations (ignore failures)
    recs = _try_recommendations(sp, seed_tracks, limit=limit_per_seed)
    cand.extend(recs)
    stats["recs_optional"] = len(unique(cand)) - stats["artist_tops"] - stats["related_tops"]

    # Final: include seeds + all candidates, de-duped
    all_ids = unique(seed_tracks + cand)
    return all_ids, stats
