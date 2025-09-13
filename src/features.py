# src/features.py
from __future__ import annotations
from typing import List
import re, time
import numpy as np
import pandas as pd
from spotipy import SpotifyException

# -----------------------
# Field definitions
# -----------------------
AUDIO_FIELDS = [
    "acousticness","danceability","energy","instrumentalness",
    "liveness","speechiness","valence","tempo","loudness"
]
NUMERIC_FEATURES = [
    "danceability","energy","key","loudness","mode","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms","time_signature"
]

META_FIELDS = ["popularity","year"]
ALL_FIELDS = AUDIO_FIELDS + META_FIELDS

ID_RE = re.compile(r"^[0-9A-Za-z]{22}$")  # Spotify ID shape


# -----------------------
# Helpers
# -----------------------
def year_from_date(date_str: str | None) -> int:
    try:
        return int((date_str or "")[:4])
    except Exception:
        return 0

def _clean_ids(ids: List[str]) -> List[str]:
    seen, out = set(), []
    for x in ids:
        if not x: continue
        if x in seen: continue
        if not ID_RE.match(x): continue
        seen.add(x); out.append(x)
    return out

def _ensure_audio_df(ids: List[str], df: pd.DataFrame | None) -> pd.DataFrame:
    """Ensure a frame exists with 'id' and all AUDIO_FIELDS."""
    if df is None or df.empty or "id" not in df.columns:
        df = pd.DataFrame({"id": ids})
    for col in AUDIO_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    return df

def _ensure_meta_df(ids: List[str], df: pd.DataFrame | None) -> pd.DataFrame:
    """Ensure a frame exists with 'id', popularity, year."""
    if df is None or df.empty or "id" not in df.columns:
        df = pd.DataFrame({"id": ids})
    if "popularity" not in df.columns:
        df["popularity"] = 0
    if "year" not in df.columns:
        df["year"] = 0
    return df


# -----------------------
# Spotify fetchers (batched + robust)
# -----------------------
def fetch_track_audio_features(sp, track_ids: List[str]) -> pd.DataFrame:
    rows = []
    ids = _clean_ids(track_ids)

    def call_chunk(chunk: List[str]):
        feats = sp.audio_features(tracks=chunk)
        for tid, f in zip(chunk, feats):
            if f is None:
                rows.append({"id": tid, **{k: np.nan for k in AUDIO_FIELDS}})
            else:
                rows.append({"id": tid, **{k: f.get(k, np.nan) for k in AUDIO_FIELDS}})

    i, B = 0, 50  # <=100 allowed; start conservative
    while i < len(ids):
        chunk = ids[i:i+B]
        try:
            call_chunk(chunk)
            i += B
        except SpotifyException as e:
            status = getattr(e, "http_status", None)
            if status == 429:  # rate limit
                retry = 2
                try: retry = int(getattr(e, "headers", {}).get("Retry-After", 2))
                except Exception: pass
                time.sleep(max(retry, 2)); continue
            if B > 1:
                B = max(1, B // 2); continue
            i += 1  # skip single problematic id
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["id"]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df

def fetch_track_meta(sp, track_ids: List[str]) -> pd.DataFrame:
    out = []
    ids = _clean_ids(track_ids)
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        try:
            tracks = sp.tracks(chunk).get("tracks", [])
        except SpotifyException:
            # try halves; if that also fails, skip
            tracks = []
            if len(chunk) > 1:
                mid = len(chunk)//2
                for half in (chunk[:mid], chunk[mid:]):
                    try:
                        tracks += sp.tracks(half).get("tracks", [])
                    except SpotifyException:
                        pass
        for t in tracks or []:
            if not t: continue
            out.append({
                "id": t.get("id"),
                "popularity": t.get("popularity", 0),
                "year": year_from_date(t.get("album", {}).get("release_date")),
                "explicit": t.get("explicit", False),
                "artist_id": (t.get("artists") or [{}])[0].get("id"),
                "name": t.get("name"),
                "uri": t.get("uri"),
            })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.dropna(subset=["id"]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


# -----------------------
# Feature matrix builders
# -----------------------
def build_feature_matrix(sp, track_ids: List[str]) -> pd.DataFrame:
    ids = _clean_ids(track_ids)
    meta = fetch_track_meta(sp, ids)
    af = fetch_track_audio_features(sp, ids)

    # ensure both frames have id + required cols
    meta = _ensure_meta_df(ids if meta.empty else meta["id"].tolist(), meta)
    af   = _ensure_audio_df(ids if af.empty else af["id"].tolist(), af)

    # Align on the union of all ids
    union_ids = list(dict.fromkeys((meta["id"].tolist() or []) + (af["id"].tolist() or [])))
    meta = meta[meta["id"].isin(union_ids)]
    af   = af[af["id"].isin(union_ids)]

    df = (
        meta.merge(af, on="id", how="outer")
            .dropna(subset=["id"])
            .drop_duplicates(subset=["id"])
            .reset_index(drop=True)
    )

    # Impute audio fields
    for col in AUDIO_FIELDS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median(skipna=True))

    # Ensure meta cols exist
    for col in ["popularity", "year"]:
        if col not in df:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Standardize
    for col in ALL_FIELDS:
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        mu, sigma = float(np.nanmean(x)), float(np.nanstd(x))
        df[col] = (x - mu) / (sigma if sigma > 1e-8 else 1.0)

    return df

def to_vector_row(df_row: pd.Series) -> np.ndarray:
    return df_row[ALL_FIELDS].to_numpy(dtype=float)

def to_matrix(df: pd.DataFrame) -> np.ndarray:
    # make sure all expected columns exist, missing -> 0
    df = df.copy()
    for c in NUMERIC_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    # keep only the expected cols, coerce to float32
    X = (
        df[NUMERIC_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
    # final guard (just in case)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X