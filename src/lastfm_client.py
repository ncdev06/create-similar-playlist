from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class LastFMError(RuntimeError):
    """Raised when Last.fm returns an API or transport error."""


@dataclass(frozen=True)
class TrackRef:
    artist: str
    title: str
    mbid: str = ""
    url: str = ""

    @property
    def key(self) -> str:
        return f"{self.artist.strip().casefold()}::{self.title.strip().casefold()}"


@dataclass(frozen=True)
class SimilarTrack:
    track: TrackRef
    match: float


class LastFMClient:
    BASE_URL = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str, timeout: int = 12) -> None:
        if not api_key:
            raise ValueError("LASTFM_API_KEY is required")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "CreateSimilarPlaylist/2.0 (portfolio project)"}
        )

    def _get(self, method: str, **params: Any) -> dict[str, Any]:
        query = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            "autocorrect": 1,
            **params,
        }
        try:
            response = self.session.get(
                self.BASE_URL, params=query, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LastFMError(f"Last.fm request failed: {exc}") from exc

        if "error" in data:
            raise LastFMError(
                f"Last.fm error {data.get('error')}: {data.get('message', 'unknown error')}"
            )
        return data

    def similar_tracks(self, seed: TrackRef, limit: int = 40) -> list[SimilarTrack]:
        data = self._get(
            "track.getSimilar",
            artist=seed.artist,
            track=seed.title,
            limit=max(1, min(limit, 100)),
        )
        raw_tracks = data.get("similartracks", {}).get("track", [])
        results: list[SimilarTrack] = []

        for item in raw_tracks:
            artist_value = item.get("artist", {})
            artist = (
                artist_value.get("name", "")
                if isinstance(artist_value, dict)
                else str(artist_value)
            )
            title = str(item.get("name", "")).strip()
            if not artist or not title:
                continue
            try:
                match = float(item.get("match", 0.0))
            except (TypeError, ValueError):
                match = 0.0

            results.append(
                SimilarTrack(
                    track=TrackRef(
                        artist=artist.strip(),
                        title=title,
                        mbid=str(item.get("mbid", "") or ""),
                        url=str(item.get("url", "") or ""),
                    ),
                    match=max(0.0, min(match, 1.0)),
                )
            )

        return results

    def top_tags(self, track: TrackRef, limit: int = 12) -> list[str]:
        data = self._get(
            "track.getTopTags",
            artist=track.artist,
            track=track.title,
        )
        tags = data.get("toptags", {}).get("tag", [])
        out: list[str] = []
        for tag in tags:
            name = str(tag.get("name", "")).strip().casefold()
            if name and name not in out:
                out.append(name)
            if len(out) >= limit:
                break
        return out
