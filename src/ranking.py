from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt

from src.lastfm_client import TrackRef


@dataclass(frozen=True)
class RankedTrack:
    track: TrackRef
    score: float
    collaborative_score: float
    tag_score: float
    seed_coverage: float
    tags: tuple[str, ...]


def build_scores(seeds, similar_by_seed):
    seed_keys = {seed.key for seed in seeds}
    tracks = {}
    match_sum = {}
    reciprocal_rank = {}
    coverage_sets = {}

    for seed in seeds:
        for rank, item in enumerate(similar_by_seed.get(seed.key, []), start=1):
            key = item.track.key
            if key in seed_keys:
                continue
            tracks[key] = item.track
            match_sum[key] = match_sum.get(key, 0.0) + item.match
            reciprocal_rank[key] = reciprocal_rank.get(key, 0.0) + 1.0 / (20 + rank)
            coverage_sets.setdefault(key, set()).add(seed.key)

    def normalize(values):
        maximum = max(values.values(), default=0.0) or 1.0
        return {key: value / maximum for key, value in values.items()}

    match_norm = normalize(match_sum)
    rank_norm = normalize(reciprocal_rank)

    collaborative = {
        key: 0.72 * match_norm.get(key, 0.0) + 0.28 * rank_norm.get(key, 0.0)
        for key in tracks
    }
    coverage = {
        key: len(coverage_sets.get(key, set())) / max(1, len(seeds))
        for key in tracks
    }
    return tracks, collaborative, coverage


def _tag_terms(tags):
    terms = []
    for tag in tags:
        value = " ".join(str(tag).casefold().strip().split())
        if not value:
            continue
        terms.append(value)
        # Exact genre phrases are useful, but individual words help related tags overlap.
        words = [word for word in value.split() if len(word) > 2]
        terms.extend(words)
    return terms


def _tfidf_vectors(tag_lists):
    documents = [_tag_terms(tags) for tags in tag_lists]
    document_count = max(1, len(documents))
    doc_frequency = Counter()
    for terms in documents:
        doc_frequency.update(set(terms))

    vectors = []
    for terms in documents:
        counts = Counter(terms)
        vector = {}
        for term, count in counts.items():
            idf = log((1 + document_count) / (1 + doc_frequency[term])) + 1.0
            vector[term] = (1.0 + log(count)) * idf
        norm = sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({term: value / norm for term, value in vector.items()})
    return vectors


def _cosine(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def _mean_vector(vectors):
    if not vectors:
        return {}
    out = {}
    for vector in vectors:
        for term, value in vector.items():
            out[term] = out.get(term, 0.0) + value / len(vectors)
    norm = sqrt(sum(value * value for value in out.values())) or 1.0
    return {term: value / norm for term, value in out.items()}


def rank_candidates(
    seeds,
    tracks,
    collaborative,
    coverage,
    tags_by_key,
    k=20,
    max_per_artist=2,
    diversity_strength=0.18,
):
    ordered = sorted(
        tracks.values(),
        key=lambda track: (
            collaborative.get(track.key, 0.0),
            coverage.get(track.key, 0.0),
        ),
        reverse=True,
    )

    all_tracks = list(seeds) + ordered
    vectors = _tfidf_vectors([tags_by_key.get(track.key, []) for track in all_tracks])
    seed_vectors = vectors[: len(seeds)]
    candidate_vectors = vectors[len(seeds) :]
    seed_profile = _mean_vector(seed_vectors)

    tag_score = {
        track.key: _cosine(vector, seed_profile)
        for track, vector in zip(ordered, candidate_vectors)
    }

    base_score = {
        track.key: (
            0.60 * collaborative.get(track.key, 0.0)
            + 0.25 * tag_score.get(track.key, 0.0)
            + 0.15 * coverage.get(track.key, 0.0)
        )
        for track in ordered
    }

    selected = []
    selected_indices = []
    artist_counts = {}
    remaining = set(range(len(ordered)))

    while remaining and len(selected) < k:
        best_index = None
        best_value = float("-inf")

        for index in remaining:
            track = ordered[index]
            artist_key = track.artist.casefold()
            if artist_counts.get(artist_key, 0) >= max_per_artist:
                continue

            redundancy = 0.0
            if selected_indices:
                redundancy = max(
                    _cosine(candidate_vectors[index], candidate_vectors[chosen])
                    for chosen in selected_indices
                )

            value = base_score[track.key] - diversity_strength * redundancy
            if value > best_value:
                best_index = index
                best_value = value

        if best_index is None:
            break

        remaining.remove(best_index)
        selected_indices.append(best_index)
        track = ordered[best_index]
        artist_key = track.artist.casefold()
        artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1

        selected.append(
            RankedTrack(
                track=track,
                score=round(max(0.0, best_value), 4),
                collaborative_score=round(collaborative.get(track.key, 0.0), 4),
                tag_score=round(tag_score.get(track.key, 0.0), 4),
                seed_coverage=round(coverage.get(track.key, 0.0), 4),
                tags=tuple(tags_by_key.get(track.key, [])[:6]),
            )
        )

    return selected
