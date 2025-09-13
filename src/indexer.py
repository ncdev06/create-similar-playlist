import os
import numpy as np
from typing import Tuple


# Try FAISS first; fall back to sklearn NN
try:
    import faiss # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False
from sklearn.neighbors import NearestNeighbors


from .config import INDEX_DIR




def _clean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    # normalize row-wise if not already
    # (safe even if already normalized)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    X = X / denom
    # replace NaN/inf that could sneak in
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def save_index(vecs: np.ndarray, ids: list[str]) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    vecs = _clean(vecs)
    np.save(os.path.join(INDEX_DIR, "ids.npy"), np.array(ids, dtype=object))
    np.save(os.path.join(INDEX_DIR, "vecs.npy"), vecs)

    if HAVE_FAISS:
        index = faiss.IndexHNSWFlat(vecs.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.add(vecs)
        faiss.write_index(index, os.path.join(INDEX_DIR, "hnsw.index"))


def load_index() -> Tuple[np.ndarray, np.ndarray, object | None]:
    ids = np.load(os.path.join(INDEX_DIR, "ids.npy"), allow_pickle=True)
    vecs = np.load(os.path.join(INDEX_DIR, "vecs.npy"))

    # NEW: fix any lingering NaN/Inf in old artifacts
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    if HAVE_FAISS and os.path.exists(os.path.join(INDEX_DIR, "hnsw.index")):
        import faiss
        index = faiss.read_index(os.path.join(INDEX_DIR, "hnsw.index"))
        return ids, vecs, index
    else:
        nn = NearestNeighbors(metric="cosine", n_neighbors=100)
        nn.fit(vecs)
        return ids, vecs, nn




def search(index_obj, vecs: np.ndarray, query: np.ndarray, top_k: int = 2000):
    if HAVE_FAISS and not isinstance(index_obj, NearestNeighbors):
        import faiss
        D, I = index_obj.search(query.astype("float32"), top_k)
        return I, D
    else:
        # sklearn
        distances, indices = index_obj.kneighbors(query, n_neighbors=min(top_k, vecs.shape[0]))
        return indices, distances