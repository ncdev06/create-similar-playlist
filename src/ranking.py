from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    seed_keys={s.key for s in seeds}; tracks={}; match={}; rrf={}; cov={}
    for seed in seeds:
        for rank,item in enumerate(similar_by_seed.get(seed.key,[]),1):
            k=item.track.key
            if k in seed_keys: continue
            tracks[k]=item.track; match[k]=match.get(k,0)+item.match; rrf[k]=rrf.get(k,0)+1/(20+rank); cov.setdefault(k,set()).add(seed.key)
    def norm(d):
        m=max(d.values(),default=0) or 1
        return {k:v/m for k,v in d.items()}
    mn,rn=norm(match),norm(rrf)
    collab={k:.72*mn.get(k,0)+.28*rn.get(k,0) for k in tracks}
    coverage={k:len(cov.get(k,set()))/max(1,len(seeds)) for k in tracks}
    return tracks,collab,coverage

def rank_candidates(seeds,tracks,collab,coverage,tags_by_key,k=20,max_per_artist=2,diversity_strength=.18):
    ordered=sorted(tracks.values(),key=lambda t:(collab.get(t.key,0),coverage.get(t.key,0)),reverse=True)
    docs=[' '.join(tags_by_key.get(t.key,[])) for t in seeds+ordered]
    if any(x.strip() for x in docs):
        mat=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True).fit_transform(docs)
        seed_prof=mat[:len(seeds)].mean(axis=0); cand=mat[len(seeds):]
        tagvals=cosine_similarity(cand,seed_prof).ravel()
    else:
        cand=np.zeros((len(ordered),1)); tagvals=np.zeros(len(ordered))
    tagscore={t.key:float(s) for t,s in zip(ordered,tagvals)}
    base={t.key:.60*collab.get(t.key,0)+.25*tagscore.get(t.key,0)+.15*coverage.get(t.key,0) for t in ordered}
    selected=[]; idxs=[]; counts={}; remaining=set(range(len(ordered)))
    while remaining and len(selected)<k:
        best=None; bestv=-1e9
        for i in remaining:
            t=ordered[i]; a=t.artist.casefold()
            if counts.get(a,0)>=max_per_artist: continue
            redundancy=0.0
            if idxs and cand.shape[1]>0: redundancy=float(np.max(cosine_similarity(cand[i],cand[idxs]).ravel()))
            v=base[t.key]-diversity_strength*redundancy
            if v>bestv: best,bestv=i,v
        if best is None: break
        remaining.remove(best); idxs.append(best); t=ordered[best]; a=t.artist.casefold(); counts[a]=counts.get(a,0)+1
        selected.append(RankedTrack(t,round(max(0,bestv),4),round(collab.get(t.key,0),4),round(tagscore.get(t.key,0),4),round(coverage.get(t.key,0),4),tuple(tags_by_key.get(t.key,[])[:6])))
    return selected
