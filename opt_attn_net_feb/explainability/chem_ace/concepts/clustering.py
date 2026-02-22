from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import logging
from typing import Dict, Iterable, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..config import ConceptDiscoveryConfig
from ..optional_deps import has_hdbscan, require_hdbscan
from ..types import ConceptCandidate, ConceptMembership, PatchEmbeddingRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredConceptSet:
    """Output container for a single concept-discovery snapshot."""

    layer_name: str
    candidates: list[ConceptCandidate]
    memberships: list[ConceptMembership]
    metadata: dict[str, object]



def _coherence_from_points(points: np.ndarray, centroid: np.ndarray) -> float:
    dists = np.linalg.norm(points - centroid.reshape(1, -1), axis=1)
    return float(1.0 / (1.0 + float(np.mean(dists))))



def _centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0).astype(np.float32)



def _medoid_index(points: np.ndarray, centroid: np.ndarray) -> int:
    dists = np.linalg.norm(points - centroid.reshape(1, -1), axis=1)
    return int(np.argmin(dists))



def _cluster_with_kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    kk = max(2, min(int(k), int(x.shape[0])))
    model = KMeans(n_clusters=kk, random_state=int(seed), n_init=10)
    return model.fit_predict(x)



def _cluster_with_hierarchical(x: np.ndarray, distance_threshold: float) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=float(distance_threshold))
    return model.fit_predict(x)



def _cluster_with_hdbscan(x: np.ndarray, min_cluster_size: int) -> np.ndarray:
    hdbscan = require_hdbscan()
    model = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)))
    return model.fit_predict(x)



def _algorithm_labels(
    *,
    x: np.ndarray,
    config: ConceptDiscoveryConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for algo in config.algorithms:
        key = str(algo).lower()
        if key == "kmeans":
            out[key] = _cluster_with_kmeans(x, k=config.kmeans_k, seed=seed)
        elif key == "hierarchical":
            out[key] = _cluster_with_hierarchical(x, distance_threshold=config.hierarchical_distance_threshold)
        elif key == "hdbscan":
            if has_hdbscan():
                out[key] = _cluster_with_hdbscan(x, min_cluster_size=config.hdbscan_min_cluster_size)
            else:
                logger.warning("hdbscan not installed; skipping hdbscan clustering")
        else:
            logger.warning("Unknown clustering algorithm requested", extra={"algorithm": algo})
    return out



def _concept_id(layer_name: str, algorithm: str, patch_ids: list[str]) -> str:
    payload = f"{layer_name}|{algorithm}|{'|'.join(sorted(patch_ids))}"
    return sha1(payload.encode("utf-8")).hexdigest()



def _build_from_labels(
    *,
    x: np.ndarray,
    patch_ids: list[str],
    layer_name: str,
    algorithm: str,
    labels: np.ndarray,
    config: ConceptDiscoveryConfig,
) -> tuple[list[ConceptCandidate], list[ConceptMembership]]:
    candidates: list[ConceptCandidate] = []
    memberships: list[ConceptMembership] = []

    for cluster_id in sorted(set(int(v) for v in labels.tolist())):
        if cluster_id < 0:
            continue
        idx = np.where(labels == cluster_id)[0]
        if idx.size == 0:
            continue

        support = int(idx.size)
        if support < int(config.min_support):
            continue

        points = x[idx]
        cent = _centroid(points)
        coherence = _coherence_from_points(points, cent)
        if coherence < float(config.min_coherence):
            continue

        medoid_local = _medoid_index(points, cent)
        medoid_patch_id = patch_ids[int(idx[medoid_local])]
        concept_local_id = _concept_id(
            layer_name=layer_name,
            algorithm=algorithm,
            patch_ids=[patch_ids[int(i)] for i in idx.tolist()],
        )

        candidates.append(
            ConceptCandidate(
                concept_local_id=concept_local_id,
                layer_name=str(layer_name),
                algorithm=str(algorithm),
                support=support,
                coherence=float(coherence),
                centroid=cent,
                medoid_patch_id=str(medoid_patch_id),
                metadata={"cluster_label": int(cluster_id)},
            )
        )

        dists = np.linalg.norm(points - cent.reshape(1, -1), axis=1)
        for local_i, global_i in enumerate(idx.tolist()):
            dist = float(dists[int(local_i)])
            memberships.append(
                ConceptMembership(
                    concept_local_id=concept_local_id,
                    patch_id=str(patch_ids[int(global_i)]),
                    membership_score=float(1.0 / (1.0 + dist)),
                    distance_to_centroid=dist,
                )
            )

    return candidates, memberships



def _deduplicate_concepts(
    *,
    candidates: list[ConceptCandidate],
    memberships: list[ConceptMembership],
    similarity_threshold: float,
) -> tuple[list[ConceptCandidate], list[ConceptMembership]]:
    if len(candidates) <= 1:
        return candidates, memberships

    ordered = sorted(candidates, key=lambda c: (c.support, c.coherence), reverse=True)
    kept: list[ConceptCandidate] = []
    kept_ids: set[str] = set()
    dropped_to_kept: dict[str, str] = {}

    for cand in ordered:
        if not kept:
            kept.append(cand)
            kept_ids.add(cand.concept_local_id)
            continue
        sims = cosine_similarity(
            cand.centroid.reshape(1, -1),
            np.stack([k.centroid for k in kept], axis=0),
        )[0]
        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])
        if max_sim >= float(similarity_threshold):
            dropped_to_kept[cand.concept_local_id] = kept[max_idx].concept_local_id
        else:
            kept.append(cand)
            kept_ids.add(cand.concept_local_id)

    remapped_memberships: list[ConceptMembership] = []
    for m in memberships:
        target = dropped_to_kept.get(m.concept_local_id, m.concept_local_id)
        if target in kept_ids:
            remapped_memberships.append(
                ConceptMembership(
                    concept_local_id=target,
                    patch_id=m.patch_id,
                    membership_score=m.membership_score,
                    distance_to_centroid=m.distance_to_centroid,
                )
            )

    unique_memberships: dict[tuple[str, str], ConceptMembership] = {}
    for m in remapped_memberships:
        key = (m.concept_local_id, m.patch_id)
        prev = unique_memberships.get(key)
        if prev is None or m.membership_score > prev.membership_score:
            unique_memberships[key] = m

    return kept, list(unique_memberships.values())



def discover_concepts(
    *,
    embeddings: Iterable[PatchEmbeddingRecord],
    config: ConceptDiscoveryConfig,
    seed: int,
) -> DiscoveredConceptSet:
    """Run ACE-style concept discovery from patch embeddings."""
    emb_list = list(embeddings)
    if len(emb_list) == 0:
        return DiscoveredConceptSet(layer_name="", candidates=[], memberships=[], metadata={"n_embeddings": 0})

    layer_names = {e.layer_name for e in emb_list}
    if len(layer_names) != 1:
        raise ValueError("All embeddings in one discovery call must come from the same layer_name")

    layer_name = next(iter(layer_names))
    patch_ids = [e.patch_id for e in emb_list]
    x = np.stack([np.asarray(e.vector, dtype=np.float32) for e in emb_list], axis=0)

    labels_per_algo = _algorithm_labels(x=x, config=config, seed=seed)

    all_candidates: list[ConceptCandidate] = []
    all_memberships: list[ConceptMembership] = []
    for algo, labels in labels_per_algo.items():
        cands, mems = _build_from_labels(
            x=x,
            patch_ids=patch_ids,
            layer_name=layer_name,
            algorithm=algo,
            labels=labels,
            config=config,
        )
        all_candidates.extend(cands)
        all_memberships.extend(mems)

    dedup_candidates, dedup_memberships = _deduplicate_concepts(
        candidates=all_candidates,
        memberships=all_memberships,
        similarity_threshold=float(config.dedup_centroid_similarity_threshold),
    )

    return DiscoveredConceptSet(
        layer_name=layer_name,
        candidates=dedup_candidates,
        memberships=dedup_memberships,
        metadata={
            "n_embeddings": int(len(emb_list)),
            "n_algorithms": int(len(labels_per_algo)),
            "n_candidates_before_dedup": int(len(all_candidates)),
            "n_candidates_after_dedup": int(len(dedup_candidates)),
        },
    )


__all__ = ["DiscoveredConceptSet", "discover_concepts"]
