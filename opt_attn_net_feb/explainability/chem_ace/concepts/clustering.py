from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import logging
from typing import Dict, Iterable, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..config import ConceptDiscoveryConfig
from ..optional_deps import has_hdbscan, require_hdbscan
from ..types import ConceptCandidate, ConceptMembership, PatchEmbeddingRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredConceptSet:
    """
    Represents a discovered concept set in a specific neural network layer.

    This class is used to encapsulate information about a concept set that has
    been discovered. It includes the layer name, candidate concepts, membership
    information, and any additional metadata associated with the discovered
    concepts.

    Attributes:
        layer_name: The name of the neural network layer associated with the
            concept set.
        candidates: A list of candidate concepts in the discovered concept set.
        memberships: A list of concept membership details that associate
            instances to the discovered concepts.
        metadata: A dictionary containing additional metadata for the
            discovered concept set.
    """

    layer_name: str
    candidates: list[ConceptCandidate]
    memberships: list[ConceptMembership]
    metadata: dict[str, object]



def _coherence_from_points(points: np.ndarray, centroid: np.ndarray) -> float:
    """
    Calculates the coherence score from the given set of points and a centroid.

    The coherence score is determined based on the average distance of points
    from the centroid. A lower average distance results in a higher coherence score.

    Parameters:
    points (np.ndarray): A NumPy array representing the points in a multidimensional
        space.
    centroid (np.ndarray): A NumPy array representing the centroid in the same
        dimensional space as the points.

    Returns:
    float: The calculated coherence score, inversely proportional to the average
        distance of the points to the centroid.
    """
    dists = np.linalg.norm(points - centroid.reshape(1, -1), axis=1)
    return float(1.0 / (1.0 + float(np.mean(dists))))



def _centroid(points: np.ndarray) -> np.ndarray:
    """
    Computes the centroid of a given set of points by calculating the mean along the
    specified axis.

    Parameters:
    points (np.ndarray): A NumPy array containing the points for which the centroid
        is to be computed.

    Returns:
    np.ndarray: A NumPy array containing the centroid coordinates as floating-point
        numbers.
    """
    return points.mean(axis=0).astype(np.float32)



def _medoid_index(points: np.ndarray, centroid: np.ndarray) -> int:
    """
    Finds the index of the medoid closest to the given centroid.

    A medoid is the most centrally located point in a dataset. This function
    computes the distance of each point in the dataset from the given centroid
    and returns the index of the point with the smallest distance.

    Parameters:
    points (np.ndarray): A 2D array of points, where each row represents a point
        in n-dimensional space.
    centroid (np.ndarray): A 1D array representing the centroid in n-dimensional
        space.

    Returns:
    int: The index of the medoid point in the dataset.
    """
    dists = np.linalg.norm(points - centroid.reshape(1, -1), axis=1)
    return int(np.argmin(dists))



def _cluster_with_kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Clusters data using the KMeans clustering algorithm.

    This function utilizes the KMeans algorithm from the scikit-learn library to cluster the input
    data into the specified number of clusters. The function ensures that the number of clusters
    is constrained between 2 and the number of data points in the input.

    Args:
        x (np.ndarray): A 2D NumPy array representing the data to be clustered.
        k (int): The desired number of clusters.
        seed (int): The random seed for reproducibility of results.

    Returns:
        np.ndarray: An array of cluster labels, where each element corresponds to the cluster index
        assigned to the respective data point.
    """
    kk = max(2, min(int(k), int(x.shape[0])))
    model = KMeans(n_clusters=kk, random_state=int(seed), n_init=10)
    return model.fit_predict(x)


def _cluster_with_kmeans_adaptive(
    x: np.ndarray,
    *,
    k: int,
    seed: int,
    minibatch_over: int,
    minibatch_size: int,
) -> np.ndarray:
    """
    Clusters data using KMeans algorithm, with an adaptive approach for larger datasets.
    If the number of data points exceeds a threshold, MiniBatchKMeans is utilized for
    improved performance on larger datasets. Otherwise, a standard KMeans clustering
    is applied.

    Parameters:
        x (np.ndarray): Input data to be clustered.
        k (int): Target number of clusters.
        seed (int): Random seed for reproducibility.
        minibatch_over (int): Minimum data size to switch to MiniBatchKMeans.
        minibatch_size (int): Batch size for MiniBatchKMeans.

    Returns:
        np.ndarray: Array of cluster labels for each point in the input data.
    """
    n = int(x.shape[0])
    kk = max(2, min(int(k), n))
    if n >= int(max(2, minibatch_over)):
        bs = int(max(256, min(int(minibatch_size), n)))
        model = MiniBatchKMeans(
            n_clusters=kk,
            random_state=int(seed),
            batch_size=bs,
            n_init=3,
            reassignment_ratio=0.01,
        )
        return model.fit_predict(x)
    return _cluster_with_kmeans(x=x, k=kk, seed=seed)


def _estimated_pdist_bytes(n_samples: int) -> int:
    """
    Calculates the estimated memory usage in bytes for storing condensed pairwise
    distances when using scipy.spatial.distance.pdist.

    This calculation is based on the number of samples provided and assumes the
    distances are stored as float64 data type. The result represents the memory
    requirement for an array containing all pairwise distances.

    Parameters:
    n_samples: int
        The number of samples for which pairwise distances will be computed.

    Returns:
    int
        The estimated memory usage in bytes required to store the condensed
        pairwise distances.
    """
    # scipy.spatial.distance.pdist allocates condensed pairwise distances (float64).
    n = int(max(0, n_samples))
    return (n * (n - 1) // 2) * int(np.dtype(np.float64).itemsize)



def _cluster_with_hierarchical(
    x: np.ndarray,
    *,
    distance_threshold: float,
    max_samples: int,
    max_pairwise_gb: float,
) -> np.ndarray:
    """
    Clusters data points using hierarchical clustering.

    This function applies hierarchical clustering on the given data based
    on the specified distance threshold. Hierarchical clustering builds a dendrogram
    based on pairwise distances and allows determination of clusters by cutting
    the tree at a specific distance.

    Attributes representing type parameters must be passed with correct types;
    otherwise, computational or operational errors may arise.

    Raises RuntimeError if either the number of samples exceeds the maximum allowable
    samples for hierarchical clustering or the estimated memory for pairwise distances
    exceeds the configuration limit.

    Parameters:
        x (np.ndarray): The input data matrix of shape (n_samples, n_features).
        distance_threshold (float): The distance threshold to determine cluster formation.
        max_samples (int): The maximum number of samples allowed for hierarchical clustering.
        max_pairwise_gb (float): The maximum allowable memory in gigabytes for pairwise
            distance estimation.

    Returns:
        np.ndarray: Cluster labels for each data point, where each cluster is assigned
            a unique integer starting from 0.

    """
    n = int(x.shape[0])
    if n > int(max_samples):
        raise RuntimeError(
            f"hierarchical skipped: n_samples={n} exceeds hierarchical_max_samples={int(max_samples)}"
        )
    req_bytes = _estimated_pdist_bytes(n)
    req_gb = float(req_bytes) / float(1024 ** 3)
    if req_gb > float(max_pairwise_gb):
        raise RuntimeError(
            "hierarchical skipped: estimated pairwise distance memory "
            f"{req_gb:.2f} GiB exceeds hierarchical_max_pairwise_gb={float(max_pairwise_gb):.2f}"
        )
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=float(distance_threshold))
    return model.fit_predict(x)



def _cluster_with_hdbscan(x: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    Clusters data points using the HDBSCAN clustering algorithm, which is suitable
    for clustering datasets with varying densities. The function applies the
    HDBSCAN algorithm to the input data and returns the cluster labels for each
    data point.

    Parameters:
    x: np.ndarray
        The input data array where rows represent data points and columns
        represent features.
    min_cluster_size: int
        The minimum size of clusters. Cluster sizes smaller than this value
        will not be considered valid clusters.

    Returns:
    np.ndarray
        An array of cluster labels assigned to each data point. The labels
        indicate which cluster each point belongs to, or -1 if the data point
        is considered noise.
    """
    hdbscan = require_hdbscan()
    model = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)))
    return model.fit_predict(x)



def _algorithm_labels(
    *,
    x: np.ndarray,
    config: ConceptDiscoveryConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Generates cluster labels for input data using specified clustering algorithms.

    This function applies various clustering algorithms as defined in the provided
    configuration to the input data. It handles specific parameters for each
    algorithm, processes the data accordingly, and logs any issues encountered
    during execution, such as skipped algorithms or errors.

    Arguments:
        x (np.ndarray): The input data for clustering, represented as a NumPy array.
        config (ConceptDiscoveryConfig): Configuration object specifying the clustering
            algorithms to use and their respective parameters.
        seed (int): Random seed used for reproducibility in stochastic clustering algorithms.

    Returns:
        Dict[str, np.ndarray]: A dictionary where the keys are algorithm names (in lowercase)
        and the values are the corresponding cluster labels as NumPy arrays.

    Raises:
        MemoryError: Raised when a clustering algorithm exceeds available memory.
        RuntimeError: Raised when an algorithm is skipped due to a specific guardrail condition.
        Exception: General exception handling for unexpected failures during clustering processes.
    """
    out: Dict[str, np.ndarray] = {}
    for algo in config.algorithms:
        key = str(algo).lower()
        try:
            if key == "kmeans":
                out[key] = _cluster_with_kmeans_adaptive(
                    x,
                    k=config.kmeans_k,
                    seed=seed,
                    minibatch_over=config.kmeans_minibatch_over,
                    minibatch_size=config.kmeans_minibatch_size,
                )
            elif key == "hierarchical":
                out[key] = _cluster_with_hierarchical(
                    x,
                    distance_threshold=config.hierarchical_distance_threshold,
                    max_samples=config.hierarchical_max_samples,
                    max_pairwise_gb=config.hierarchical_max_pairwise_gb,
                )
            elif key == "hdbscan":
                if has_hdbscan():
                    if int(x.shape[0]) > int(config.hdbscan_max_samples):
                        logger.warning(
                            "hdbscan skipped due sample count",
                            extra={
                                "n_samples": int(x.shape[0]),
                                "hdbscan_max_samples": int(config.hdbscan_max_samples),
                            },
                        )
                        continue
                    out[key] = _cluster_with_hdbscan(x, min_cluster_size=config.hdbscan_min_cluster_size)
                else:
                    logger.warning("hdbscan not installed; skipping hdbscan clustering")
            else:
                logger.warning("Unknown clustering algorithm requested", extra={"algorithm": algo})
        except MemoryError:
            logger.exception(
                "Clustering algorithm failed with MemoryError; skipping",
                extra={"algorithm": key, "n_samples": int(x.shape[0])},
            )
            continue
        except RuntimeError as exc:
            logger.warning(
                "Clustering algorithm skipped by guardrail",
                extra={
                    "algorithm": key,
                    "n_samples": int(x.shape[0]),
                    "reason": str(exc),
                },
            )
            continue
        except Exception:
            logger.exception(
                "Clustering algorithm failed; skipping",
                extra={"algorithm": key, "n_samples": int(x.shape[0])},
            )
            continue
    return out



def _concept_id(layer_name: str, algorithm: str, patch_ids: list[str]) -> str:
    """
    Generates a unique concept identifier based on given layer name, algorithm, and patch IDs.

    This function creates a SHA-1 hash by combining the provided layer name, algorithm, and a sorted list
    of patch IDs in a specific format. The purpose of this identifier is to represent a unique combination
    of these inputs.

    Args:
        layer_name: The name of the layer for which the identifier is being generated.
        algorithm: The algorithm applied to generate or process the concept.
        patch_ids: A list of patch IDs, which will be sorted before hashing.

    Returns:
        A hexadecimal SHA-1 hash string representing the unique identifier of the concept.
    """
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
    """
    Builds concept candidates and their memberships from labeled data.

    This function processes clustering labels and data points to identify meaningful
    concept candidates. Each candidate is generated based on its cluster's coherence,
    support, and centroid, while memberships are computed based on distances to the
    cluster's centroid.

    Parameters:
    x : np.ndarray
        The data points array, where each row represents a point in higher-dimensional space.
    patch_ids : list[str]
        List of strings representing ID for each data point.
    layer_name : str
        The name of the layer in which the clustering applies.
    algorithm : str
        The algorithm used to generate the clustering labels.
    labels : np.ndarray
        Array of clustering labels corresponding to the data points.
    config : ConceptDiscoveryConfig
        Configuration instance specifying constraints like minimum support
        and minimum coherence for concept generation.

    Returns:
    tuple[list[ConceptCandidate], list[ConceptMembership]]
        A tuple containing:
        - A list of concept candidates that satisfy the constraints.
        - A list of memberships detailing the association of data points
          with identified concepts.
    """
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
    """
    Deduplicates a list of concept candidates and their memberships based on a similarity threshold.

    This function processes a list of concept candidates and removes duplicates by evaluating
    their similarity scores. Candidates with similarity scores above the specified threshold
    are grouped together. Their associated memberships are remapped to reflect the remaining
    unique candidates, ensuring consistency. The function outputs the filtered list of candidates
    and memberships.

    Arguments:
        candidates (list[ConceptCandidate]): A list of concept candidate objects to process.
        memberships (list[ConceptMembership]): A list of memberships associated with the candidates.
        similarity_threshold (float): A threshold indicating the acceptable similarity level
            above which candidates are considered duplicates.

    Returns:
        tuple[list[ConceptCandidate], list[ConceptMembership]]: A tuple containing:
            - A list of deduplicated concept candidates.
            - A list of updated concept memberships associated with the deduplicated candidates.
    """
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
    """
    Discover concepts from given embeddings using specified configuration and seed.

    This function performs concept discovery by clustering the provided patch
    embeddings using algorithms specified in the configuration. It verifies that
    all embeddings belong to the same layer, processes the embeddings, applies
    clustering, and deduplicates the resulting concepts based on the given
    criteria. The resulting conceptual structures include candidates, their
    memberships, and additional metadata summarizing the discovery.

    Arguments:
        embeddings: An Iterable of PatchEmbeddingRecord representing patches
            and their corresponding embeddings to be processed.
        config: An instance of ConceptDiscoveryConfig containing configuration
            details for the clustering process.
        seed: An integer seed for ensuring reproducibility during clustering.

    Returns:
        A DiscoveredConceptSet containing discovered concepts, their memberships,
        and metadata summarizing the process.

    Raises:
        ValueError: If embeddings come from different layer_names.
    """
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
    if len(labels_per_algo) == 0:
        logger.warning(
            "No clustering algorithms produced labels; returning empty concept set",
            extra={"n_embeddings": int(len(emb_list)), "algorithms": list(config.algorithms)},
        )
        return DiscoveredConceptSet(
            layer_name=layer_name,
            candidates=[],
            memberships=[],
            metadata={
                "n_embeddings": int(len(emb_list)),
                "n_algorithms": 0,
                "n_candidates_before_dedup": 0,
                "n_candidates_after_dedup": 0,
                "algorithms_requested": [str(a) for a in config.algorithms],
            },
        )

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
