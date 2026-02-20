from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MILTrainDataset(Dataset):
    """
    MILTrainDataset is a Dataset class tailored for managing and serving batches of
    data typically used for multiple-instance learning (MIL) tasks. It organizes
    and prepares input data for downstream model training or evaluation.

    The class is designed to handle data comprising multiple feature sets, labels,
    and weights for classification, absorption, and fluorescence tasks. It provides
    flexibility to manage instances, including grouping them into bags, limiting
    instance counts per bag, and ensuring reproducibility through a random seed.

    Attributes:
        ids: List of unique molecule or sample identifiers.
        X2d: Two-dimensional array of features, one row per molecule/sample [N, F2].
        y_cls: Tensor of regression values for the main classification task.
        w_cls: Tensor of weights for the main classification task.
        y_abs: Tensor of regression values for the absorption task.
        m_abs: Tensor of masks for missing absorption values.
        w_abs: Tensor of weights for the absorption task.
        y_fluo: Tensor of regression values for the fluorescence task.
        m_fluo: Tensor of masks for missing fluorescence values.
        w_fluo: Tensor of weights for the fluorescence task.
        starts: Array of start indices for each molecule's instance bag.
        counts: Array of instance counts for each molecule's bag.
        id2pos: Dictionary mapping molecule IDs to positions in the instances dataset.
        Xinst: Sorted instance-level features dataset [total_instances, F3].
        max_instances: Maximum number of instances allowed in each bag.
        rng: Random number generator initialized with the given seed.

    Methods:
        __len__:
            Returns the number of unique molecules/samples in the dataset.

        __getitem__:
            Retrieves data for the molecule/sample at a given index. This includes
            features, instance bags, corresponding labels, and weights for various
            tasks.

    Raises:
        ValueError:
            If the length of the ids list does not match the number of rows in the
            X2d feature array.
    """

    def __init__(
        self,
        ids: List[str],
        X2d: np.ndarray,
        y_cls: np.ndarray,
        w_cls: np.ndarray,
        y_abs: np.ndarray,
        m_abs: np.ndarray,
        w_abs: np.ndarray,
        y_fluo: np.ndarray,
        m_fluo: np.ndarray,
        w_fluo: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
        id2pos: Dict[str, int],
        Xinst_sorted: np.ndarray,
        max_instances: int,
        seed: int,
    ):
        self.ids = [str(x) for x in ids]
        self.X2d = np.asarray(X2d, dtype=np.float32)

        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.w_cls = torch.tensor(w_cls, dtype=torch.float32)
        self.y_abs = torch.tensor(y_abs, dtype=torch.float32)
        self.m_abs = torch.tensor(m_abs, dtype=torch.bool)
        self.w_abs = torch.tensor(w_abs, dtype=torch.float32)
        self.y_fluo = torch.tensor(y_fluo, dtype=torch.float32)
        self.m_fluo = torch.tensor(m_fluo, dtype=torch.bool)
        self.w_fluo = torch.tensor(w_fluo, dtype=torch.float32)

        self.starts = starts
        self.counts = counts
        self.id2pos = id2pos
        self.Xinst = Xinst_sorted

        self.max_instances = int(max_instances)
        self.rng = np.random.default_rng(seed)

        if len(self.ids) != self.X2d.shape[0]:
            raise ValueError(f"MILTrainDataset: len(ids)={len(self.ids)} != X2d rows={self.X2d.shape[0]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        mol_id = self.ids[i]
        p = self.id2pos[mol_id]
        s = int(self.starts[p])
        c = int(self.counts[p])
        bag = self.Xinst[s : s + c]

        if self.max_instances > 0 and bag.shape[0] > self.max_instances:
            idx = self.rng.choice(bag.shape[0], size=self.max_instances, replace=False)
            bag = bag[idx]

        return (
            self.X2d[i],  # np float32 [F2]
            bag,  # np float32 [Ni,F3]
            self.y_cls[i],
            self.w_cls[i],  # torch [4]
            self.y_abs[i],
            self.m_abs[i],
            self.w_abs[i],  # [2]
            self.y_fluo[i],
            self.m_fluo[i],
            self.w_fluo[i],  # [4]
        )


class MILExportDataset(Dataset):
    """
    MILExportDataset represents a multiple instance learning dataset for exporting purposes.

    This class is designed to work with datasets where data is structured in a way compatible
    with multiple-instance models. Each instance corresponds to a bag of data samples associated
    with a unique identifier. The dataset supports operations for accessing individual bags
    of instances, along with their related metadata such as confidence scores and 2D descriptors.

    Attributes:
        ids: A list of unique string identifiers for each bag in the dataset.
        X2d: A NumPy array containing 2D descriptors, one for each bag.
        starts: A NumPy array indicating the start index of each bag in the associated instance array.
        counts: A NumPy array indicating the number of instances in each bag.
        id2pos: A dictionary mapping each unique identifier in ids to its corresponding position.
        Xinst: A sorted NumPy array containing individual instance data.
        conf: A sorted NumPy array containing confidence values for all instances.
        max_instances: An integer limiting the number of instances per bag when retrieving; defaults to 0, meaning no limit.
        rng: A random number generator for sampling instances in bags exceeding the max_instances limit.

    Methods:
        __len__:
            Returns the total number of bags in the dataset.

        __getitem__:
            Retrieves a specific bag and associated metadata by its index.

    Raises:
        ValueError: If there is a mismatch between the length of ids and the number of rows in X2d.
    """

    def __init__(
        self,
        ids: List[str],
        X2d: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
        id2pos: Dict[str, int],
        Xinst_sorted: np.ndarray,
        conf_sorted: np.ndarray,
        max_instances: int = 0,
        seed: int = 0,
    ):
        self.ids = [str(x) for x in ids]
        self.X2d = np.asarray(X2d, dtype=np.float32)
        self.starts = starts
        self.counts = counts
        self.id2pos = id2pos
        self.Xinst = Xinst_sorted
        self.conf = conf_sorted
        self.max_instances = int(max_instances)
        self.rng = np.random.default_rng(seed)

        if len(self.ids) != self.X2d.shape[0]:
            raise ValueError(f"MILExportDataset: len(ids)={len(self.ids)} != X2d rows={self.X2d.shape[0]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[str, List[str], np.ndarray, np.ndarray]:
        mol_id = self.ids[i]
        p = self.id2pos[mol_id]
        s = int(self.starts[p])
        c = int(self.counts[p])

        bag = self.Xinst[s : s + c]
        conf = self.conf[s : s + c].tolist()

        if self.max_instances > 0 and bag.shape[0] > self.max_instances:
            idx = self.rng.choice(bag.shape[0], size=self.max_instances, replace=False)
            bag = bag[idx]
            conf = [conf[j] for j in idx.tolist()]

        return mol_id, conf, self.X2d[i], bag
