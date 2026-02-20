from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def collate_train(batch):
    """
    Collates a batch of training data into tensors for model input and output. Each batch is composed of
    multiple individual examples, and this function pads and organizes them in an efficient format suitable
    for the model.

    Args:
        batch (list[tuple]): A list of tuples where each tuple contains training data for a specific example.
            The data in each tuple includes:
            - 2D input features as a numpy array (b[0]).
            - 3D input feature bags as a numpy array (b[1]).
            - Class labels as tensors (b[2]).
            - Class label weights as tensors (b[3]).
            - Regression output labels for absorption prediction as tensors (b[4]).
            - Mask for valid absorption labels as tensors (b[5]).
            - Weights for absorption labels as tensors (b[6]).
            - Regression output labels for fluorescence prediction as tensors (b[7]).
            - Mask for valid fluorescence labels as tensors (b[8]).
            - Weights for fluorescence labels as tensors (b[9]).

    Returns:
        tuple: A tuple containing the following tensors:
            - x2d (torch.FloatTensor): Tensor containing 2D features with shape [B, ...].
            - x3d (torch.FloatTensor): Tensor containing padded 3D features with shape [B, max_len, F3].
            - kpm_t (torch.BoolTensor): Tensor indicating key padding mask, where True represents padding, with shape [B, max_len].
            - y_cls (torch.Tensor): Tensor for class labels with shape [B, ...].
            - w_cls (torch.Tensor): Tensor for class label weights with shape [B, ...].
            - y_abs (torch.Tensor): Tensor for regression labels (absorption) with shape [B, ...].
            - m_abs (torch.Tensor): Tensor for masks (valid absorption labels) with shape [B, ...].
            - w_abs (torch.Tensor): Tensor for weights (absorption labels) with shape [B, ...].
            - y_fluo (torch.Tensor): Tensor for regression labels (fluorescence) with shape [B, ...].
            - m_fluo (torch.Tensor): Tensor for masks (valid fluorescence labels) with shape [B, ...].
            - w_fluo (torch.Tensor): Tensor for weights (fluorescence labels) with shape [B, ...].

    Raises:
        None
    """
    B = len(batch)
    x2d_np = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    bags_np = [b[1] for b in batch]
    lens = np.array([int(x.shape[0]) for x in bags_np], dtype=np.int64)
    max_len = int(lens.max())

    F3 = int(bags_np[0].shape[1])
    x3d_pad = np.zeros((B, max_len, F3), dtype=np.float32)
    for i, bag in enumerate(bags_np):
        x3d_pad[i, : bag.shape[0], :] = bag

    # key_padding_mask True=PAD
    kpm = np.ones((B, max_len), dtype=bool)
    for i, L in enumerate(lens):
        kpm[i, :L] = False

    x2d = torch.from_numpy(x2d_np).float()
    x3d = torch.from_numpy(x3d_pad).float()
    kpm_t = torch.from_numpy(kpm)

    y_cls = torch.stack([b[2] for b in batch], dim=0)
    w_cls = torch.stack([b[3] for b in batch], dim=0)

    y_abs = torch.stack([b[4] for b in batch], dim=0)
    m_abs = torch.stack([b[5] for b in batch], dim=0)
    w_abs = torch.stack([b[6] for b in batch], dim=0)

    y_fluo = torch.stack([b[7] for b in batch], dim=0)
    m_fluo = torch.stack([b[8] for b in batch], dim=0)
    w_fluo = torch.stack([b[9] for b in batch], dim=0)

    return x2d, x3d, kpm_t, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo


def collate_export(batch):
    """
    Collates a batch of molecular data into tensors and padded arrays, preparing it for model input. The method processes the batch, extracting molecule IDs, padding conformation lists, and generating 2D and 3D arrays for model consumption. Additionally, a key padding mask is constructed to identify valid and padded entries in the batch.

    Args:
        batch (list[tuple]): A batch of molecular data, where each tuple contains:
            - mol_id (str): Molecule identifier.
            - conf_list (list): A list of conformations associated with the molecule.
            - x2d (numpy.ndarray): A 2D numpy array representation for the molecule.
            - bags_np (numpy.ndarray): 3D conformer data stored as a numpy array.

    Returns:
        tuple:
            - mol_ids (list[str]): List of molecule identifiers.
            - conf_pad (numpy.ndarray): Padded array of molecule conformations.
            - x2d (torch.Tensor): 2D tensor representation of molecules.
            - x3d (torch.Tensor): 3D padded tensor representation of conformers.
            - kpm (torch.Tensor): Boolean tensor serving as a key padding mask, indicating valid (False) and padded (True) entries.
    """
    # returns mol_ids(list[str]), conf_pad(object array), x2d, x3d, kpm
    B = len(batch)
    mol_ids = [b[0] for b in batch]
    conf_lists = [b[1] for b in batch]
    x2d_np = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
    bags_np = [b[3] for b in batch]

    lens = np.array([len(x) for x in conf_lists], dtype=np.int64)
    max_len = int(lens.max())
    F3 = int(bags_np[0].shape[1])

    x3d_pad = np.zeros((B, max_len, F3), dtype=np.float32)
    conf_pad = np.empty((B, max_len), dtype=object)
    conf_pad[:] = ""

    for i in range(B):
        L = int(lens[i])
        x3d_pad[i, :L, :] = bags_np[i]
        conf_pad[i, :L] = conf_lists[i]

    kpm = np.ones((B, max_len), dtype=bool)
    for i, L in enumerate(lens):
        kpm[i, :int(L)] = False

    return (
        mol_ids,
        conf_pad,
        torch.from_numpy(x2d_np).float(),
        torch.from_numpy(x3d_pad).float(),
        torch.from_numpy(kpm),
    )
