import math
import random
import numpy as np
import torch
import transformers.trainer_utils as trainer_utils


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trainer_utils.set_seed(seed)


def get_theoretical_min_edits(sequence: str, current_gc: float, target_gc: float) -> float:
    """Minimum base edits to cross the target GC threshold."""
    L = len(sequence)
    current_gc_bases = current_gc * L
    target_label = 1 if current_gc < target_gc else 0
    if target_label == 1:
        target_gc_bases = math.ceil(target_gc * L)
        needed = target_gc_bases - current_gc_bases
    else:
        target_gc_bases = math.floor(target_gc * L)
        needed = current_gc_bases - target_gc_bases
    return max(0, needed)


def classify_edits(orig_seq: str, cf_seq: str, target_label: int):
    """Classify base-level edits as meaningful, redundant, or counterproductive.

    Returns:
        (n_meaningful, n_redundant, n_counterproductive)
    """
    n_meaningful = n_redundant = n_counterproductive = 0
    for o, n in zip(orig_seq, cf_seq):
        if o == n:
            continue
        orig_is_gc = o in ('G', 'C')
        new_is_gc = n in ('G', 'C')
        if orig_is_gc == new_is_gc:
            n_redundant += 1
        elif (target_label == 1) == (not orig_is_gc and new_is_gc):
            n_meaningful += 1
        else:
            n_counterproductive += 1
    return n_meaningful, n_redundant, n_counterproductive
