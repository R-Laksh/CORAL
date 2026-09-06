"""Dataset adapters with optional deep-learning dependencies loaded lazily."""
from importlib import import_module

_EXPORTS = {
    "SyntheticGCDataset": ".gc_dataset", "generate_benchmark_sequences": ".gc_dataset",
    "SeqgraDataset": ".seqgra_dataset", "one_hot": ".seqgra_dataset",
    "decode_idx": ".seqgra_dataset", "decode_idx_batch": ".seqgra_dataset",
    "ALPH": ".seqgra_dataset", "ALPH_DECODE": ".seqgra_dataset",
    "load_orientation_neighbourhoods": ".tfbs_mpra",
    "AA_ORDER": ".gb1", "EDIT_POSITIONS": ".gb1", "WT_FULL_SEQUENCE": ".gb1",
    "WT_GENOTYPE": ".gb1", "genotype_from_ids": ".gb1", "genotype_hamming": ".gb1",
    "genotype_to_full_sequence": ".gb1", "hamming_shell": ".gb1",
    "load_gb1_processed": ".gb1", "sequence_to_ids": ".gb1",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
