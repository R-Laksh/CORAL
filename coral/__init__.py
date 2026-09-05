"""CORAL public API, with optional model dependencies loaded on demand."""
from importlib import import_module

_EXPORTS = {
    "GenomicGCModel": ".models.gc_model", "train_head": ".models.gc_model",
    "CNN1D": ".models.seqgra_models", "DeepSTARR": ".models.seqgra_models",
    "SyntheticGCDataset": ".datasets.gc_dataset",
    "generate_benchmark_sequences": ".datasets.gc_dataset",
    "SeqgraDataset": ".datasets.seqgra_dataset", "one_hot": ".datasets.seqgra_dataset",
    "decode_idx": ".datasets.seqgra_dataset", "decode_idx_batch": ".datasets.seqgra_dataset",
    "HammingTableLoss": ".losses.losses", "GCMarginLoss": ".losses.losses",
    "ProbToLogitMarginLoss": ".losses.losses",
    "LearnedGCOptimizer": ".optimizers.gc", "GCOracleOneHot": ".optimizers.gc",
    "LedidiGCOptimizer": ".optimizers.gc",
    "SeqgraCORALOptimizer": ".optimizers.seqgra", "SeqgraOracleOneHot": ".optimizers.seqgra",
    "LedidiSeqgraCFOptimizer": ".optimizers.seqgra",
    "GrammarChecker": ".grammar", "set_seed": ".utils",
    "get_theoretical_min_edits": ".utils", "classify_edits": ".utils",
    "FiniteEditGraph": ".optimizers.distributional",
    "DistributionalCFOptimizer": ".optimizers.distributional",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
