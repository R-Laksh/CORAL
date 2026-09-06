"""Model adapters with optional heavy dependencies loaded lazily."""
from importlib import import_module

_EXPORTS = {
    "GenomicGCModel": ".gc_model", "train_head": ".gc_model",
    "CNN1D": ".seqgra_models", "DeepSTARR": ".seqgra_models",
    "BPNetCountScore": ".biological", "ConjunctiveConstraint": ".biological",
    "ESMAlphabet": ".biological", "ESMSoftSequenceRegressor": ".biological",
    "FairESMSoftSequenceRegressor": ".biological", "ThresholdConstraint": ".biological",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
