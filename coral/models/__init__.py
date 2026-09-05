"""Model adapters; loading one does not import unrelated optional backends."""
from importlib import import_module

_EXPORTS = {
    "GenomicGCModel": ".gc_model", "train_head": ".gc_model",
    "CNN1D": ".seqgra_models", "DeepSTARR": ".seqgra_models",
    "FrozenESM": ".esm", "FrozenESMC": ".esm", "ESMFunctionalPredictor": ".esm",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
