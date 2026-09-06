"""Optimisers with optional deep-learning dependencies loaded lazily."""
from importlib import import_module

_EXPORTS = {
    "CORALOptimizer": ".coral",
    "LearnedGCOptimizer": ".gc", "GCOracleOneHot": ".gc", "LedidiGCOptimizer": ".gc",
    "SeqgraCORALOptimizer": ".seqgra", "SeqgraOracleOneHot": ".seqgra",
    "LedidiSeqgraCFOptimizer": ".seqgra",
    "FiniteEditGraph": ".distributional", "DistributionalCFOptimizer": ".distributional",
    "ALMHTwistedSearch": ".alm_h", "ALMState": ".alm_h", "SearchConfig": ".alm_h",
    "SearchResult": ".alm_h", "augmented_penalty": ".alm_h",
    "STGumbelALMSearch": ".st_alm", "STALMConfig": ".st_alm",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
