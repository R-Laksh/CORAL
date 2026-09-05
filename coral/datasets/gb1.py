"""Measured GB1 four-site fitness landscape (Wu et al., eLife 2016).

The pinned public mirror contains measured sequences, not a complete 20**4
hypercube. Missing variants are unknown. Fitness combines survival of folding
and IgG-Fc binding selection, and is normalised to wild type = 1.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
WT = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SITES = np.array([38, 39, 40, 53])  # zero-based, paper: 39/40/41/54
REVISION = "29bd1ffac5427d9fe8862f48f1f24b9fa18dff8c"
SHA256 = "2f115d4eaf03b6083dcc22f7451b3ddfad41c9d8e519286c4e69b6d06db78f1c"
URL = ("https://huggingface.co/datasets/SaProtHub/Dataset-GB1-fitness/resolve/"
       + REVISION + "/dataset.csv")


def encode(sequences):
    lookup = {a: i for i, a in enumerate(ALPHABET)}
    return np.array([[lookup[a] for a in s] for s in sequences], dtype=np.int64)


def decode(ids):
    return ["".join(ALPHABET[int(a)] for a in row) for row in np.asarray(ids)]


def hash_order(sequences, salt="coral-gb1-v1"):
    return np.argsort([hashlib.sha256((salt + s).encode()).hexdigest()
                       for s in sequences], kind="stable")


@dataclass
class GB1Landscape:
    sequences: np.ndarray
    ids: np.ndarray
    fitness: np.ndarray
    mutation_order: np.ndarray

    def split(self, validation_size=512, test_size=512, train_triples=0):
        """Fit on 0--2 mutations, validate on 3, test on 4; no mirror random split."""
        train = np.flatnonzero(self.mutation_order <= 2)
        val = np.flatnonzero(self.mutation_order == 3)
        test = np.flatnonzero(self.mutation_order == 4)
        triples = val[hash_order(self.sequences[val])]
        val = triples[:validation_size]
        if train_triples:
            if train_triples > len(triples) - len(val):
                raise ValueError("Not enough disjoint triple mutants")
            train = np.concatenate((train, triples[validation_size:validation_size + train_triples]))
        test = test[hash_order(self.sequences[test])[:test_size]]
        return train, val, test

    def measured_optimum(self, source, target):
        """Exact minimum edits among *measured* variants meeting raw target.

        This is not a certified optimum for unmeasured sequences or the model.
        """
        feasible = self.fitness >= target
        if not feasible.any():
            return None
        return int(np.count_nonzero(self.ids[feasible][:, SITES] != source[SITES],
                                    axis=1).min())


def load_gb1(path, verify=True):
    path = Path(path)
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("GB1 download does not match the pinned SHA256")
    frame = pd.read_csv(path)
    if len(frame) != 149361 or frame.protein.duplicated().any():
        raise ValueError("Expected 149361 unique measured GB1 variants")
    sequences = frame.protein.to_numpy()
    ids = encode(sequences)
    wild = encode([WT])[0]
    if ids.shape != (149361, 56):
        raise ValueError("Unexpected sequence length")
    if not np.array_equal(np.flatnonzero((ids != wild).any(axis=0)), SITES):
        raise ValueError("Variants outside the four experimental sites")
    fitness = frame.label.to_numpy(dtype=float)
    if not np.isfinite(fitness).all() or (fitness < 0).any():
        raise ValueError("Fitness must be finite and non-negative")
    if fitness[np.flatnonzero(sequences == WT)].tolist() != [1.0]:
        raise ValueError("Wild-type normalisation failed")
    return GB1Landscape(sequences, ids, fitness, (ids != wild).sum(axis=1))
