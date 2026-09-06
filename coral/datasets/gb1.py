"""Utilities for the four-site GB1 binding-fitness landscape of Wu et al. (2016)."""
from __future__ import annotations

from itertools import combinations, product
import re
from pathlib import Path

import numpy as np
import pandas as pd

AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
WT_GENOTYPE = "VDGV"
WT_FULL_SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
EDIT_POSITIONS = (38, 39, 40, 53)  # zero-based positions 39,40,41,54 in GB1


def load_gb1_processed(path: str | Path) -> pd.DataFrame:
    """Load measured GB1 rows; never substitute an inferred complete landscape."""
    df = pd.read_csv(path)
    if "LogFitness" not in df.columns:
        raise ValueError("Expected measured LogFitness column in processed GB1 table")
    genotype_col = None
    aa4 = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{4}$")
    preferred = ["Variants", "variants", "variant", "Genotype", "genotype"]
    for col in preferred + [c for c in df.columns if c not in preferred]:
        if col not in df.columns:
            continue
        vals = df[col].astype(str)
        if len(vals) and float(vals.map(lambda s: bool(aa4.fullmatch(s))).mean()) > 0.99:
            genotype_col = col
            break
    if genotype_col is None:
        raise ValueError("Could not identify the four-amino-acid GB1 genotype column")
    out = pd.DataFrame({
        "genotype": df[genotype_col].astype(str),
        "fitness": pd.to_numeric(df["LogFitness"], errors="coerce"),
    }).dropna()
    out = out.drop_duplicates("genotype", keep="first").reset_index(drop=True)
    if WT_GENOTYPE not in set(out.genotype):
        raise ValueError("GB1 wild type VDGV is missing from the measured table")
    return out


def genotype_to_full_sequence(genotype: str) -> str:
    if len(genotype) != 4 or any(a not in AA_ORDER for a in genotype):
        raise ValueError(f"Invalid GB1 genotype: {genotype!r}")
    seq = list(WT_FULL_SEQUENCE)
    for pos, aa in zip(EDIT_POSITIONS, genotype):
        seq[pos] = aa
    return "".join(seq)


def sequence_to_ids(sequence: str, aa_order=AA_ORDER) -> np.ndarray:
    lookup = {aa: i for i, aa in enumerate(aa_order)}
    return np.asarray([lookup[aa] for aa in sequence], dtype=np.int64)


def genotype_from_ids(ids, aa_order=AA_ORDER) -> str:
    arr = np.asarray(ids)
    return "".join(aa_order[int(arr[p])] for p in EDIT_POSITIONS)


def genotype_hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def hamming_shell(genotype: str, distance: int, aa_order=AA_ORDER) -> list[str]:
    if not 0 <= distance <= 4:
        raise ValueError("distance must be in [0, 4]")
    if distance == 0:
        return [genotype]
    variants: list[str] = []
    for positions in combinations(range(4), distance):
        choices = [[aa for aa in aa_order if aa != genotype[p]] for p in positions]
        for repl in product(*choices):
            g = list(genotype)
            for p, aa in zip(positions, repl):
                g[p] = aa
            variants.append("".join(g))
    return variants
