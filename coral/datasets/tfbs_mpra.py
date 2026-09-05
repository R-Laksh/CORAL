"""Auditable adapter for Georgakopoulos-Soares et al. (2023) HepG2 MPRA.

Only complete eight-state orientation cubes are retained. TF identities, order,
positions and background are fixed inside a cube. Palindromic/reduced cubes
are excluded, not assigned fictitious strand interventions.
"""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SOURCE_REPOSITORY = "https://github.com/IliasGeoSo/TFBSs_grammar"
SOURCE_COMMIT = "2a65724010cac893e5dd667f198145a7c7c9803f"
SOURCE_DOI = "10.1038/s41467-023-37960-5"
SOURCE_HASHES = {
    "Library_MPRA_TFBSs.txt": "3e3325853cafd64b762be7f501a6f661587910234365c238cfd482646db72bf0",
    "MPRA_TFBSs_1-byInsert.tsv": "f5c4afeea472e168d247b73a4d60500f22061bd806491662db591db4332d7fa9",
    "MPRA_TFBSs_2-byInsert.tsv": "594ce805fa33362a058f7b95e4ec3f8b26d68eaa962da11b7c8dda4fd45810f7",
    "MPRA_TFBSs_3-byInsert.tsv": "6e113ebc05d75abbf4cab9fa8057f82f4a3d3a67a1e8150572568383fadb4af3",
}
# First four matching values in the authors' all_FOXA1_AP1_CREB1 plot data.
# They verify the one-based seq_N -> FASTA record mapping, not just ID bounds.
MAPPING_ANCHORS = {
    123915: -0.009177971368648502, 123916: 0.17963811359374948,
    123917: -0.33342667744512877, 123918: -0.12374994133744968,
}


@dataclass
class OrientationNeighbourhood:
    identifier: str
    background: int
    factors: tuple
    positions: tuple
    distances: tuple
    ids: tuple
    states: np.ndarray
    sequences: tuple
    replicate_log2: np.ndarray
    tags: np.ndarray
    expression: np.ndarray

    @property
    def family(self):
        """All orders, orientations, positions and backgrounds of a TF multiset."""
        return tuple(sorted(self.factors))

    def split(self, seed=20260905):
        value = hashlib.sha256((str(seed) + ":" + ",".join(self.family)).encode()).digest()
        quantile = int.from_bytes(value[:8], "big") / 2**64
        return "train" if quantile < .7 else ("validation" if quantile < .85 else "test")


def _parse_triplet(header, sequence, number):
    match = re.fullmatch(r">Construct([12]) Three Motifs (\S+), (.+)", header)
    if not match:
        return None
    factors = tuple(match[2].rstrip(",").split(","))
    atoms = re.findall(r"(Non-Template|Template):([ACGT]+)", match[3])
    positions = tuple(map(int, re.findall(r"Pos\d+:(\d+)", match[3])))
    distances = tuple(map(int, re.findall(r"Distance\d+:(\d+)", match[3])))
    if len(factors) != 3 or len(atoms) != 3 or len(positions) != 3 or len(distances) != 2:
        raise ValueError(f"malformed triplet design at record {number}")
    if len(sequence) != 230 or set(sequence) - set("ACGT"):
        raise ValueError(f"invalid 230bp sequence at record {number}")
    # Design coordinates exclude the 15bp cloning prefix.
    for (_, motif), position in zip(atoms, positions):
        if sequence[15 + position:15 + position + len(motif)] != motif:
            raise ValueError(f"motif/coordinate mismatch at record {number}")
    return {
        "number": number, "background": int(match[1]), "factors": factors,
        "positions": positions, "distances": distances,
        "state": tuple(int(orientation == "Template") for orientation, _ in atoms),
        "sequence": sequence,
    }


def load_orientation_neighbourhoods(source_dir, min_tags=3, verify_hashes=True):
    """Read independent replicate files and return measured cubes plus provenance.

    The target is log2(mean(RNA/DNA)), matching the authors' plotting values.
    Individual replicate log ratios are also kept for paired endpoint audits.
    No raw counts or unreported barcode-level variances are manufactured.
    """
    if min_tags < 1:
        raise ValueError("min_tags must be positive")
    source = Path(source_dir)
    if (source / "MPRA_library_data").is_dir():
        source = source / "MPRA_library_data"
    hashes = {}
    for name, expected in SOURCE_HASHES.items():
        with (source / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if verify_hashes and hashes[name] != expected:
            raise ValueError(f"source checksum mismatch for {name}; inspect the data release")
    replicates = []
    for number in (1, 2, 3):
        table = pd.read_csv(source / f"MPRA_TFBSs_{number}-byInsert.tsv", sep="\t")
        if set(table.columns) != {"name", "RNA", "DNA", "ratio", "tags"}:
            raise ValueError("unexpected replicate schema")
        if table.name.duplicated().any():
            raise ValueError("duplicate construct IDs in a replicate")
        table = table.set_index("name")
        replicates.append(table)
    for number, expected in MAPPING_ANCHORS.items():
        actual = np.log2(np.mean([r.at[f"seq_{number}", "ratio"] for r in replicates]))
        if not np.isclose(actual, expected, atol=1e-10, rtol=0):
            raise ValueError("replicate/FASTA mapping failed the independent plot-data anchors")

    designs, total_records, triplet_records = defaultdict(list), 0, 0
    with (source / "Library_MPRA_TFBSs.txt").open() as handle:
        while True:
            header = handle.readline().strip()
            if not header:
                break
            sequence = handle.readline().strip()
            total_records += 1
            if not header.startswith(">") or not sequence:
                raise ValueError("expected two-line FASTA records")
            row = _parse_triplet(header, sequence, total_records)
            if row is not None:
                triplet_records += 1
                key = (row["background"], row["factors"], row["positions"], row["distances"])
                designs[key].append(row)
    full_cube = set(itertools.product((0, 1), repeat=3))
    neighbourhoods, incomplete, low_quality = [], 0, 0
    for key, rows in sorted(designs.items()):
        if (len(rows) != 8 or {r["state"] for r in rows} != full_cube
                or len({r["sequence"] for r in rows}) != 8):
            incomplete += 1
            continue
        rows.sort(key=lambda r: r["state"])
        ids = tuple(f"seq_{r['number']}" for r in rows)
        ratios = np.array([r.reindex(ids).ratio.to_numpy() for r in replicates]).T
        tags = np.array([r.reindex(ids).tags.to_numpy() for r in replicates]).T
        if (not np.isfinite(ratios).all() or (ratios <= 0).any()
                or not np.isfinite(tags).all() or (tags < min_tags).any()):
            low_quality += 1
            continue
        identifier = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        neighbourhoods.append(OrientationNeighbourhood(
            identifier, key[0], key[1], key[2], key[3], ids,
            np.array([r["state"] for r in rows]), tuple(r["sequence"] for r in rows),
            np.log2(ratios), tags.astype(int), np.log2(ratios.mean(axis=1)),
        ))
    sequences = [s for n in neighbourhoods for s in n.sequences]
    if len(set(sequences)) != len(sequences):
        raise ValueError("duplicate DNA across retained cubes; group duplicates before splitting")
    rep = np.concatenate([n.replicate_log2 for n in neighbourhoods])
    provenance = {
        "doi": SOURCE_DOI, "repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT,
        "sha256": hashes, "design_records": total_records, "triplet_records": triplet_records,
        "design_neighbourhoods": len(designs), "incomplete_or_reduced_cubes": incomplete,
        "complete_cubes_failing_quality": low_quality,
        "retained_neighbourhoods": len(neighbourhoods), "retained_sequences": len(sequences),
        "min_tags_each_of_three_replicates": min_tags,
        "mapping": "one-based FASTA record index; verified against four author plot values",
        "replicate_pearson_log2": {
            f"{a+1}-{b+1}": float(pearsonr(rep[:, a], rep[:, b]).statistic)
            for a, b in itertools.combinations(range(3), 2)
        },
    }
    return neighbourhoods, provenance


def predictor_features(neighbourhoods, factor_vocabulary):
    """A transparent screening model interface, not a genomic foundation model."""
    lookup = {factor: index for index, factor in enumerate(factor_vocabulary)}
    arrays = []
    for n in neighbourhoods:
        for state in n.states:
            arrays.append([n.background - 1, *[lookup[x] for x in n.factors],
                           *state, *n.positions, *n.distances])
    return np.asarray(arrays, dtype=float)
