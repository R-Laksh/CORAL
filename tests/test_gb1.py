from pathlib import Path
import pandas as pd

from coral.datasets.gb1 import (
    EDIT_POSITIONS, WT_FULL_SEQUENCE, WT_GENOTYPE,
    genotype_hamming, genotype_to_full_sequence, hamming_shell, load_gb1_processed,
)


def test_gb1_full_sequence_coordinates_match_wild_type():
    assert len(WT_FULL_SEQUENCE) == 56
    assert "".join(WT_FULL_SEQUENCE[p] for p in EDIT_POSITIONS) == WT_GENOTYPE
    assert genotype_to_full_sequence("AAAA")[38] == "A"


def test_gb1_shell_sizes():
    assert len(hamming_shell(WT_GENOTYPE, 1)) == 4 * 19
    assert len(hamming_shell(WT_GENOTYPE, 2)) == 6 * 19 * 19
    assert all(genotype_hamming(WT_GENOTYPE, g) == 2 for g in hamming_shell(WT_GENOTYPE, 2))


def test_gb1_loader_keeps_measured_logfitness(tmp_path: Path):
    p = tmp_path / "gb1.csv"
    pd.DataFrame({
        "Variants": ["VDGV", "ADGV", "AAGV"],
        "LogFitness": [0.0, -0.2, 1.4],
        "phenotype": [9.0, 9.0, 9.0],
    }).to_csv(p, index=False)
    got = load_gb1_processed(p)
    assert got.shape[0] == 3
    assert got.loc[got.genotype == "AAGV", "fitness"].item() == 1.4
