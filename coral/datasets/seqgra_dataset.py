from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ALPH = {"A": 0, "C": 1, "G": 2, "T": 3}
ALPH_DECODE = {0: "A", 1: "C", 2: "G", 3: "T"}

def one_hot(seq: str) -> np.ndarray:
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = ALPH.get(ch)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

def decode_idx(idx) -> str:
    return "".join(ALPH_DECODE[int(i)] for i in idx)

def decode_idx_batch(idx_batch) -> list:
    return [decode_idx(row) for row in idx_batch]

class SeqgraDataset(Dataset):
    """Loads a SeqGra .txt file"""

    def __init__(self, txt_path: Path, label_pos: str = "c1"):
        df = pd.read_csv(txt_path, sep="\t")
        self.seqs = df["x"].tolist()
        self.x = np.stack([one_hot(s) for s in self.seqs])
        yraw = df["y"].astype(str).tolist()
        self.y = np.array([1.0 if y == label_pos else 0.0 for y in yraw], dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])
