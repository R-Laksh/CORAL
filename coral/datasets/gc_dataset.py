import random
from torch.utils.data import Dataset


class SyntheticGCDataset(Dataset):
    """Random DNA sequences with target GC content generated on demand."""

    def __init__(self, num_samples=2000, min_len=48, max_len=600):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        L = random.randint(self.min_len // 6, self.max_len // 6) * 6
        target_gc = random.random()
        seq = []
        for _ in range(L):
            if random.random() < target_gc:
                seq.append(random.choice(["G", "C"]))
            else:
                seq.append(random.choice(["A", "T"]))
        seq = "".join(seq)
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        return seq, gc


def generate_benchmark_sequences(n_sequences: int, seq_length: int,
                                  seed: int = 42, target_gc: float = 0.5):
    """Generate random DNA sequences with diverse GC content."""

    def _generate_sequence(length, gc_bias):
        seq = []
        for _ in range(length):
            if random.random() < gc_bias:
                seq.append(random.choice(["G", "C"]))
            else:
                seq.append(random.choice(["A", "T"]))
        return "".join(seq)

    rng = random.Random(seed)
    sequences = []
    for _ in range(n_sequences):
        gc_bias = rng.random()
        seq = _generate_sequence(seq_length, gc_bias)
        gc_content = (seq.count("G") + seq.count("C")) / len(seq)
        target_label = 1 if gc_content < target_gc else 0
        while ((target_label == 1 and gc_content >= target_gc) or
               (target_label == 0 and gc_content <= target_gc)):
            gc_bias = rng.random()
            seq = _generate_sequence(seq_length, gc_bias)
            gc_content = (seq.count("G") + seq.count("C")) / len(seq)
            target_label = 1 if gc_content < target_gc else 0
        sequences.append(seq)
    return sequences
