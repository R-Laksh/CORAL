class GrammarChecker:
    """Computes minimum edit distances to/from SeqGra grammar constraints."""

    def __init__(self, grammar: str):
        self.grammar = grammar.lower()
        self.MOTIF_A = "TATAAA"
        self.MOTIF_B = "CCAAT"
        self.LEN_A = len(self.MOTIF_A)
        self.LEN_B = len(self.MOTIF_B)

    def hamming(self, s1, s2):
        if len(s1) != len(s2):
            return float('inf')
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def min_edits_to_plant_motif(self, seq, motif):
        m = len(motif)
        if len(seq) < m:
            return float('inf')
        return min(self.hamming(seq[i:i+m], motif) for i in range(len(seq) - m + 1))

    def count_motif(self, seq, motif):
        count, m, i = 0, len(motif), 0
        while i <= len(seq) - m:
            if seq[i:i+m] == motif:
                count += 1
                i += m
            else:
                i += 1
        return count

    def min_edits_to_break_motif(self, seq, motif):
        return self.count_motif(seq, motif)

    def min_edits_to_AND(self, seq, min_gap=3, max_gap=5):
        best = float('inf')
        L = len(seq)
        for i in range(L - self.LEN_A - self.LEN_B - min_gap + 1):
            cost_A = self.hamming(seq[i:i+self.LEN_A], self.MOTIF_A)
            for gap in range(min_gap, max_gap + 1):
                j = i + self.LEN_A + gap
                if j + self.LEN_B > L:
                    continue
                cost_B = self.hamming(seq[j:j+self.LEN_B], self.MOTIF_B)
                if cost_A + cost_B < best:
                    best = cost_A + cost_B
        return best

    def min_edits_to_break_AND(self, seq, min_gap=3, max_gap=5):
        posA = [i for i in range(len(seq) - self.LEN_A + 1) if seq[i:i+self.LEN_A] == self.MOTIF_A]
        posB = [j for j in range(len(seq) - self.LEN_B + 1) if seq[j:j+self.LEN_B] == self.MOTIF_B]
        edges = [(i, j) for i in posA for j in posB if min_gap <= j - (i + self.LEN_A) <= max_gap]
        if not edges:
            return 0
        active_A = list(set(e[0] for e in edges))
        min_cost = float('inf')
        for k in range(1 << len(active_A)):
            broken_A = set(active_A[idx] for idx in range(len(active_A)) if (k & (1 << idx)))
            broken_B = set(v for u, v in edges if u not in broken_A)
            min_cost = min(min_cost, len(broken_A) + len(broken_B))
        return min_cost

    def edits_for_NIMPLY_pos(self, seq):
        best = float('inf')
        L = len(seq)
        for i in range(L - self.LEN_A + 1):
            c_A = self.hamming(seq[i:i+self.LEN_A], self.MOTIF_A)
            c_break_B = 0
            for gap in range(3, 6):
                j = i + self.LEN_A + gap
                if j + self.LEN_B <= L:
                    if seq[j:j+self.LEN_B] == self.MOTIF_B:
                        c_break_B += 1
            best = min(best, c_A + c_break_B)
        return best

    def edits_for_NIMPLY_neg(self, seq):
        c_no_A = self.min_edits_to_break_motif(seq, self.MOTIF_A)
        posA = [i for i in range(len(seq) - self.LEN_A + 1) if seq[i:i+self.LEN_A] == self.MOTIF_A]
        if not posA:
            return 0
        L = len(seq)
        total_cost = 0
        for i in posA:
            c_plant_B = float('inf')
            for gap in range(3, 6):
                j = i + self.LEN_A + gap
                if j + self.LEN_B <= L:
                    c_plant_B = min(c_plant_B, self.hamming(seq[j:j+self.LEN_B], self.MOTIF_B))
            total_cost += min(1, c_plant_B)
        return min(c_no_A, total_cost)

    def min_edits_to_k_motifs(self, seq, motif, k):
        import numpy as np
        m = len(motif)
        dp = np.full((len(seq) + 1, k + 1), float('inf'))
        dp[0][0] = 0
        for i in range(len(seq)):
            for j in range(k + 1):
                if dp[i][j] != float('inf'):
                    dp[i+1][j] = min(dp[i+1][j], dp[i][j])
                    if i + m <= len(seq) and j < k:
                        cost = self.hamming(seq[i:i+m], motif)
                        dp[i+m][j+1] = min(dp[i+m][j+1], dp[i][j] + cost)
        return dp[len(seq)][k]

    def get_distances(self, seq):
        """Return (d_pos, d_neg): min edits to satisfy / violate the grammar."""
        g = self.grammar
        if g == "not":
            d_pos = self.min_edits_to_break_motif(seq, self.MOTIF_A)
            d_neg = self.min_edits_to_plant_motif(seq, self.MOTIF_A)
        elif g == "or":
            d_pos = min(self.min_edits_to_plant_motif(seq, self.MOTIF_A),
                        self.min_edits_to_plant_motif(seq, self.MOTIF_B))
            d_neg = (self.min_edits_to_break_motif(seq, self.MOTIF_A) +
                     self.min_edits_to_break_motif(seq, self.MOTIF_B))
        elif g in ["and_nand", "and_xor", "dummy"]:
            d_pos = self.min_edits_to_AND(seq)
            d_neg = self.min_edits_to_break_AND(seq)
        elif g == "xor_xnor":
            cA = self.count_motif(seq, self.MOTIF_A)
            cB = self.count_motif(seq, self.MOTIF_B)
            if cA > 0 and cB == 0:
                d_pos = 0
            elif cB > 0 and cA == 0:
                d_pos = 0
            elif cA > 0 and cB > 0:
                d_pos = min(self.min_edits_to_break_motif(seq, self.MOTIF_A),
                            self.min_edits_to_break_motif(seq, self.MOTIF_B))
            else:
                d_pos = min(self.min_edits_to_plant_motif(seq, self.MOTIF_A),
                            self.min_edits_to_plant_motif(seq, self.MOTIF_B))
            d_neg = min(
                self.min_edits_to_break_motif(seq, self.MOTIF_A) + self.min_edits_to_break_motif(seq, self.MOTIF_B),
                self.min_edits_to_AND(seq)
            )
        elif g == "nimply":
            d_pos = self.edits_for_NIMPLY_pos(seq)
            d_neg = self.edits_for_NIMPLY_neg(seq)
        elif g == "count3":
            d_pos = self.min_edits_to_k_motifs(seq, self.MOTIF_A, 3)
            c_A = self.count_motif(seq, self.MOTIF_A)
            d_neg = max(0, c_A - 2)
        else:
            raise ValueError(f"Unknown grammar {g}")
        return float(d_pos), float(d_neg)
